from __future__ import annotations

import random
from pathlib import Path
from typing import Any
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .agrl_core import (
    BOSS,
    COIN,
    EMPTY,
    EXIT,
    MOVES,
    START,
    TRAP,
    WALL,
    BossResult,
    MazeSample,
    PlayerState,
    RunResult,
    apply_move,
    frame,
    observe_3x3,
    solve_boss_battle,
    tile_event,
)
from .agrl_lowlevel_bc import ACTION_TO_ID, ID_TO_ACTION, LOW_ACTIONS, action_between, valid_move_mask

TILES = ("?", WALL, EMPTY, COIN, TRAP, BOSS, EXIT, "@")
TILE_TO_ID = {t: i for i, t in enumerate(TILES)}
SCALAR_DIM = 12


class CNNLowLevelPolicy(nn.Module):
    def __init__(self, channels: int = len(TILES), scalar_dim: int = SCALAR_DIM, output_dim: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 5 * 5 + scalar_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        x = self.conv(grid).flatten(1)
        return self.head(torch.cat([x, scalars], dim=1))


def encode_state(sample: MazeSample, state: PlayerState, max_size: int = 15) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((len(TILES), max_size, max_size), dtype=np.float32)
    for r in range(max_size):
        for c in range(max_size):
            if r >= sample.rows or c >= sample.cols:
                ch = WALL
            elif (r, c) == state.position:
                ch = "@"
            else:
                ch = state.known.get((r, c), "?")
                if ch == START:
                    ch = EMPTY
            grid[TILE_TO_ID.get(ch, 0), r, c] = 1.0
    scalars = np.asarray(
        [
            state.position[0] / max(1, max_size - 1),
            state.position[1] / max(1, max_size - 1),
            sample.rows / max_size,
            sample.cols / max_size,
            state.resource / 300.0,
            state.steps / max(1.0, sample.rows * sample.cols * 2),
            float(state.boss_defeated),
            len(state.collected_coins) / max(1, len(sample.coins)),
            len(state.triggered_traps) / max(1, len(sample.traps)),
            len(state.known) / max(1, sample.rows * sample.cols),
            float(state.resource >= sample.boss_config.revive_cost),
            {"Easy": 0.0, "Medium": 0.33, "Hard": 0.66, "Extreme": 1.0}.get(sample.difficulty, 0.33),
        ],
        dtype=np.float32,
    )
    return grid, scalars


def oracle_path(sample: MazeSample):
    raw = (sample.expert_solution or {}).get("recommended_path") or []
    return [tuple(p) for p in raw if isinstance(p, (list, tuple)) and len(p) == 2]


def build_examples(samples: list[MazeSample], max_size: int = 15):
    grids = []
    scalars = []
    labels = []
    skipped = 0
    for sample in samples:
        path = oracle_path(sample)
        if len(path) < 2:
            skipped += 1
            continue
        state = PlayerState(position=sample.start, path_history=[sample.start])
        observe_3x3(sample, state)
        for cur, nxt in zip(path, path[1:]):
            if state.position != cur:
                skipped += 1
                break
            action = action_between(cur, nxt)
            g, s = encode_state(sample, state, max_size)
            grids.append(g)
            scalars.append(s)
            labels.append(ACTION_TO_ID[action])
            apply_move(sample, state, nxt)
            if state.position == sample.boss and not state.boss_defeated:
                boss = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
                if boss.success and state.resource >= sample.boss_config.revive_cost:
                    state.boss_defeated = True
                else:
                    break
            if state.position == sample.end:
                break
    meta = {"examples": len(labels), "skipped": skipped, "max_size": max_size}
    return np.stack(grids), np.stack(scalars), np.asarray(labels, dtype=np.int64), meta


def train_cnn_bc(train_samples, val_samples, epochs: int = 40, batch_size: int = 512, lr: float = 8e-4, max_size: int = 15, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g_train, s_train, y_train, train_meta = build_examples(train_samples, max_size)
    g_val, s_val, y_val, val_meta = build_examples(val_samples, max_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLowLevelPolicy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    metrics: dict[str, Any] = {"device": str(device), "train_meta": train_meta, "val_meta": val_meta, "epochs": [], "max_size": max_size}
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(len(y_train))
        model.train()
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            gb = torch.tensor(g_train[idx], dtype=torch.float32, device=device)
            sb = torch.tensor(s_train[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            loss = F.cross_entropy(model(gb, sb), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for start in range(0, len(y_val), batch_size):
                gb = torch.tensor(g_val[start:start+batch_size], dtype=torch.float32, device=device)
                sb = torch.tensor(s_val[start:start+batch_size], dtype=torch.float32, device=device)
                pred = model(gb, sb).argmax(dim=1).cpu().numpy()
                correct += int((pred == y_val[start:start+batch_size]).sum())
                total += len(pred)
        val_acc = correct / max(1, total)
        row = {"epoch": epoch, "loss": sum(losses) / max(1, len(losses)), "val_action_acc": val_acc}
        metrics["epochs"].append(row)
        print(f"epoch={epoch} loss={row['loss']:.4f} val_action_acc={val_acc:.4f}", flush=True)
    return model.cpu().eval(), metrics


def choose_action(model: CNNLowLevelPolicy, sample: MazeSample, state: PlayerState, max_size: int = 15) -> str | None:
    g, s = encode_state(sample, state, max_size)
    with torch.no_grad():
        logits = model(torch.tensor(g[None], dtype=torch.float32), torch.tensor(s[None], dtype=torch.float32)).squeeze(0)
    mask = valid_move_mask(sample, state)
    if not mask.any():
        return None
    mask_t = torch.tensor(mask, dtype=torch.bool)
    return ID_TO_ACTION[int(logits.masked_fill(~mask_t, -1e9).argmax().item())]


def run_cnn_bc_strategy(sample: MazeSample, model: CNNLowLevelPolicy, max_size: int = 15, max_steps: int | None = None) -> RunResult:
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    frames = [frame(sample, state, "START", "start", None)]
    max_steps = max_steps or sample.rows * sample.cols * 4
    boss_result = BossResult(False, 0, [], False, state.resource)
    visits = {}
    started = time.perf_counter()
    while state.alive and not state.done and state.steps < max_steps:
        visits[state.position] = visits.get(state.position, 0) + 1
        action = choose_action(model, sample, state, max_size)
        if action is None:
            state.alive = False
            state.done = True
            break
        dr, dc = MOVES[action]
        nxt = (state.position[0] + dr, state.position[1] + dc)
        apply_move(sample, state, nxt)
        frames.append(frame(sample, state, action, tile_event(sample, state, nxt), None))
        if visits.get(state.position, 0) > 8:
            state.alive = False
            state.done = True
            break
        if state.position == sample.boss and not state.boss_defeated:
            boss_result = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
            if boss_result.success and state.resource >= sample.boss_config.revive_cost:
                state.boss_defeated = True
                frames.append(frame(sample, state, "BOSS_FIGHT", "boss_clear:" + ",".join(boss_result.skill_sequence), None))
            else:
                state.alive = False
                state.done = True
                frames.append(frame(sample, state, "BOSS_FIGHT", "boss_fail", None))
        if state.position == sample.end:
            state.done = True
            if not state.boss_defeated:
                state.alive = False
    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    return RunResult(
        strategy="cnn_lowlevel_oracle_bc",
        sample_id=sample.sample_id,
        difficulty=sample.difficulty,
        success=success,
        boss_success=state.boss_defeated,
        final_resource=state.resource,
        total_steps=state.steps,
        final_score=state.resource / max(1, state.steps),
        trap_count=len(state.triggered_traps),
        coin_count=len(state.collected_coins),
        boss_rounds=boss_result.min_rounds,
        runtime_ms=(time.perf_counter() - started) * 1000,
        frames=frames,
    )


def save_model(path: str | Path, model: CNNLowLevelPolicy, metrics: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, path)


def load_model(path: str | Path):
    payload = torch.load(path, map_location="cpu")
    model = CNNLowLevelPolicy()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metrics"]
