from __future__ import annotations

from dataclasses import asdict
import json
import random
from pathlib import Path
from typing import Any

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
    load_samples,
    observe_3x3,
    save_json,
    solve_boss_battle,
    tile_event,
)


LOW_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
ACTION_TO_ID = {a: i for i, a in enumerate(LOW_ACTIONS)}
ID_TO_ACTION = {i: a for a, i in ACTION_TO_ID.items()}
TILES = ("?", WALL, EMPTY, COIN, TRAP, BOSS, EXIT, "@")
TILE_TO_ID = {t: i for i, t in enumerate(TILES)}


class LowLevelPolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def state_vector(sample: MazeSample, state: PlayerState, max_size: int = 15) -> np.ndarray:
    vec: list[float] = []
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
            one = [0.0] * len(TILES)
            one[TILE_TO_ID.get(ch, 0)] = 1.0
            vec.extend(one)
    vec.extend(
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
        ]
    )
    return np.asarray(vec, dtype=np.float32)


def action_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr, dc = b[0] - a[0], b[1] - a[1]
    for action, delta in MOVES.items():
        if delta == (dr, dc):
            return action
    raise ValueError(f"non-adjacent path step: {a} -> {b}")


def build_bc_examples(samples: list[MazeSample], max_size: int = 15) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    skipped = 0
    for sample in samples:
        path_data = (sample.expert_solution or {}).get("recommended_path") or []
        path = [tuple(p) for p in path_data]
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
            xs.append(state_vector(sample, state, max_size))
            ys.append(ACTION_TO_ID[action])
            apply_move(sample, state, nxt)
            if state.position == sample.boss and not state.boss_defeated:
                boss = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
                if boss.success and state.resource >= sample.boss_config.revive_cost:
                    state.boss_defeated = True
                else:
                    break
            if state.position == sample.end:
                break
    meta = {"examples": len(xs), "skipped_samples_or_steps": skipped, "max_size": max_size}
    return np.stack(xs), np.asarray(ys, dtype=np.int64), meta


def valid_move_mask(sample: MazeSample, state: PlayerState) -> np.ndarray:
    mask = np.zeros(len(LOW_ACTIONS), dtype=np.bool_)
    for action, (dr, dc) in MOVES.items():
        idx = ACTION_TO_ID[action]
        nxt = (state.position[0] + dr, state.position[1] + dc)
        ch = sample.char_at(nxt)
        if ch == WALL:
            continue
        if ch == EXIT and not state.boss_defeated:
            continue
        mask[idx] = True
    return mask


def choose_action(model: LowLevelPolicy, sample: MazeSample, state: PlayerState, max_size: int = 15) -> str | None:
    vec = torch.from_numpy(state_vector(sample, state, max_size)).unsqueeze(0)
    with torch.no_grad():
        logits = model(vec).squeeze(0)
    mask = valid_move_mask(sample, state)
    if not mask.any():
        return None
    mask_t = torch.tensor(mask, dtype=torch.bool)
    return ID_TO_ACTION[int(logits.masked_fill(~mask_t, -1e9).argmax().item())]


def train_bc(
    train_samples: list[MazeSample],
    val_samples: list[MazeSample],
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    max_size: int = 15,
    seed: int = 42,
) -> tuple[LowLevelPolicy, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x_train, y_train, train_meta = build_bc_examples(train_samples, max_size)
    x_val, y_val, val_meta = build_bc_examples(val_samples, max_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LowLevelPolicy(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    metrics: dict[str, Any] = {"device": str(device), "train_meta": train_meta, "val_meta": val_meta, "epochs": []}
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(len(x_train))
        losses = []
        model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = torch.tensor(x_train[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_val, dtype=torch.float32, device=device))
            pred = logits.argmax(dim=1).cpu().numpy()
        val_acc = float((pred == y_val).mean()) if len(y_val) else 0.0
        row = {"epoch": epoch, "loss": sum(losses) / max(1, len(losses)), "val_action_acc": val_acc}
        metrics["epochs"].append(row)
        print(f"epoch={epoch} loss={row['loss']:.4f} val_action_acc={val_acc:.4f}", flush=True)
    metrics["input_dim"] = int(x_train.shape[1])
    metrics["actions"] = list(LOW_ACTIONS)
    metrics["max_size"] = max_size
    return model.cpu().eval(), metrics


def run_bc_strategy(sample: MazeSample, model: LowLevelPolicy, max_size: int = 15, max_steps: int | None = None) -> RunResult:
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    frames = [frame(sample, state, "START", "start", None)]
    max_steps = max_steps or sample.rows * sample.cols * 4
    boss_result = BossResult(False, 0, [], False, state.resource)
    visits: dict[tuple[int, int], int] = {}
    started = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    cpu_start = __import__("time").perf_counter()
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
    runtime_ms = (__import__("time").perf_counter() - cpu_start) * 1000
    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    return RunResult(
        strategy="lowlevel_bc",
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
        runtime_ms=runtime_ms,
        frames=frames,
    )


def save_model(path: str | Path, model: LowLevelPolicy, metrics: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, path)


def load_model(path: str | Path) -> tuple[LowLevelPolicy, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    metrics = payload["metrics"]
    model = LowLevelPolicy(int(metrics["input_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metrics
