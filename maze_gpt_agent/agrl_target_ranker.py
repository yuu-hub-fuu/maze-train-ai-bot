from __future__ import annotations

from dataclasses import asdict
import json
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
    START,
    TRAP,
    TRAP_DAMAGE,
    WALL,
    BossResult,
    Coord,
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
from .agrl_safe_ratio_planner import safe_memory_path


CAND_TYPES = (EMPTY, COIN, TRAP, BOSS, EXIT, "FRONTIER")
CAND_TYPE_TO_ID = {x: i for i, x in enumerate(CAND_TYPES)}


class TargetRanker(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.LayerNorm(192),
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def oracle_path(sample: MazeSample) -> list[Coord]:
    raw = (sample.expert_solution or {}).get("recommended_path") or []
    return [(int(p[0]), int(p[1])) for p in raw if isinstance(p, (list, tuple)) and len(p) == 2]


def unknown_adjacent(sample: MazeSample, state: PlayerState, pos: Coord) -> int:
    total = 0
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nxt = (pos[0] + dr, pos[1] + dc)
        if 0 <= nxt[0] < sample.rows and 0 <= nxt[1] < sample.cols and nxt not in state.known:
            total += 1
    return total


def candidate_type(ch: str, unk: int) -> str:
    if ch == START:
        ch = EMPTY
    if ch in (COIN, TRAP, BOSS, EXIT):
        return ch
    if unk > 0:
        return "FRONTIER"
    return EMPTY


def path_stats(sample: MazeSample, state: PlayerState, path: list[Coord]) -> tuple[int, int]:
    trap_loss = 0
    coin_gain = 0
    for pos in path[1:]:
        ch = sample.char_at(pos)
        if ch == TRAP and pos not in state.triggered_traps:
            trap_loss += TRAP_DAMAGE
        elif ch == COIN and pos not in state.collected_coins:
            coin_gain += 50
    return coin_gain, trap_loss


def feature_for_candidate(sample: MazeSample, state: PlayerState, pos: Coord, actions: list[str], path: list[Coord]) -> np.ndarray:
    ch = state.known.get(pos, EMPTY)
    unk = unknown_adjacent(sample, state, pos)
    ctype = candidate_type(ch, unk)
    one = [0.0] * len(CAND_TYPES)
    one[CAND_TYPE_TO_ID[ctype]] = 1.0
    coin_gain, trap_loss = path_stats(sample, state, path)
    dr = pos[0] - state.position[0]
    dc = pos[1] - state.position[1]
    dist = len(actions)
    vec = []
    vec.extend(one)
    vec.extend(
        [
            state.position[0] / max(1, sample.rows - 1),
            state.position[1] / max(1, sample.cols - 1),
            pos[0] / max(1, sample.rows - 1),
            pos[1] / max(1, sample.cols - 1),
            dr / max(1, sample.rows),
            dc / max(1, sample.cols),
            dist / max(1, sample.rows * sample.cols),
            coin_gain / 300.0,
            trap_loss / 300.0,
            unk / 4.0,
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


def enumerate_candidates(sample: MazeSample, state: PlayerState, max_candidates: int = 96) -> list[dict[str, Any]]:
    rows = []
    for pos, ch in state.known.items():
        if pos == state.position or ch == WALL:
            continue
        if ch == EXIT and not state.boss_defeated:
            continue
        if ch == BOSS and state.boss_defeated:
            continue
        if ch == BOSS and state.resource < sample.boss_config.revive_cost:
            continue
        found = safe_memory_path(sample, state, pos, allow_boss=(ch == BOSS), allow_exit=(ch == EXIT))
        if found is None:
            continue
        actions, path = found
        if not actions:
            continue
        unk = unknown_adjacent(sample, state, pos)
        ctype = candidate_type(ch, unk)
        priority = 0 if ctype in (COIN, BOSS, EXIT) else 1
        rows.append(
            {
                "pos": pos,
                "actions": actions,
                "path": path,
                "feature": feature_for_candidate(sample, state, pos, actions, path),
                "priority": priority,
                "distance": len(actions),
                "type": ctype,
            }
        )
    rows.sort(key=lambda x: (x["priority"], x["distance"], x["pos"]))
    return rows[:max_candidates]


def best_oracle_candidate_index(path: list[Coord], path_index: int, candidates: list[dict[str, Any]]) -> int | None:
    future_rank = {pos: idx for idx, pos in enumerate(path[path_index + 1 :], start=path_index + 1)}
    best = None
    for i, cand in enumerate(candidates):
        pos = cand["pos"]
        if pos not in future_rank:
            continue
        score = future_rank[pos]
        if best is None or score > best[0]:
            best = (score, i)
    return None if best is None else int(best[1])


def build_rank_examples(samples: list[MazeSample], negatives_per_positive: int = 6, seed: int = 42) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = random.Random(seed)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    groups = 0
    skipped = 0
    for sample in samples:
        path = oracle_path(sample)
        if len(path) < 2:
            skipped += 1
            continue
        state = PlayerState(position=sample.start, path_history=[sample.start])
        observe_3x3(sample, state)
        for idx, nxt in enumerate(path[1:], start=0):
            if state.position != path[idx]:
                skipped += 1
                break
            candidates = enumerate_candidates(sample, state)
            label = best_oracle_candidate_index(path, idx, candidates)
            if label is not None:
                groups += 1
                xs.append(candidates[label]["feature"])
                ys.append(1.0)
                neg_ids = [i for i in range(len(candidates)) if i != label]
                rng.shuffle(neg_ids)
                for ni in neg_ids[:negatives_per_positive]:
                    xs.append(candidates[ni]["feature"])
                    ys.append(0.0)
            else:
                skipped += 1
            apply_move(sample, state, nxt)
            if state.position == sample.boss and not state.boss_defeated:
                boss = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
                if boss.success and state.resource >= sample.boss_config.revive_cost:
                    state.boss_defeated = True
                else:
                    break
            if state.position == sample.end:
                break
    meta = {"examples": len(xs), "groups": groups, "skipped_steps": skipped, "negatives_per_positive": negatives_per_positive}
    return np.stack(xs), np.asarray(ys, dtype=np.float32), meta


def train_ranker(train_samples: list[MazeSample], val_samples: list[MazeSample], epochs: int = 20, batch_size: int = 1024, lr: float = 1e-3, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x_train, y_train, train_meta = build_rank_examples(train_samples, seed=seed)
    x_val, y_val, val_meta = build_rank_examples(val_samples, seed=seed + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TargetRanker(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    metrics: dict[str, Any] = {"device": str(device), "input_dim": int(x_train.shape[1]), "train_meta": train_meta, "val_meta": val_meta, "epochs": []}
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(1.0, y_train.sum())], dtype=torch.float32, device=device)
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(len(x_train))
        losses = []
        model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = torch.tensor(x_train[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[idx], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_val, dtype=torch.float32, device=device)).cpu().numpy()
        pred = (logits > 0).astype(np.float32)
        acc = float((pred == y_val).mean()) if len(y_val) else 0.0
        row = {"epoch": epoch, "loss": sum(losses) / max(1, len(losses)), "val_binary_acc": acc}
        metrics["epochs"].append(row)
        print(f"epoch={epoch} loss={row['loss']:.4f} val_binary_acc={acc:.4f}", flush=True)
    return model.cpu().eval(), metrics


def choose_target(model: TargetRanker, sample: MazeSample, state: PlayerState):
    candidates = enumerate_candidates(sample, state)
    if not candidates:
        return None
    x = torch.tensor(np.stack([c["feature"] for c in candidates]), dtype=torch.float32)
    with torch.no_grad():
        scores = model(x).numpy()
    best = int(scores.argmax())
    return candidates[best], float(scores[best])


def run_ranker_strategy(sample: MazeSample, model: TargetRanker, max_decisions: int | None = None) -> RunResult:
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    frames = [frame(sample, state, "START", "start", None)]
    max_decisions = max_decisions or sample.rows * sample.cols
    boss_result = BossResult(False, 0, [], False, state.resource)
    visits: dict[Coord, int] = {}
    started = time.perf_counter()
    for _ in range(max_decisions):
        if not state.alive or state.done:
            break
        choice = choose_target(model, sample, state)
        if choice is None:
            state.alive = False
            state.done = True
            break
        cand, score = choice
        state.decision_history.append({"action": "RANK_TARGET", "target": {"position": list(cand["pos"]), "type": cand["type"], "score": score}, "path_len": len(cand["actions"])})
        for action, pos in zip(cand["actions"], cand["path"][1:]):
            visits[pos] = visits.get(pos, 0) + 1
            apply_move(sample, state, pos)
            frames.append(frame(sample, state, action, tile_event(sample, state, pos), None))
            if visits[pos] > 8:
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
                break
            if state.position == sample.end:
                state.done = True
                if not state.boss_defeated:
                    state.alive = False
                break
        if state.done:
            break
    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    return RunResult(
        strategy="target_ranker_oracle",
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


def save_model(path: str | Path, model: TargetRanker, metrics: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, path)


def load_model(path: str | Path) -> tuple[TargetRanker, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    metrics = payload["metrics"]
    model = TargetRanker(int(metrics["input_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metrics
