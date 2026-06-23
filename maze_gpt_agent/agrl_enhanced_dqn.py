from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import random
import time
from collections import deque

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
    TRAP_DAMAGE,
    WALL,
    BossResult,
    Coord,
    MazeSample,
    PlayerState,
    RunResult,
    Target,
    apply_move,
    frame,
    known_positions,
    observe_3x3,
    solve_boss_battle,
    tile_event,
)
from .agrl_safe_ratio_planner import best_safe_coin, safe_memory_path

ACTIONS = (
    "BEST_COIN",
    "NEAREST_COIN",
    "GO_BOSS",
    "GO_EXIT",
    "EXPLORE_NEAREST",
    "EXPLORE_INFO",
    "EXPLORE_FAR",
)
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
ID_TO_ACTION = {i: a for a, i in ACTION_TO_ID.items()}
TILES = ("?", WALL, EMPTY, COIN, TRAP, BOSS, EXIT, "@")
TILE_TO_ID = {t: i for i, t in enumerate(TILES)}
SCALAR_DIM = 13


class DuelingCNNQ(nn.Module):
    def __init__(self, actions: int = len(ACTIONS), channels: int = len(TILES), scalar_dim: int = SCALAR_DIM):
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
        hidden = 64 * 5 * 5 + scalar_dim
        self.shared = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, actions))

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        x = self.conv(grid).flatten(1)
        h = self.shared(torch.cat([x, scalars], dim=1))
        value = self.value(h)
        adv = self.advantage(h)
        return value + adv - adv.mean(dim=1, keepdim=True)


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
                if ch == COIN and (r, c) in state.collected_coins:
                    ch = EMPTY
                if ch == TRAP and (r, c) in state.triggered_traps:
                    ch = EMPTY
                if ch == BOSS and state.boss_defeated:
                    ch = EMPTY
            grid[TILE_TO_ID.get(ch, 0), r, c] = 1.0
    known_open = sum(1 for ch in state.known.values() if ch != WALL)
    known_frontiers = len(frontier_candidates(sample, state))
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
            known_open / max(1, sample.rows * sample.cols),
            known_frontiers / max(1, sample.rows * sample.cols),
            float(state.resource >= sample.boss_config.revive_cost),
        ],
        dtype=np.float32,
    )
    return grid, scalars


def unknown_adjacent(sample: MazeSample, state: PlayerState, pos: Coord) -> int:
    out = 0
    for dr, dc in MOVES.values():
        nxt = (pos[0] + dr, pos[1] + dc)
        if 0 <= nxt[0] < sample.rows and 0 <= nxt[1] < sample.cols and nxt not in state.known:
            out += 1
    return out


def frontier_candidates(sample: MazeSample, state: PlayerState) -> list[tuple[Coord, list[str], list[Coord], int]]:
    rows = []
    for pos, ch in state.known.items():
        if ch == WALL:
            continue
        unk = unknown_adjacent(sample, state, pos)
        if unk <= 0:
            continue
        found = safe_memory_path(sample, state, pos)
        if found is None:
            continue
        actions, path = found
        if not actions:
            continue
        rows.append((pos, actions, path, unk))
    return rows


def make_target(kind: str, pos: Coord, gain: int, distance: int, risk: int, score: float) -> Target:
    return Target(f"{kind}-{pos[0]}-{pos[1]}", kind, pos, gain, distance, risk, 0, score, True)


def nearest_coin(sample: MazeSample, state: PlayerState):
    best = None
    for coin in known_positions(state, COIN):
        if coin in state.collected_coins:
            continue
        found = safe_memory_path(sample, state, coin)
        if found is None:
            continue
        actions, path = found
        if not actions:
            continue
        trap_loss = sum(TRAP_DAMAGE for p in path[1:] if sample.char_at(p) == TRAP and p not in state.triggered_traps)
        cand = (len(actions), trap_loss, coin, actions, path)
        if best is None or cand < best:
            best = cand
    if best is None:
        return None
    dist, risk, coin, actions, path = best
    return make_target("coin", coin, 50, dist, risk, 50 - dist - risk), actions, path


def choose_frontier(sample: MazeSample, state: PlayerState, mode: str):
    rows = frontier_candidates(sample, state)
    if not rows:
        return None
    if mode == "nearest":
        pos, actions, path, unk = min(rows, key=lambda x: (len(x[1]), -x[3], x[0]))
    elif mode == "info":
        pos, actions, path, unk = max(rows, key=lambda x: (x[3] / max(1, len(x[1])), x[3], -len(x[1])))
    elif mode == "far":
        pos, actions, path, unk = max(rows, key=lambda x: (len(x[1]), x[3]))
    else:
        raise ValueError(mode)
    score = unk * 2.0 - len(actions)
    return make_target("frontier", pos, 0, len(actions), 0, score), actions, path


def target_path_for_action(sample: MazeSample, state: PlayerState, action: str):
    if action == "BEST_COIN":
        return best_safe_coin(sample, state)
    if action == "NEAREST_COIN":
        return nearest_coin(sample, state)
    if action == "GO_BOSS":
        if state.boss_defeated or state.known.get(sample.boss) != BOSS or state.resource < sample.boss_config.revive_cost:
            return None
        found = safe_memory_path(sample, state, sample.boss, allow_boss=True)
        if found is None:
            return None
        actions, path = found
        if not actions:
            return None
        return make_target("boss", sample.boss, 0, len(actions), 0, 0), actions, path
    if action == "GO_EXIT":
        if not state.boss_defeated or state.known.get(sample.end) != EXIT:
            return None
        found = safe_memory_path(sample, state, sample.end, allow_exit=True)
        if found is None:
            return None
        actions, path = found
        if not actions:
            return None
        return make_target("exit", sample.end, 0, len(actions), 0, 0), actions, path
    if action == "EXPLORE_NEAREST":
        return choose_frontier(sample, state, "nearest")
    if action == "EXPLORE_INFO":
        return choose_frontier(sample, state, "info")
    if action == "EXPLORE_FAR":
        return choose_frontier(sample, state, "far")
    return None


def valid_action_mask(sample: MazeSample, state: PlayerState) -> np.ndarray:
    mask = np.zeros(len(ACTIONS), dtype=np.bool_)
    for idx, action in ID_TO_ACTION.items():
        if target_path_for_action(sample, state, action) is not None:
            mask[idx] = True
    return mask


def step_action(sample: MazeSample, state: PlayerState, action: str) -> tuple[float, bool, dict[str, Any]]:
    before_resource = state.resource
    before_steps = state.steps
    before_known = len(state.known)
    chosen = target_path_for_action(sample, state, action)
    if chosen is None:
        return -30.0, False, {"event": "invalid", "action": action}
    target, actions, path = chosen
    if not actions:
        return -30.0, False, {"event": "empty_path", "action": action}
    for pos in path[1:]:
        apply_move(sample, state, pos)
        if state.position == sample.boss and not state.boss_defeated:
            boss = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
            if boss.success and state.resource >= sample.boss_config.revive_cost:
                state.boss_defeated = True
            else:
                state.alive = False
                state.done = True
                break
        if state.position == sample.end:
            state.done = True
            if not state.boss_defeated:
                state.alive = False
            break
    step_cost = state.steps - before_steps
    known_gain = len(state.known) - before_known
    resource_delta = state.resource - before_resource
    reward = resource_delta - 1.8 * step_cost + 0.08 * known_gain
    if action.startswith("EXPLORE"):
        reward += 0.12 * known_gain
    if action.endswith("COIN") and resource_delta > 0:
        reward += 8.0
    done = False
    if not state.alive:
        reward -= 120.0
        done = True
    elif state.position == sample.boss and state.boss_defeated:
        reward += 80.0
    if state.position == sample.end:
        if state.boss_defeated and state.alive:
            reward += 100.0 + 180.0 * (state.resource / max(1, state.steps))
            done = True
        else:
            reward -= 100.0
            done = True
    return reward, done, {"event": "move", "action": action, "target": asdict(target), "known_gain": known_gain}


def teacher_action(sample: MazeSample, state: PlayerState, mask: np.ndarray) -> int | None:
    preferred = []
    if state.boss_defeated and state.known.get(sample.end) == EXIT:
        preferred.append("GO_EXIT")
    if state.known.get(sample.boss) == BOSS and not state.boss_defeated and state.resource >= sample.boss_config.revive_cost:
        preferred.append("GO_BOSS")
    if best_safe_coin(sample, state) is not None:
        preferred.append("BEST_COIN")
    # Unlike the old teacher, prefer high-information exploration before nearest exploration.
    preferred.extend(["EXPLORE_INFO", "EXPLORE_NEAREST", "EXPLORE_FAR"])
    for action in preferred:
        idx = ACTION_TO_ID[action]
        if mask[idx]:
            return idx
    valid = np.flatnonzero(mask)
    return int(valid[0]) if valid.size else None


def masked_argmax(q: torch.Tensor, mask: np.ndarray) -> int:
    if not mask.any():
        return ACTION_TO_ID["EXPLORE_NEAREST"]
    mt = torch.tensor(mask, dtype=torch.bool, device=q.device)
    return int(q.masked_fill(~mt, -1e9).argmax().item())


def train_enhanced_dqn(samples: list[MazeSample], episodes: int = 5000, gamma: float = 0.92, lr: float = 5e-4, batch_size: int = 64, seed: int = 42, teacher_start: float = 0.6, log_every: int = 250):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DuelingCNNQ().to(device)
    target = DuelingCNNQ().to(device)
    target.load_state_dict(policy.state_dict())
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    replay = deque(maxlen=50000)
    rewards = []
    successes = []
    losses = []
    print(f"enhanced_dqn_device={device}", flush=True)
    for ep in range(episodes):
        sample = samples[ep % len(samples)]
        state = PlayerState(position=sample.start, path_history=[sample.start])
        observe_3x3(sample, state)
        progress = ep / max(1, episodes)
        epsilon = max(0.04, 0.45 * (1 - progress))
        teacher_ratio = max(0.0, teacher_start * (1 - progress / 0.55)) if progress < 0.55 else 0.0
        ep_reward = 0.0
        for _ in range(sample.rows * sample.cols):
            grid, scalars = encode_state(sample, state)
            mask = valid_action_mask(sample, state)
            if not mask.any():
                ep_reward -= 50.0
                break
            taught = teacher_action(sample, state, mask)
            if taught is not None and random.random() < teacher_ratio:
                action_id = taught
            elif random.random() < epsilon:
                action_id = int(random.choice(np.flatnonzero(mask)))
            else:
                with torch.no_grad():
                    q = policy(torch.tensor(grid[None], dtype=torch.float32, device=device), torch.tensor(scalars[None], dtype=torch.float32, device=device)).squeeze(0)
                action_id = masked_argmax(q, mask)
            reward, done, _info = step_action(sample, state, ID_TO_ACTION[action_id])
            next_grid, next_scalars = encode_state(sample, state)
            next_mask = valid_action_mask(sample, state) if not done else np.zeros(len(ACTIONS), dtype=np.bool_)
            replay.append((grid, scalars, action_id, reward, next_grid, next_scalars, done, next_mask))
            ep_reward += reward
            if len(replay) >= batch_size:
                batch = random.sample(replay, batch_size)
                grids = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=device)
                scal = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32, device=device)
                acts = torch.tensor([b[2] for b in batch], dtype=torch.long, device=device)
                rs = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)
                ngrids = torch.tensor(np.stack([b[4] for b in batch]), dtype=torch.float32, device=device)
                nscal = torch.tensor(np.stack([b[5] for b in batch]), dtype=torch.float32, device=device)
                dones = torch.tensor([b[6] for b in batch], dtype=torch.float32, device=device)
                nmasks = torch.tensor(np.stack([b[7] for b in batch]), dtype=torch.bool, device=device)
                q = policy(grids, scal).gather(1, acts[:, None]).squeeze(1)
                with torch.no_grad():
                    policy_next = policy(ngrids, nscal).masked_fill(~nmasks, -1e9)
                    next_ids = policy_next.argmax(dim=1)
                    target_next_all = target(ngrids, nscal)
                    has_next = nmasks.any(dim=1)
                    q_next = target_next_all.gather(1, next_ids[:, None]).squeeze(1)
                    q_next = torch.where(has_next, q_next, torch.zeros_like(q_next))
                    y = rs + gamma * (1.0 - dones) * q_next
                loss = F.smooth_l1_loss(q, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            if done:
                break
        success = float(state.alive and state.done and state.position == sample.end and state.boss_defeated)
        rewards.append(ep_reward)
        successes.append(success)
        if ep % 50 == 0:
            target.load_state_dict(policy.state_dict())
        if log_every and (ep + 1) % log_every == 0:
            print(f"episode={ep+1} epsilon={epsilon:.3f} teacher={teacher_ratio:.3f} recent_success={sum(successes[-log_every:])/log_every:.3f} recent_reward={sum(rewards[-log_every:])/log_every:.2f}", flush=True)
    metrics = {
        "episodes": episodes,
        "device": str(device),
        "actions": list(ACTIONS),
        "avg_reward_last_100": sum(rewards[-100:]) / max(1, min(100, len(rewards))),
        "success_last_100": sum(successes[-100:]) / max(1, min(100, len(successes))),
        "avg_loss_last_100": sum(losses[-100:]) / max(1, min(100, len(losses))) if losses else 0.0,
        "teacher_start": teacher_start,
        "architecture": "CNN Dueling Double DQN with expanded exploration action space",
    }
    return policy.cpu().eval(), metrics


def choose_action(model: DuelingCNNQ, sample: MazeSample, state: PlayerState) -> str:
    grid, scalars = encode_state(sample, state)
    with torch.no_grad():
        q = model(torch.tensor(grid[None], dtype=torch.float32), torch.tensor(scalars[None], dtype=torch.float32)).squeeze(0)
    return ID_TO_ACTION[masked_argmax(q, valid_action_mask(sample, state))]


def run_enhanced_strategy(sample: MazeSample, model: DuelingCNNQ, max_decisions: int | None = None) -> RunResult:
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    frames = [frame(sample, state, "START", "start", None)]
    boss_result = BossResult(False, 0, [], False, state.resource)
    max_decisions = max_decisions or sample.rows * sample.cols
    started = time.perf_counter()
    for _ in range(max_decisions):
        if not state.alive or state.done:
            break
        action = choose_action(model, sample, state)
        reward, done, info = step_action(sample, state, action)
        frames.append(frame(sample, state, action, f"reward={reward:.2f};event={info.get('event')};known_gain={info.get('known_gain')}", None))
        if info.get("event") in {"invalid", "empty_path"}:
            state.alive = False
            state.done = True
            break
        if done:
            break
    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    return RunResult(
        strategy="cnn_dueling_double_dqn",
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


def save_model(path: str | Path, model: DuelingCNNQ, metrics: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, path)


def load_model(path: str | Path):
    payload = torch.load(path, map_location="cpu")
    model = DuelingCNNQ()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metrics"]

