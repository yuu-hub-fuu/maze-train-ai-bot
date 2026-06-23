from __future__ import annotations

from collections import deque
from dataclasses import asdict
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .agrl_core import (
    BOSS,
    EXIT,
    HIGH_LEVEL_ACTIONS,
    MOVES,
    TRAP,
    TRAP_DAMAGE,
    WALL,
    MazeSample,
    PlayerState,
    Target,
    apply_move,
    observe_3x3,
    solve_boss_battle,
)
from .agrl_dqn import (
    ACTION_TO_ID as BASE_ACTION_TO_ID,
    ID_TO_ACTION as BASE_ID_TO_ACTION,
    safe_target_path_from_high_action as base_safe_target_path,
    state_vector as base_state_vector,
    teacher_action_id as base_teacher_action_id,
)
from .agrl_safe_ratio_planner import safe_memory_path


FRONTIER_ACTIONS = (
    "NEAREST_GOLD",
    "BEST_VALUE_GOLD",
    "MAIN_PATH_GOLD",
    "GO_BOSS",
    "GO_EXIT",
    "AVOID_TRAP",
    "EXPLORE_NEAREST",
    "EXPLORE_INFO_DENSITY",
    "EXPLORE_CASHOUT_AWARE",
)
ACTION_TO_ID = {name: idx for idx, name in enumerate(FRONTIER_ACTIONS)}
ID_TO_ACTION = {idx: name for name, idx in ACTION_TO_ID.items()}


class FrontierDQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def unknown_adjacent(sample: MazeSample, state: PlayerState, pos: tuple[int, int]) -> int:
    total = 0
    for dr, dc in MOVES.values():
        nxt = (pos[0] + dr, pos[1] + dc)
        if 0 <= nxt[0] < sample.rows and 0 <= nxt[1] < sample.cols and nxt not in state.known:
            total += 1
    return total


def path_trap_loss(sample: MazeSample, state: PlayerState, path: list[tuple[int, int]]) -> int:
    return sum(TRAP_DAMAGE for p in path[1:] if sample.char_at(p) == TRAP and p not in state.triggered_traps)


def frontier_candidates(sample: MazeSample, state: PlayerState):
    out = []
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
        risk = path_trap_loss(sample, state, path)
        target = Target(f"frontier-{pos[0]}-{pos[1]}", "explore", pos, 0, len(actions), risk, 0, unk - len(actions), True)
        out.append((target, actions, path, unk, risk))
    return out


def cashout_distance_after(sample: MazeSample, state: PlayerState, frontier_path: list[tuple[int, int]]) -> int:
    sim = state.clone()
    for pos in frontier_path[1:]:
        apply_move(sample, sim, pos)
    dist = 0
    if not sim.boss_defeated and sim.known.get(sample.boss) == BOSS and sim.resource >= sample.boss_config.revive_cost:
        found = safe_memory_path(sample, sim, sample.boss, allow_boss=True)
        if found is not None:
            dist += len(found[0])
            for pos in found[1][1:]:
                apply_move(sample, sim, pos)
            sim.boss_defeated = True
    if sim.boss_defeated and sim.known.get(sample.end) == EXIT:
        found = safe_memory_path(sample, sim, sample.end, allow_exit=True)
        if found is not None:
            dist += len(found[0])
    return dist


def select_frontier(sample: MazeSample, state: PlayerState, mode: str):
    cands = frontier_candidates(sample, state)
    if not cands:
        return None
    if mode == "nearest":
        return min(cands, key=lambda x: (len(x[1]), -x[3], x[0].position))[:3]
    if mode == "info_density":
        return max(cands, key=lambda x: (x[3] / max(1, len(x[1])), -x[4], -len(x[1])))[:3]
    if mode == "cashout_aware":
        return max(cands, key=lambda x: (2.0 * x[3] - 1.0 * len(x[1]) - 0.35 * cashout_distance_after(sample, state, x[2]) - 0.05 * x[4]))[:3]
    raise ValueError(mode)


def _candidate_stats(sample: MazeSample, state: PlayerState, mode: str) -> list[float]:
    picked = select_frontier(sample, state, mode)
    if picked is None:
        return [0.0, 1.0, 0.0, 0.0, 1.0]
    target, actions, path = picked
    unk = unknown_adjacent(sample, state, target.position)
    risk = path_trap_loss(sample, state, path)
    cash = cashout_distance_after(sample, state, path)
    return [
        1.0,
        min(len(actions), 99) / 99.0,
        unk / 4.0,
        min(risk, 300) / 300.0,
        min(cash, 225) / 225.0,
    ]


def state_vector(sample: MazeSample, state: PlayerState) -> np.ndarray:
    vec = list(base_state_vector(sample, state))
    for mode in ("nearest", "info_density", "cashout_aware"):
        vec.extend(_candidate_stats(sample, state, mode))
    frontier_count = len(frontier_candidates(sample, state))
    vec.extend([
        min(frontier_count, 64) / 64.0,
        float(state.known.get(sample.boss) == BOSS),
        float(state.known.get(sample.end) == EXIT),
        float(state.resource >= sample.boss_config.revive_cost),
    ])
    return np.asarray(vec, dtype=np.float32)


def safe_target_path_from_action(sample: MazeSample, state: PlayerState, action: str):
    if action == "EXPLORE_NEAREST":
        return select_frontier(sample, state, "nearest")
    if action == "EXPLORE_INFO_DENSITY":
        return select_frontier(sample, state, "info_density")
    if action == "EXPLORE_CASHOUT_AWARE":
        return select_frontier(sample, state, "cashout_aware")
    return base_safe_target_path(sample, state, action)


def step_action(sample: MazeSample, state: PlayerState, action: str) -> tuple[float, bool, dict[str, Any]]:
    before_resource = state.resource
    before_steps = state.steps
    before_known = len(state.known)
    chosen = safe_target_path_from_action(sample, state, action)
    if chosen is None:
        return -25.0, False, {"event": "no_safe_target", "action": action}
    target, actions, path = chosen
    if not actions:
        return -35.0, False, {"event": "empty_path", "action": action, "target": asdict(target) if target else None}
    for pos in path[1:]:
        apply_move(sample, state, pos)
    step_cost = state.steps - before_steps
    known_gain = len(state.known) - before_known
    resource_delta = state.resource - before_resource
    reward = resource_delta - step_cost + 0.10 * known_gain
    if action in {"BEST_VALUE_GOLD", "NEAREST_GOLD", "MAIN_PATH_GOLD"}:
        reward += 10.0
    if action.startswith("EXPLORE") and known_gain > 0:
        reward += 2.0 + 0.25 * known_gain
    done = False
    if state.position == sample.boss and not state.boss_defeated:
        boss = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
        if boss.success and state.resource >= sample.boss_config.revive_cost:
            state.boss_defeated = True
            reward += 80.0
        else:
            state.alive = False
            state.done = True
            reward -= 120.0
            done = True
    if state.position == sample.end:
        if state.boss_defeated:
            state.done = True
            reward += 100.0 + 80.0 * (state.resource / max(1, state.steps))
            done = True
        else:
            state.alive = False
            state.done = True
            reward -= 100.0
            done = True
    return reward, done, {"event": "move", "action": action, "target": asdict(target) if target else None}


def valid_action_mask(sample: MazeSample, state: PlayerState) -> np.ndarray:
    mask = np.zeros(len(FRONTIER_ACTIONS), dtype=np.bool_)
    for action_id, action in ID_TO_ACTION.items():
        if safe_target_path_from_action(sample, state, action) is not None:
            mask[action_id] = True
    return mask


def masked_argmax(values: torch.Tensor, mask: np.ndarray) -> int:
    if not mask.any():
        return ACTION_TO_ID["EXPLORE_NEAREST"]
    mask_t = torch.tensor(mask, dtype=torch.bool, device=values.device)
    return int(values.masked_fill(~mask_t, -1e9).argmax(dim=-1).item())


def teacher_action_id(sample: MazeSample, state: PlayerState, mask: np.ndarray) -> int | None:
    base_mask = np.zeros(len(HIGH_LEVEL_ACTIONS), dtype=np.bool_)
    for idx, action in BASE_ID_TO_ACTION.items():
        mapped = "EXPLORE_NEAREST" if action == "EXPLORE" else action
        if mapped in ACTION_TO_ID and mask[ACTION_TO_ID[mapped]]:
            base_mask[idx] = True
    taught = base_teacher_action_id(sample, state, base_mask)
    if taught is None:
        return None
    action = BASE_ID_TO_ACTION[taught]
    mapped = "EXPLORE_NEAREST" if action == "EXPLORE" else action
    if mapped in ACTION_TO_ID and mask[ACTION_TO_ID[mapped]]:
        return ACTION_TO_ID[mapped]
    valid_ids = np.flatnonzero(mask)
    return int(valid_ids[0]) if valid_ids.size else None


def train_frontier_dqn(
    samples: list[MazeSample],
    episodes: int = 6000,
    gamma: float = 0.90,
    lr: float = 8e-4,
    batch_size: int = 64,
    seed: int = 42,
    log_every: int = 100,
    teacher_start: float = 0.70,
) -> tuple[FrontierDQN, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    probe = PlayerState(position=samples[0].start, path_history=[samples[0].start])
    observe_3x3(samples[0], probe)
    input_dim = len(state_vector(samples[0], probe))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"frontier_dqn_device={device}", flush=True)
    policy = FrontierDQN(input_dim, len(FRONTIER_ACTIONS)).to(device)
    target = FrontierDQN(input_dim, len(FRONTIER_ACTIONS)).to(device)
    target.load_state_dict(policy.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    replay: deque[tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = deque(maxlen=50000)
    rewards: list[float] = []
    successes: list[float] = []
    losses: list[float] = []
    action_counts = {name: 0 for name in FRONTIER_ACTIONS}
    for ep in range(episodes):
        sample = samples[ep % len(samples)]
        state = PlayerState(position=sample.start, path_history=[sample.start])
        observe_3x3(sample, state)
        progress = ep / max(1, episodes)
        epsilon = max(0.04, 0.40 * (1 - progress))
        teacher_ratio = max(0.0, teacher_start * (1 - progress / 0.55)) if progress < 0.55 else 0.0
        ep_reward = 0.0
        for _ in range(sample.rows * sample.cols):
            sv = state_vector(sample, state)
            mask = valid_action_mask(sample, state)
            if not mask.any():
                ep_reward -= 30.0
                break
            taught = teacher_action_id(sample, state, mask)
            if taught is not None and random.random() < teacher_ratio:
                action_id = taught
            elif random.random() < epsilon:
                action_id = int(random.choice(np.flatnonzero(mask)))
            else:
                with torch.no_grad():
                    q_values = policy(torch.from_numpy(sv).unsqueeze(0).to(device)).squeeze(0)
                    action_id = masked_argmax(q_values, mask)
            action = ID_TO_ACTION[action_id]
            action_counts[action] += 1
            reward, done, _info = step_action(sample, state, action)
            nv = state_vector(sample, state)
            next_mask = valid_action_mask(sample, state) if not done else np.zeros(len(FRONTIER_ACTIONS), dtype=np.bool_)
            replay.append((sv, action_id, reward, nv, done, next_mask))
            ep_reward += reward
            if len(replay) >= batch_size:
                batch = random.sample(replay, batch_size)
                states = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=device)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
                rs = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
                next_states = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=device)
                dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)
                next_masks = torch.tensor(np.stack([b[5] for b in batch]), dtype=torch.bool, device=device)
                q = policy(states).gather(1, actions[:, None]).squeeze(1)
                with torch.no_grad():
                    next_policy = policy(next_states).masked_fill(~next_masks, -1e9).argmax(dim=1)
                    next_target = target(next_states).gather(1, next_policy[:, None]).squeeze(1)
                    has_next = next_masks.any(dim=1)
                    q_next = torch.where(has_next, next_target, torch.zeros(batch_size, device=device))
                    y = rs + gamma * (1.0 - dones) * q_next
                loss = F.smooth_l1_loss(q, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.item()))
            if done:
                break
        success = float(state.done and state.alive and state.position == sample.end and state.boss_defeated)
        rewards.append(ep_reward)
        successes.append(success)
        if ep % 25 == 0:
            target.load_state_dict(policy.state_dict())
        if log_every and (ep + 1) % log_every == 0:
            recent_success = sum(successes[-log_every:]) / log_every
            recent_reward = sum(rewards[-log_every:]) / log_every
            print(
                f"episode={ep+1} epsilon={epsilon:.3f} teacher={teacher_ratio:.3f} "
                f"recent_success={recent_success:.3f} recent_reward={recent_reward:.2f}",
                flush=True,
            )
    metrics = {
        "episodes": episodes,
        "input_dim": input_dim,
        "device": str(device),
        "actions": list(FRONTIER_ACTIONS),
        "avg_reward_last_100": sum(rewards[-100:]) / max(1, min(100, len(rewards))),
        "success_last_100": sum(successes[-100:]) / max(1, min(100, len(successes))),
        "avg_loss_last_100": sum(losses[-100:]) / max(1, min(100, len(losses))) if losses else 0.0,
        "teacher_start": teacher_start,
        "action_counts": action_counts,
        "architecture": "MLP DQN with explicit frontier option actions and frontier-value state features",
    }
    return policy, metrics


def choose_action(model: FrontierDQN, sample: MazeSample, state: PlayerState) -> str:
    vec = torch.from_numpy(state_vector(sample, state)).unsqueeze(0)
    with torch.no_grad():
        mask = valid_action_mask(sample, state)
        if not mask.any():
            return "EXPLORE_NEAREST"
        action_id = masked_argmax(model(vec).squeeze(0), mask)
        return ID_TO_ACTION[action_id]


def save_frontier_dqn(path: str | Path, model: FrontierDQN, metrics: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, path)


def load_frontier_dqn(path: str | Path) -> tuple[FrontierDQN, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    input_dim = int(payload["metrics"]["input_dim"])
    model = FrontierDQN(input_dim, len(FRONTIER_ACTIONS))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metrics"]