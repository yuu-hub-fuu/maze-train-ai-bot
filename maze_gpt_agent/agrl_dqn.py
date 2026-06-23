from __future__ import annotations

from collections import deque
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
    HIGH_LEVEL_ACTIONS,
    TRAP,
    WALL,
    MazeSample,
    PlayerState,
    apply_move,
    choose_target,
    frontier_target,
    known_positions,
    memory_shortest_path,
    observe_3x3,
    rcspp_path,
    solve_boss_battle,
    target_from_high_action,
    vision_3x3,
)
from .agrl_safe_ratio_planner import best_safe_coin, safe_frontier_target, safe_memory_path


ACTION_TO_ID = {name: idx for idx, name in enumerate(HIGH_LEVEL_ACTIONS)}
ID_TO_ACTION = {idx: name for name, idx in ACTION_TO_ID.items()}


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def state_vector(sample: MazeSample, state: PlayerState) -> np.ndarray:
    tile_ids = {WALL: 0, EMPTY: 1, COIN: 2, TRAP: 3, BOSS: 4, EXIT: 5, "@": 6, "?": 7}
    vec: list[float] = []
    for row in vision_3x3(sample, state):
        for ch in row:
            one_hot = [0.0] * len(tile_ids)
            one_hot[tile_ids.get(ch, 7)] = 1.0
            vec.extend(one_hot)
    known_count = max(1, len(state.known))
    total = sample.rows * sample.cols
    nearest_gold = min((len(memory_shortest_path(sample, state, c) or []) for c in known_positions(state, COIN)), default=99)
    boss_dist = len(memory_shortest_path(sample, state, sample.boss) or []) if state.known.get(sample.boss) == BOSS else 99
    exit_dist = len(memory_shortest_path(sample, state, sample.end) or []) if state.known.get(sample.end) == EXIT else 99
    vec.extend(
        [
            state.resource / 300.0,
            state.steps / max(1.0, sample.rows * sample.cols),
            float(state.boss_defeated),
            len(state.collected_coins) / max(1, len(sample.coins)),
            len(state.triggered_traps) / max(1, len(sample.traps)),
            known_count / total,
            nearest_gold / 99.0,
            boss_dist / 99.0,
            exit_dist / 99.0,
            float(state.resource >= sample.boss_config.revive_cost),
            {"Easy": 0.0, "Medium": 0.33, "Hard": 0.66, "Extreme": 1.0}.get(sample.difficulty, 0.33),
        ]
    )
    return np.asarray(vec, dtype=np.float32)


def safe_target_path_from_high_action(sample: MazeSample, state: PlayerState, high_action: str):
    if high_action == "GO_EXIT":
        if not state.boss_defeated or state.known.get(sample.end) != EXIT:
            return None
        found = safe_memory_path(sample, state, sample.end, allow_exit=True)
        if found is None:
            return None
        actions, path = found
        if not actions:
            return None
        target = target_from_high_action(sample, state, high_action)
        return target, actions, path
    if high_action == "GO_BOSS":
        if state.boss_defeated:
            return None
        if state.known.get(sample.boss) != BOSS:
            return None
        if state.resource < sample.boss_config.revive_cost:
            return None
        found = safe_memory_path(sample, state, sample.boss, allow_boss=True)
        if found is None:
            return None
        actions, path = found
        if not actions:
            return None
        target = target_from_high_action(sample, state, high_action)
        return target, actions, path
    if high_action in {"BEST_VALUE_GOLD", "NEAREST_GOLD", "MAIN_PATH_GOLD", "AVOID_TRAP"}:
        coin = best_safe_coin(sample, state)
        if coin is not None:
            target, actions, path = coin
            return target, actions, path
    if high_action == "EXPLORE":
        frontier = safe_frontier_target(sample, state)
        if frontier is not None:
            target, actions, path = frontier
            return target, actions, path
    return None


def step_high_action(sample: MazeSample, state: PlayerState, high_action: str) -> tuple[float, bool, dict[str, Any]]:
    before_resource = state.resource
    before_steps = state.steps
    before_known = len(state.known)
    chosen = safe_target_path_from_high_action(sample, state, high_action)
    if chosen is None:
        return -25.0, False, {"event": "no_safe_target", "action": high_action}
    target, actions, path = chosen
    if not actions:
        return -35.0, False, {"event": "empty_path", "action": high_action, "target": asdict(target) if target else None}
    for pos in path[1:]:
        apply_move(sample, state, pos)
    step_cost = state.steps - before_steps
    known_gain = len(state.known) - before_known
    resource_delta = state.resource - before_resource
    reward = resource_delta - 1.0 * step_cost + 0.10 * known_gain
    if high_action in {"BEST_VALUE_GOLD", "NEAREST_GOLD", "MAIN_PATH_GOLD"}:
        reward += 10.0
    if high_action == "EXPLORE" and known_gain > 0:
        reward += 2.0
    done = False
    if state.position == sample.boss and not state.boss_defeated:
        boss = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
        if boss.success and state.resource >= sample.boss_config.revive_cost:
            state.boss_defeated = True
            reward += 80.0
        else:
            state.alive = False
            state.done = True
            reward -= 100.0
            done = True
    if state.position == sample.end:
        if state.boss_defeated:
            state.done = True
            reward += 100.0 + 80.0 * (state.resource / max(1, state.steps))
            done = True
        else:
            state.alive = False
            state.done = True
            reward -= 80.0
            done = True
    return reward, done, {"event": "move", "action": high_action, "target": asdict(target) if target else None}


def valid_action_mask(sample: MazeSample, state: PlayerState) -> np.ndarray:
    mask = np.zeros(len(HIGH_LEVEL_ACTIONS), dtype=np.bool_)
    for action_id, action in ID_TO_ACTION.items():
        if safe_target_path_from_high_action(sample, state, action) is not None:
            mask[action_id] = True
    return mask


def masked_argmax(values: torch.Tensor, mask: np.ndarray) -> int:
    if not mask.any():
        return ACTION_TO_ID["EXPLORE"]
    mask_t = torch.tensor(mask, dtype=torch.bool, device=values.device)
    return int(values.masked_fill(~mask_t, -1e9).argmax(dim=-1).item())


def _expert_path(sample: MazeSample) -> list[tuple[int, int]]:
    solution = sample.expert_solution or {}
    raw_path = solution.get("recommended_path") or []
    out: list[tuple[int, int]] = []
    for pos in raw_path:
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            out.append((int(pos[0]), int(pos[1])))
    return out


def oracle_guided_action_id(sample: MazeSample, state: PlayerState, mask: np.ndarray) -> int | None:
    solution = sample.expert_solution or {}
    if solution.get("label_source") != "bounded_full_map_ratio_oracle":
        return None
    path = _expert_path(sample)
    if not path:
        return None
    try:
        idx = len(path) - 1 - list(reversed(path)).index(state.position)
    except ValueError:
        return None
    future = path[idx + 1 :]
    if not future:
        return None

    next_coin = None
    next_boss = None
    next_exit = None
    for pos in future:
        ch = sample.char_at(pos)
        if ch == COIN and pos not in state.collected_coins:
            next_coin = pos
            break
        if ch == BOSS and not state.boss_defeated:
            next_boss = pos
            break
        if ch == EXIT:
            next_exit = pos
            break

    preferred: list[str] = []
    if next_coin is not None:
        preferred.extend(["BEST_VALUE_GOLD", "NEAREST_GOLD", "EXPLORE"])
    elif next_boss is not None:
        if state.known.get(sample.boss) == BOSS and state.resource >= sample.boss_config.revive_cost:
            preferred.append("GO_BOSS")
        preferred.extend(["BEST_VALUE_GOLD", "EXPLORE"])
    elif next_exit is not None:
        if state.boss_defeated and state.known.get(sample.end) == EXIT:
            preferred.append("GO_EXIT")
        preferred.append("EXPLORE")
    else:
        preferred.append("EXPLORE")

    for action in preferred:
        action_id = ACTION_TO_ID[action]
        if mask[action_id]:
            return action_id
    return None


def teacher_action_id(sample: MazeSample, state: PlayerState, mask: np.ndarray) -> int | None:
    oracle_action = oracle_guided_action_id(sample, state, mask)
    if oracle_action is not None:
        return oracle_action
    preferred: list[str] = []
    if state.boss_defeated and state.known.get(sample.end) == EXIT:
        preferred.append("GO_EXIT")
    if state.known.get(sample.boss) == BOSS and not state.boss_defeated and state.resource >= sample.boss_config.revive_cost:
        preferred.append("GO_BOSS")
    if best_safe_coin(sample, state) is not None:
        preferred.append("BEST_VALUE_GOLD")
    if safe_frontier_target(sample, state) is not None:
        preferred.append("EXPLORE")
    for action in preferred:
        action_id = ACTION_TO_ID[action]
        if mask[action_id]:
            return action_id
    valid_ids = np.flatnonzero(mask)
    if valid_ids.size:
        return int(valid_ids[0])
    return None


def train_dqn(
    samples: list[MazeSample],
    episodes: int = 1500,
    gamma: float = 0.90,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    log_every: int = 100,
    teacher_start: float = 0.75,
) -> tuple[DQN, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    probe = PlayerState(position=samples[0].start, path_history=[samples[0].start])
    observe_3x3(samples[0], probe)
    input_dim = len(state_vector(samples[0], probe))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dqn_device={device}", flush=True)
    policy = DQN(input_dim, len(HIGH_LEVEL_ACTIONS)).to(device)
    target = DQN(input_dim, len(HIGH_LEVEL_ACTIONS)).to(device)
    target.load_state_dict(policy.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    replay: deque[tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = deque(maxlen=30000)
    rewards: list[float] = []
    successes: list[float] = []
    losses: list[float] = []
    for ep in range(episodes):
        sample = samples[ep % len(samples)]
        state = PlayerState(position=sample.start, path_history=[sample.start])
        observe_3x3(sample, state)
        progress = ep / max(1, episodes)
        epsilon = max(0.03, 0.35 * (1 - progress))
        teacher_ratio = max(0.0, teacher_start * (1 - progress / 0.65)) if progress < 0.65 else 0.0
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
            reward, done, _info = step_high_action(sample, state, action)
            nv = state_vector(sample, state)
            next_mask = valid_action_mask(sample, state) if not done else np.zeros(len(HIGH_LEVEL_ACTIONS), dtype=np.bool_)
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
                    q_next_all = target(next_states).masked_fill(~next_masks, -1e9)
                    has_next = next_masks.any(dim=1)
                    q_next = torch.where(has_next, q_next_all.max(dim=1).values, torch.zeros(batch_size, device=device))
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
        "actions": list(HIGH_LEVEL_ACTIONS),
        "avg_reward_last_100": sum(rewards[-100:]) / max(1, min(100, len(rewards))),
        "success_last_100": sum(successes[-100:]) / max(1, min(100, len(successes))),
        "avg_loss_last_100": sum(losses[-100:]) / max(1, min(100, len(losses))) if losses else 0.0,
        "teacher_start": teacher_start,
        "reward_objective": "terminal reward includes 80 * final_resource / steps; per-step penalty is -1",
    }
    return policy, metrics


def choose_dqn_action(model: DQN, sample: MazeSample, state: PlayerState) -> str:
    vec = torch.from_numpy(state_vector(sample, state)).unsqueeze(0)
    with torch.no_grad():
        mask = valid_action_mask(sample, state)
        if not mask.any():
            return "EXPLORE"
        action_id = masked_argmax(model(vec).squeeze(0), mask)
        return ID_TO_ACTION[action_id]


def save_dqn(path: str | Path, model: DQN, metrics: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, path)


def load_dqn(path: str | Path) -> tuple[DQN, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    input_dim = int(payload["metrics"]["input_dim"])
    model = DQN(input_dim, len(HIGH_LEVEL_ACTIONS))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metrics"]

