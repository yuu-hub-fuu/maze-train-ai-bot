from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
from collections import deque
import random
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
    Coord,
    MazeSample,
    PlayerState,
    RunResult,
    Target,
    apply_move,
    frame,
    observe_3x3,
    solve_boss_battle,
    tile_event,
)
from .agrl_safe_ratio_planner import safe_memory_path

TILES = ("?", WALL, EMPTY, COIN, TRAP, BOSS, EXIT, "@")
TILE_TO_ID = {t: i for i, t in enumerate(TILES)}
SCALAR_DIM = 11
MAX_SIZE = 15


class WaypointQ(nn.Module):
    def __init__(self, channels: int = len(TILES), scalar_dim: int = SCALAR_DIM, max_size: int = MAX_SIZE):
        super().__init__()
        self.max_size = max_size
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(),
        )
        self.scalar = nn.Sequential(nn.Linear(scalar_dim, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Conv2d(96 + 64 + 2, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
        )
        rr, cc = torch.meshgrid(torch.linspace(0, 1, max_size), torch.linspace(0, 1, max_size), indexing="ij")
        self.register_buffer("coord", torch.stack([rr, cc], dim=0)[None])

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        b = grid.shape[0]
        h = self.encoder(grid)
        s = self.scalar(scalars).view(b, 64, 1, 1).expand(-1, -1, self.max_size, self.max_size)
        coord = self.coord.expand(b, -1, -1, -1)
        q = self.head(torch.cat([h, s, coord], dim=1)).squeeze(1)
        return q.reshape(b, self.max_size * self.max_size)


def encode_state(sample: MazeSample, state: PlayerState, max_size: int = MAX_SIZE):
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
    frontiers = len(candidate_waypoints(sample, state))
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
            frontiers / max(1, sample.rows * sample.cols),
        ],
        dtype=np.float32,
    )
    return grid, scalars


def unknown_adjacent(sample: MazeSample, state: PlayerState, pos: Coord) -> int:
    return sum(
        1
        for dr, dc in MOVES.values()
        if 0 <= pos[0] + dr < sample.rows and 0 <= pos[1] + dc < sample.cols and (pos[0] + dr, pos[1] + dc) not in state.known
    )


def candidate_waypoints(sample: MazeSample, state: PlayerState) -> dict[Coord, tuple[list[str], list[Coord]]]:
    out: dict[Coord, tuple[list[str], list[Coord]]] = {}
    for pos, ch in state.known.items():
        if pos == state.position or ch == WALL:
            continue
        if ch == EXIT and not state.boss_defeated:
            continue
        if ch == BOSS and state.boss_defeated:
            continue
        if ch == BOSS and state.resource < sample.boss_config.revive_cost:
            continue
        if ch not in (COIN, BOSS, EXIT) and unknown_adjacent(sample, state, pos) <= 0:
            continue
        found = safe_memory_path(sample, state, pos, allow_boss=(ch == BOSS), allow_exit=(ch == EXIT))
        if found is None:
            continue
        actions, path = found
        if actions:
            out[pos] = (actions, path)
    return out


def mask_from_candidates(cands: dict[Coord, tuple[list[str], list[Coord]]], max_size: int = MAX_SIZE) -> np.ndarray:
    mask = np.zeros(max_size * max_size, dtype=np.bool_)
    for r, c in cands:
        if 0 <= r < max_size and 0 <= c < max_size:
            mask[r * max_size + c] = True
    return mask


def target_score(sample: MazeSample, state: PlayerState, pos: Coord, actions: list[str], path: list[Coord]) -> float:
    ch = state.known.get(pos, EMPTY)
    gain = 0.0
    if ch == COIN and pos not in state.collected_coins:
        gain += 50.0
    if ch == BOSS and not state.boss_defeated:
        gain += 80.0
    if ch == EXIT and state.boss_defeated:
        gain += 100.0 + 120.0 * state.resource / max(1, state.steps + len(actions))
    if unknown_adjacent(sample, state, pos) > 0:
        gain += 2.0 * unknown_adjacent(sample, state, pos)
    trap_loss = sum(30 for p in path[1:] if sample.char_at(p) == TRAP and p not in state.triggered_traps)
    return gain - trap_loss - 1.4 * len(actions)


def heuristic_waypoint(sample: MazeSample, state: PlayerState, cands: dict[Coord, tuple[list[str], list[Coord]]]) -> Coord | None:
    if not cands:
        return None
    return max(cands, key=lambda p: target_score(sample, state, p, cands[p][0], cands[p][1]))


def choose_waypoint(model: WaypointQ, sample: MazeSample, state: PlayerState) -> Coord | None:
    cands = candidate_waypoints(sample, state)
    if not cands:
        return None
    grid, scalars = encode_state(sample, state)
    with torch.no_grad():
        q = model(torch.tensor(grid[None], dtype=torch.float32), torch.tensor(scalars[None], dtype=torch.float32)).squeeze(0)
    mask = torch.tensor(mask_from_candidates(cands), dtype=torch.bool)
    idx = int(q.masked_fill(~mask, -1e9).argmax().item())
    return (idx // MAX_SIZE, idx % MAX_SIZE)


def step_waypoint(sample: MazeSample, state: PlayerState, pos: Coord) -> tuple[float, bool, dict[str, Any]]:
    cands = candidate_waypoints(sample, state)
    if pos not in cands:
        return -40.0, False, {"event": "invalid_waypoint", "pos": list(pos)}
    before_resource = state.resource
    before_steps = state.steps
    before_known = len(state.known)
    actions, path = cands[pos]
    for nxt in path[1:]:
        apply_move(sample, state, nxt)
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
    reward = (state.resource - before_resource) - 1.2 * step_cost + 0.4 * known_gain
    done = False
    if not state.alive:
        reward -= 120.0
        done = True
    if state.position == sample.end:
        if state.alive and state.boss_defeated:
            reward += 120.0 + 160.0 * state.resource / max(1, state.steps)
            done = True
        else:
            reward -= 120.0
            done = True
    return reward, done, {"event": "waypoint", "pos": list(pos), "path_len": len(actions), "known_gain": known_gain}


def train_waypoint_q(samples: list[MazeSample], episodes: int = 3000, gamma: float = 0.92, lr: float = 5e-4, batch_size: int = 64, seed: int = 45, teacher_start: float = 0.7, log_every: int = 150):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = WaypointQ().to(device)
    target = WaypointQ().to(device)
    target.load_state_dict(policy.state_dict())
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    replay = deque(maxlen=50000)
    rewards, successes, losses = [], [], []
    print(f"waypoint_q_device={device}", flush=True)
    for ep in range(episodes):
        sample = samples[ep % len(samples)]
        state = PlayerState(position=sample.start, path_history=[sample.start])
        observe_3x3(sample, state)
        progress = ep / max(1, episodes)
        epsilon = max(0.04, 0.40 * (1 - progress))
        teacher_ratio = max(0.0, teacher_start * (1 - progress / 0.6)) if progress < 0.6 else 0.0
        ep_reward = 0.0
        for _ in range(sample.rows * sample.cols):
            cands = candidate_waypoints(sample, state)
            if not cands:
                ep_reward -= 50
                break
            grid, scalars = encode_state(sample, state)
            mask = mask_from_candidates(cands)
            teacher = heuristic_waypoint(sample, state, cands)
            if teacher is not None and random.random() < teacher_ratio:
                pos = teacher
            elif random.random() < epsilon:
                pos = random.choice(list(cands.keys()))
            else:
                with torch.no_grad():
                    q = policy(torch.tensor(grid[None], dtype=torch.float32, device=device), torch.tensor(scalars[None], dtype=torch.float32, device=device)).squeeze(0)
                mt = torch.tensor(mask, dtype=torch.bool, device=device)
                idx = int(q.masked_fill(~mt, -1e9).argmax().item())
                pos = (idx // MAX_SIZE, idx % MAX_SIZE)
            action_idx = pos[0] * MAX_SIZE + pos[1]
            reward, done, _ = step_waypoint(sample, state, pos)
            next_grid, next_scalars = encode_state(sample, state)
            next_mask = mask_from_candidates(candidate_waypoints(sample, state)) if not done else np.zeros(MAX_SIZE * MAX_SIZE, dtype=np.bool_)
            replay.append((grid, scalars, action_idx, reward, next_grid, next_scalars, done, next_mask))
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
                    pn = policy(ngrids, nscal).masked_fill(~nmasks, -1e9)
                    next_ids = pn.argmax(dim=1)
                    tn = target(ngrids, nscal).gather(1, next_ids[:, None]).squeeze(1)
                    has_next = nmasks.any(dim=1)
                    tn = torch.where(has_next, tn, torch.zeros_like(tn))
                    y = rs + gamma * (1 - dones) * tn
                loss = F.smooth_l1_loss(q, y)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0); opt.step()
                losses.append(float(loss.detach().cpu()))
            if done:
                break
        success = float(state.alive and state.done and state.position == sample.end and state.boss_defeated)
        rewards.append(ep_reward); successes.append(success)
        if ep % 50 == 0:
            target.load_state_dict(policy.state_dict())
        if log_every and (ep + 1) % log_every == 0:
            print(f"episode={ep+1} epsilon={epsilon:.3f} teacher={teacher_ratio:.3f} recent_success={sum(successes[-log_every:])/log_every:.3f} recent_reward={sum(rewards[-log_every:])/log_every:.2f}", flush=True)
    metrics = {
        "episodes": episodes,
        "device": str(device),
        "action_space": "225 coordinate waypoints over remembered reachable cells",
        "success_last_100": sum(successes[-100:]) / max(1, min(100, len(successes))),
        "avg_reward_last_100": sum(rewards[-100:]) / max(1, min(100, len(rewards))),
        "avg_loss_last_100": sum(losses[-100:]) / max(1, min(100, len(losses))) if losses else 0.0,
    }
    return policy.cpu().eval(), metrics


def run_waypoint_strategy(sample: MazeSample, model: WaypointQ) -> RunResult:
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    frames = [frame(sample, state, "START", "start", None)]
    boss_result = BossResult(False, 0, [], False, state.resource)
    started = time.perf_counter()
    for _ in range(sample.rows * sample.cols):
        if not state.alive or state.done:
            break
        pos = choose_waypoint(model, sample, state)
        if pos is None:
            state.alive = False; state.done = True; break
        reward, done, info = step_waypoint(sample, state, pos)
        frames.append(frame(sample, state, "WAYPOINT", f"reward={reward:.2f};{info}", None))
        if info.get("event") == "invalid_waypoint":
            state.alive = False; state.done = True; break
        if done:
            break
    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    return RunResult(
        strategy="waypoint_q",
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


def save_model(path: str | Path, model: WaypointQ, metrics: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, path)


def load_model(path: str | Path):
    payload = torch.load(path, map_location="cpu")
    model = WaypointQ()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metrics"]
