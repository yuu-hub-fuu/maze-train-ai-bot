from __future__ import annotations

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
    apply_move,
    frame,
    observe_3x3,
    solve_boss_battle,
    tile_event,
)
from .agrl_safe_ratio_planner import best_safe_coin, safe_memory_path

TILES = ("?", WALL, EMPTY, COIN, TRAP, BOSS, EXIT, "@")
TILE_TO_ID = {t: i for i, t in enumerate(TILES)}
MAX_SIZE = 15
GRID_ACTIONS = MAX_SIZE * MAX_SIZE
CASH_BOSS = GRID_ACTIONS
CASH_EXIT = GRID_ACTIONS + 1
ACTION_DIM = GRID_ACTIONS + 2
SCALAR_DIM = 14


class OptionQ(nn.Module):
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
        self.shared = nn.Sequential(nn.Linear(96 * max_size * max_size + 64, 256), nn.ReLU())
        self.option_head = nn.Linear(256, 2)
        self.grid_head = nn.Sequential(
            nn.Conv2d(96 + 64 + 2, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
        )
        rr, cc = torch.meshgrid(torch.linspace(0, 1, max_size), torch.linspace(0, 1, max_size), indexing="ij")
        self.register_buffer("coord", torch.stack([rr, cc], dim=0)[None])

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        b = grid.shape[0]
        h = self.encoder(grid)
        s = self.scalar(scalars)
        s_map = s.view(b, 64, 1, 1).expand(-1, -1, self.max_size, self.max_size)
        coord = self.coord.expand(b, -1, -1, -1)
        grid_q = self.grid_head(torch.cat([h, s_map, coord], dim=1)).reshape(b, GRID_ACTIONS)
        global_h = self.shared(torch.cat([h.flatten(1), s], dim=1))
        option_q = self.option_head(global_h)
        return torch.cat([grid_q, option_q], dim=1)


def unknown_adjacent(sample: MazeSample, state: PlayerState, pos: Coord) -> int:
    count = 0
    for dr, dc in MOVES.values():
        nxt = (pos[0] + dr, pos[1] + dc)
        if 0 <= nxt[0] < sample.rows and 0 <= nxt[1] < sample.cols and nxt not in state.known:
            count += 1
    return count


def waypoint_candidates(sample: MazeSample, state: PlayerState) -> dict[Coord, tuple[list[str], list[Coord]]]:
    out: dict[Coord, tuple[list[str], list[Coord]]] = {}
    for pos, ch in state.known.items():
        if pos == state.position or ch == WALL:
            continue
        # BOSS/EXIT are handled by explicit cashout option heads.
        if ch in (BOSS, EXIT):
            continue
        useful = ch == COIN or unknown_adjacent(sample, state, pos) > 0
        if not useful:
            continue
        found = safe_memory_path(sample, state, pos)
        if found is None:
            continue
        actions, path = found
        if actions:
            out[pos] = (actions, path)
    return out


def can_cash_boss(sample: MazeSample, state: PlayerState) -> bool:
    if state.boss_defeated:
        return False
    if state.known.get(sample.boss) != BOSS:
        return False
    if state.resource < sample.boss_config.revive_cost:
        return False
    found = safe_memory_path(sample, state, sample.boss, allow_boss=True)
    return found is not None and bool(found[0])


def can_cash_exit(sample: MazeSample, state: PlayerState) -> bool:
    if not state.boss_defeated:
        return False
    if state.known.get(sample.end) != EXIT:
        return False
    found = safe_memory_path(sample, state, sample.end, allow_exit=True)
    return found is not None and bool(found[0])


def action_mask(sample: MazeSample, state: PlayerState) -> np.ndarray:
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    for r, c in waypoint_candidates(sample, state):
        if 0 <= r < MAX_SIZE and 0 <= c < MAX_SIZE:
            mask[r * MAX_SIZE + c] = True
    if can_cash_boss(sample, state):
        mask[CASH_BOSS] = True
    if can_cash_exit(sample, state):
        mask[CASH_EXIT] = True
    return mask


def encode_state(sample: MazeSample, state: PlayerState):
    grid = np.zeros((len(TILES), MAX_SIZE, MAX_SIZE), dtype=np.float32)
    for r in range(MAX_SIZE):
        for c in range(MAX_SIZE):
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
    candidates = waypoint_candidates(sample, state)
    scalars = np.asarray(
        [
            state.position[0] / max(1, MAX_SIZE - 1),
            state.position[1] / max(1, MAX_SIZE - 1),
            state.resource / 300.0,
            state.steps / max(1.0, sample.rows * sample.cols * 2),
            float(state.boss_defeated),
            len(state.collected_coins) / max(1, len(sample.coins)),
            len(state.triggered_traps) / max(1, len(sample.traps)),
            len(state.known) / max(1, sample.rows * sample.cols),
            len(candidates) / max(1, sample.rows * sample.cols),
            float(can_cash_boss(sample, state)),
            float(can_cash_exit(sample, state)),
            float(state.resource >= sample.boss_config.revive_cost),
            sample.rows / MAX_SIZE,
            sample.cols / MAX_SIZE,
        ],
        dtype=np.float32,
    )
    return grid, scalars


def waypoint_score(sample: MazeSample, state: PlayerState, pos: Coord, actions: list[str], path: list[Coord]) -> float:
    ch = state.known.get(pos, EMPTY)
    gain = 0.0
    if ch == COIN and pos not in state.collected_coins:
        gain += 50.0
    gain += 2.0 * unknown_adjacent(sample, state, pos)
    trap_loss = sum(30 for p in path[1:] if sample.char_at(p) == TRAP and p not in state.triggered_traps)
    # Ratio-aware local utility: value must justify travel distance.
    return gain - trap_loss - 1.6 * len(actions)


def heuristic_action(sample: MazeSample, state: PlayerState, mask: np.ndarray) -> int | None:
    # Termination is explicit and intentionally gets priority when available.
    if mask[CASH_EXIT]:
        return CASH_EXIT
    if mask[CASH_BOSS]:
        # If enough resources and boss known, stop wandering and cash into BOSS.
        return CASH_BOSS
    cands = waypoint_candidates(sample, state)
    if not cands:
        valid = np.flatnonzero(mask)
        return int(valid[0]) if valid.size else None
    best = max(cands, key=lambda p: waypoint_score(sample, state, p, cands[p][0], cands[p][1]))
    return best[0] * MAX_SIZE + best[1]


def execute_action(sample: MazeSample, state: PlayerState, action_id: int) -> tuple[float, bool, dict[str, Any]]:
    before_resource = state.resource
    before_steps = state.steps
    before_known = len(state.known)
    if action_id == CASH_BOSS:
        found = safe_memory_path(sample, state, sample.boss, allow_boss=True) if can_cash_boss(sample, state) else None
        label = "cash_boss"
    elif action_id == CASH_EXIT:
        found = safe_memory_path(sample, state, sample.end, allow_exit=True) if can_cash_exit(sample, state) else None
        label = "cash_exit"
    else:
        pos = (action_id // MAX_SIZE, action_id % MAX_SIZE)
        found = waypoint_candidates(sample, state).get(pos)
        label = "waypoint"
    if found is None:
        return -50.0, False, {"event": "invalid", "action_id": action_id}
    actions, path = found
    if not actions:
        return -50.0, False, {"event": "empty", "action_id": action_id}
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
    reward = resource_delta - 1.25 * step_cost + 0.25 * known_gain
    if label.startswith("cash"):
        reward += 25.0
    done = False
    if not state.alive:
        reward -= 120.0
        done = True
    if state.position == sample.end:
        if state.alive and state.boss_defeated:
            reward += 120.0 + 160.0 * (state.resource / max(1, state.steps))
            done = True
        else:
            reward -= 120.0
            done = True
    return reward, done, {"event": label, "action_id": action_id, "path_len": len(actions), "known_gain": known_gain}


def masked_argmax(q: torch.Tensor, mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    mt = torch.tensor(mask, dtype=torch.bool, device=q.device)
    return int(q.masked_fill(~mt, -1e9).argmax().item())


def train_option_q(samples: list[MazeSample], episodes: int = 2500, gamma: float = 0.92, lr: float = 5e-4, batch_size: int = 64, seed: int = 46, teacher_start: float = 0.75, log_every: int = 125):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = OptionQ().to(device)
    target = OptionQ().to(device)
    target.load_state_dict(policy.state_dict())
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    replay = deque(maxlen=50000)
    rewards, successes, losses = [], [], []
    print(f"option_q_device={device}", flush=True)
    for ep in range(episodes):
        sample = samples[ep % len(samples)]
        state = PlayerState(position=sample.start, path_history=[sample.start])
        observe_3x3(sample, state)
        progress = ep / max(1, episodes)
        epsilon = max(0.04, 0.38 * (1 - progress))
        teacher_ratio = max(0.0, teacher_start * (1 - progress / 0.6)) if progress < 0.6 else 0.0
        ep_reward = 0.0
        for _ in range(sample.rows * sample.cols):
            mask = action_mask(sample, state)
            if not mask.any():
                ep_reward -= 50.0
                break
            grid, scalars = encode_state(sample, state)
            taught = heuristic_action(sample, state, mask)
            if taught is not None and random.random() < teacher_ratio:
                action_id = taught
            elif random.random() < epsilon:
                action_id = int(random.choice(np.flatnonzero(mask)))
            else:
                with torch.no_grad():
                    q = policy(torch.tensor(grid[None], dtype=torch.float32, device=device), torch.tensor(scalars[None], dtype=torch.float32, device=device)).squeeze(0)
                action_id = masked_argmax(q, mask)
            reward, done, _info = execute_action(sample, state, action_id)
            ng, ns = encode_state(sample, state)
            nmask = action_mask(sample, state) if not done else np.zeros(ACTION_DIM, dtype=np.bool_)
            replay.append((grid, scalars, action_id, reward, ng, ns, done, nmask))
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
                    y = rs + gamma * (1.0 - dones) * tn
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
        "action_space": "225 waypoint coordinates plus CASH_BOSS and CASH_EXIT option heads",
        "success_last_100": sum(successes[-100:]) / max(1, min(100, len(successes))),
        "avg_reward_last_100": sum(rewards[-100:]) / max(1, min(100, len(rewards))),
        "avg_loss_last_100": sum(losses[-100:]) / max(1, min(100, len(losses))) if losses else 0.0,
    }
    return policy.cpu().eval(), metrics


def choose_action(model: OptionQ, sample: MazeSample, state: PlayerState) -> int | None:
    mask = action_mask(sample, state)
    if not mask.any():
        return None
    grid, scalars = encode_state(sample, state)
    with torch.no_grad():
        q = model(torch.tensor(grid[None], dtype=torch.float32), torch.tensor(scalars[None], dtype=torch.float32)).squeeze(0)
    return masked_argmax(q, mask)


def run_option_strategy(sample: MazeSample, model: OptionQ) -> RunResult:
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    frames = [frame(sample, state, "START", "start", None)]
    boss_result = BossResult(False, 0, [], False, state.resource)
    started = time.perf_counter()
    for _ in range(sample.rows * sample.cols):
        if not state.alive or state.done:
            break
        action_id = choose_action(model, sample, state)
        if action_id is None:
            state.alive = False; state.done = True; break
        reward, done, info = execute_action(sample, state, action_id)
        if action_id == CASH_BOSS:
            label = "CASH_BOSS"
        elif action_id == CASH_EXIT:
            label = "CASH_EXIT"
        else:
            label = f"WAYPOINT({action_id // MAX_SIZE},{action_id % MAX_SIZE})"
        frames.append(frame(sample, state, label, f"reward={reward:.2f};{info}", None))
        if info.get("event") in {"invalid", "empty"}:
            state.alive = False; state.done = True; break
        if done:
            break
    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    return RunResult(
        strategy="option_q_termination_aware",
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


def save_model(path: str | Path, model: OptionQ, metrics: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, path)


def load_model(path: str | Path):
    payload = torch.load(path, map_location="cpu")
    model = OptionQ()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metrics"]
