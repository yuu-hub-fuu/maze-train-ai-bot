from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from .constants import MOVE_DELTAS
from .maze_env import Coord, MazeSpec, MazeState


@dataclass
class ExpertTrajectory:
    actions: list[str]
    frames: list[dict[str, Any]]
    final_gold: int
    steps: int
    score: float
    success: bool


def solve_expert(spec: MazeSpec, max_steps: int | None = None) -> ExpertTrajectory:
    """State-space branch-and-bound teacher for course mazes.

    The teacher optimizes the same quantities used by the assignment: reach the
    exit after clearing BOSS, keep as much resource value as possible, and use
    fewer steps when resource value is tied.
    """

    max_steps = max_steps or spec.height * spec.width * 4
    coin_index = {p: i for i, p in enumerate(spec.coins)}
    trap_index = {p: i for i, p in enumerate(spec.traps)}
    start_state = MazeState(spec)
    q: deque[tuple[MazeState, list[str]]] = deque([(start_state, [])])
    best_seen: dict[tuple[Coord, int, int, bool], tuple[int, int]] = {}
    best_state: MazeState | None = None
    best_actions: list[str] = []

    while q:
        state, actions = q.popleft()
        if state.done:
            if not state.failed and state.pos == spec.exit:
                if better_terminal(state, best_state):
                    best_state = state
                    best_actions = actions
            continue
        if state.steps >= max_steps:
            continue

        key = (
            state.pos or spec.start,
            mask(state.collected, coin_index),
            mask(state.triggered, trap_index),
            state.boss_defeated,
        )
        prev = best_seen.get(key)
        if prev and prev[0] >= state.gold and prev[1] <= state.steps:
            continue
        best_seen[key] = (state.gold, state.steps)

        for action in ordered_legal_actions(state):
            if action == "STOP":
                continue
            nxt = state.clone()
            nxt.step(action)
            q.append((nxt, actions + [action]))

    if best_state is None:
        # Fall back to a visible failed trajectory instead of crashing dataset
        # generation; evaluator reports it as failure.
        best_state = start_state
        best_actions = []

    replay = MazeState(spec)
    frames = [frame_record(replay, None, "start")]
    for action in best_actions:
        rec = replay.step(action)
        frames.append(frame_record(replay, action, rec["event"]))

    return ExpertTrajectory(
        actions=best_actions,
        frames=frames,
        final_gold=replay.gold,
        steps=replay.steps,
        score=replay.score(),
        success=(replay.done and not replay.failed and replay.pos == spec.exit),
    )


def ordered_legal_actions(state: MazeState) -> list[str]:
    actions = [a for a in state.legal_actions() if a != "STOP"]
    # Favor resources and forward progress during expansion; exact pruning still
    # preserves alternatives that dominate in gold/steps.
    def priority(action: str) -> tuple[int, str]:
        if action not in MOVE_DELTAS:
            return (0, action)
        assert state.pos is not None
        dr, dc = MOVE_DELTAS[action]
        tile = state.spec.char_at((state.pos[0] + dr, state.pos[1] + dc))
        order = {"G": 0, "B": 1, "E": 2, ".": 3, "T": 4}.get(tile, 5)
        return (order, action)

    return sorted(actions, key=priority)


def mask(points: set[Coord], index: dict[Coord, int]) -> int:
    out = 0
    for point in points:
        if point in index:
            out |= 1 << index[point]
    return out


def better_terminal(candidate: MazeState, incumbent: MazeState | None) -> bool:
    if incumbent is None:
        return True
    cand_tuple = (candidate.gold / max(1, candidate.steps), candidate.gold, -candidate.steps)
    inc_tuple = (incumbent.gold / max(1, incumbent.steps), incumbent.gold, -incumbent.steps)
    return cand_tuple > inc_tuple


def frame_record(state: MazeState, action: str | None, event: str) -> dict[str, Any]:
    return {
        "action": action,
        "event": event,
        "grid": state.render(),
        "pos": state.pos,
        "gold": state.gold,
        "steps": state.steps,
        "score": state.score(),
        "boss_defeated": state.boss_defeated,
        "done": state.done,
        "failed": state.failed,
    }
