from __future__ import annotations

from collections import deque
from typing import Iterable

from .constants import ACTIONS, BOSS, COIN, EXIT, MOVE_DELTAS, TRAP, WALL
from .maze_env import Coord, MazeState
from .pathfinding import neighbors


def state_features(state: MazeState, target_score: str = "high") -> list[str]:
    assert state.pos is not None
    spec = state.spec
    pos = state.pos
    feats: list[str] = [
        "bias",
        f"target={target_score}",
        f"scenario={spec.scenario}",
        f"gold_bucket={bucket(state.gold, [-60, 0, 50, 100, 150, 200])}",
        f"need_bucket={bucket(spec.boss_cost - state.gold, [-100, 0, 50, 100, 150, 200])}",
        f"step_bucket={bucket(state.steps, [5, 10, 20, 40, 80])}",
        f"boss_defeated={state.boss_defeated}",
    ]
    for idx, row in enumerate(state.vision_3x3()):
        for jdx, ch in enumerate(row):
            feats.append(f"v{idx}{jdx}={ch}")

    for action, (dr, dc) in MOVE_DELTAS.items():
        nxt = (pos[0] + dr, pos[1] + dc)
        ch = state.visible_char(nxt)
        feats.append(f"{action}:tile={ch}")
        if ch != WALL:
            feats.append(f"{action}:legal")
        if ch == COIN:
            feats.append(f"{action}:coin")
        if ch == TRAP:
            feats.append(f"{action}:trap")
        if ch == BOSS:
            feats.append(f"{action}:boss")
        if ch == EXIT:
            feats.append(f"{action}:exit")

    for tile in (COIN, TRAP, BOSS, EXIT):
        d = nearest_distance(state, tile)
        feats.append(f"dist_{tile}={bucket(d if d is not None else 999, [1, 2, 4, 8, 16, 32])}")
    return feats


def action_features(state: MazeState, action: str, target_score: str = "high") -> list[str]:
    base = state_features(state, target_score)
    out = [f"{action}|{feat}" for feat in base]
    out.append(f"act={action}")
    return out


def nearest_distance(state: MazeState, tile: str) -> int | None:
    spec = state.spec
    assert state.pos is not None
    q: deque[tuple[Coord, int]] = deque([(state.pos, 0)])
    seen = {state.pos}
    while q:
        cur, dist = q.popleft()
        if dist > 0 and visible_target(state, cur, tile):
            return dist
        for _, nxt in neighbors(spec, cur, state.boss_defeated):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, dist + 1))
    return None


def visible_target(state: MazeState, pos: Coord, tile: str) -> bool:
    ch = state.spec.char_at(pos)
    if tile == COIN:
        return ch == COIN and pos not in state.collected
    if tile == TRAP:
        return ch == TRAP and pos not in state.triggered
    if tile == BOSS:
        return ch == BOSS and not state.boss_defeated
    return ch == tile


def bucket(value: int | float, cuts: Iterable[int | float]) -> str:
    for cut in cuts:
        if value <= cut:
            return str(cut)
    return "big"


def normalize_action(text: str) -> str:
    text = text.strip().upper()
    for action in ACTIONS:
        if action in text:
            return action
    return "STOP"
