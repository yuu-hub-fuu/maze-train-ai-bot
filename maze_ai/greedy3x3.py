"""Greedy real-time resource pickup under 3x3 vision (AI-player task 1).

The player only sees the 3x3 window around itself. At each step it moves to the
adjacent walkable cell with the best "value per unit distance" (all neighbours
are distance 1, so this is just the most valuable visible neighbour: a coin is
+COIN_VALUE, a trap -TRAP_DAMAGE, empty 0). Ties and the no-positive-coin case
fall back to exploring the least-visited cell, which keeps the walk moving and
avoids loops. This is a purely local greedy strategy, distinct from the global
DP planner used for full exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .spec import (
    BOSS,
    COIN,
    COIN_VALUE,
    EXIT,
    MOVES,
    TRAP,
    TRAP_DAMAGE,
    Coord,
    MazeSpec,
)


@dataclass
class GreedyResult:
    picked_value: int
    steps: int
    avg_per_step: float
    coins_collected: int
    traps_triggered: int
    reached_exit: bool
    path: list[Coord]
    frames: list[dict[str, Any]] = field(default_factory=list)


def _neighbor_value(spec: MazeSpec, pos: Coord, collected: set, triggered: set) -> int:
    ch = spec.char_at(pos)
    if ch == COIN and pos not in collected:
        return COIN_VALUE
    if ch == TRAP and pos not in triggered:
        return -TRAP_DAMAGE
    return 0


def greedy_3x3_run(spec: MazeSpec, max_steps: int | None = None) -> GreedyResult:
    max_steps = max_steps or spec.height * spec.width * 4
    pos = spec.start
    picked = 0
    collected: set[Coord] = set()
    triggered: set[Coord] = set()
    visits: dict[Coord, int] = {pos: 1}
    path = [pos]
    frames = [_frame(spec, pos, picked, 0, collected, triggered)]
    reached_exit = False

    for step in range(1, max_steps + 1):
        options = []
        for name, (dr, dc) in MOVES.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if not spec.is_walkable(nxt):
                continue
            val = _neighbor_value(spec, nxt, collected, triggered)
            options.append((val, visits.get(nxt, 0), nxt))
        if not options:
            break
        # Greedy: maximise immediate value, then prefer least-visited (explore).
        positive = [o for o in options if o[0] > 0]
        if positive:
            val, _, nxt = max(positive, key=lambda o: (o[0], -o[1]))
        else:
            # No visible coin: explore least-visited neighbour, avoiding traps.
            non_trap = [o for o in options if o[0] >= 0]
            pool = non_trap or options
            val, _, nxt = min(pool, key=lambda o: (o[1], -o[0]))

        ch = spec.char_at(nxt)
        if ch == COIN and nxt not in collected:
            collected.add(nxt)
            picked += COIN_VALUE
        elif ch == TRAP and nxt not in triggered:
            triggered.add(nxt)
            picked -= TRAP_DAMAGE
        pos = nxt
        visits[pos] = visits.get(pos, 0) + 1
        path.append(pos)
        frames.append(_frame(spec, pos, picked, step, collected, triggered))
        if ch == EXIT:
            reached_exit = True
            break
        # Stop early if everything reachable seems collected and we are looping.
        if visits[pos] > 4 and not positive:
            # wandering; allow continued exploration but cap repeated churn
            if all(visits.get((pos[0] + dr, pos[1] + dc), 0) > 2 for dr, dc in MOVES.values()
                   if spec.is_walkable((pos[0] + dr, pos[1] + dc))):
                break

    steps = len(path) - 1
    return GreedyResult(
        picked_value=picked,
        steps=steps,
        avg_per_step=(picked / steps if steps else 0.0),
        coins_collected=len(collected),
        traps_triggered=len(triggered),
        reached_exit=reached_exit,
        path=path,
        frames=frames,
    )


def _render(spec: MazeSpec, pos: Coord, collected: set, triggered: set) -> list[str]:
    rows = [list(row) for row in spec.grid]
    for (r, c) in collected:
        if rows[r][c] == COIN:
            rows[r][c] = "."
    for (r, c) in triggered:
        if rows[r][c] == TRAP:
            rows[r][c] = "."
    rows[pos[0]][pos[1]] = "@"
    return ["".join(row) for row in rows]


def _frame(spec, pos, picked, step, collected, triggered) -> dict[str, Any]:
    return {
        "step": step,
        "pos": list(pos),
        "picked": picked,
        "grid": _render(spec, pos, collected, triggered),
    }
