from __future__ import annotations

from collections import deque
import heapq
from typing import Callable, Iterable

from .constants import MOVE_DELTAS
from .maze_env import Coord, MazeSpec, manhattan


def neighbors(spec: MazeSpec, pos: Coord, boss_defeated: bool = False) -> Iterable[tuple[str, Coord]]:
    for action, (dr, dc) in MOVE_DELTAS.items():
        nxt = (pos[0] + dr, pos[1] + dc)
        if spec.is_walkable(nxt, boss_defeated):
            yield action, nxt


def bfs_path(
    spec: MazeSpec,
    start: Coord,
    goal: Coord,
    boss_defeated: bool = False,
    blocked: set[Coord] | None = None,
) -> list[str] | None:
    blocked = blocked or set()
    q: deque[Coord] = deque([start])
    parent: dict[Coord, tuple[Coord, str]] = {}
    seen = {start}
    while q:
        cur = q.popleft()
        if cur == goal:
            return reconstruct_actions(parent, start, goal)
        for action, nxt in neighbors(spec, cur, boss_defeated):
            if nxt in blocked or nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = (cur, action)
            q.append(nxt)
    return None


def astar_path(
    spec: MazeSpec,
    start: Coord,
    goal: Coord,
    boss_defeated: bool = False,
    cost_fn: Callable[[Coord], float] | None = None,
) -> list[str] | None:
    cost_fn = cost_fn or (lambda _pos: 1.0)
    heap: list[tuple[float, int, Coord]] = [(manhattan(start, goal), 0, start)]
    parent: dict[Coord, tuple[Coord, str]] = {}
    best_g = {start: 0.0}
    counter = 0
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur == goal:
            return reconstruct_actions(parent, start, goal)
        for action, nxt in neighbors(spec, cur, boss_defeated):
            ng = best_g[cur] + cost_fn(nxt)
            if ng >= best_g.get(nxt, float("inf")):
                continue
            best_g[nxt] = ng
            parent[nxt] = (cur, action)
            counter += 1
            heapq.heappush(heap, (ng + manhattan(nxt, goal), counter, nxt))
    return None


def reconstruct_actions(parent: dict[Coord, tuple[Coord, str]], start: Coord, goal: Coord) -> list[str]:
    actions: list[str] = []
    cur = goal
    while cur != start:
        prev, action = parent[cur]
        actions.append(action)
        cur = prev
    actions.reverse()
    return actions


def reachable_cells(spec: MazeSpec, start: Coord | None = None) -> set[Coord]:
    start = start or spec.start
    q: deque[Coord] = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        for _, nxt in neighbors(spec, cur, boss_defeated=True):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen
