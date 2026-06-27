"""Graph utilities over the maze grid.

A course maze is a *perfect maze*: every pair of walkable cells is joined by a
unique simple path, i.e. the walkable cells form a tree. So a single BFS from a
source yields, for every other cell, both the shortest distance and the unique
path (via parent pointers). We use this to build a small "waypoint graph" over
the interesting cells (start, coins, boss, exit) for the DP planner.
"""

from __future__ import annotations

from collections import deque

from .spec import MOVES, MazeSpec, Coord


def neighbors(spec: MazeSpec, pos: Coord):
    for dr, dc in MOVES.values():
        nxt = (pos[0] + dr, pos[1] + dc)
        if spec.is_walkable(nxt):
            yield nxt


def bfs_from(spec: MazeSpec, source: Coord) -> tuple[dict[Coord, int], dict[Coord, Coord]]:
    """Return (distance, parent) maps for every walkable cell reachable from source."""
    dist: dict[Coord, int] = {source: 0}
    parent: dict[Coord, Coord] = {}
    q: deque[Coord] = deque([source])
    while q:
        cur = q.popleft()
        for nxt in neighbors(spec, cur):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                parent[nxt] = cur
                q.append(nxt)
    return dist, parent


def reconstruct_path(parent: dict[Coord, Coord], source: Coord, target: Coord) -> list[Coord] | None:
    """Reconstruct the cell path source..target from BFS parent pointers (inclusive)."""
    if target != source and target not in parent:
        return None
    path = [target]
    cur = target
    while cur != source:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def path_to_moves(path: list[Coord]) -> list[str]:
    """Convert a cell path into UP/DOWN/LEFT/RIGHT move tokens."""
    delta_to_move = {delta: name for name, delta in MOVES.items()}
    moves: list[str] = []
    for a, b in zip(path, path[1:]):
        moves.append(delta_to_move[(b[0] - a[0], b[1] - a[1])])
    return moves


class WaypointGraph:
    """All-pairs shortest paths + concrete cell paths between waypoints.

    Waypoints are: start (index 0), each coin, the boss, and the exit. Distances
    and paths are computed with one BFS per waypoint.
    """

    def __init__(self, spec: MazeSpec, waypoints: list[Coord]):
        self.spec = spec
        self.waypoints = waypoints
        self._dist: dict[Coord, dict[Coord, int]] = {}
        self._parent: dict[Coord, dict[Coord, Coord]] = {}
        for wp in waypoints:
            dist, parent = bfs_from(spec, wp)
            self._dist[wp] = dist
            self._parent[wp] = parent
        # Precompute every waypoint-to-waypoint cell path once (used heavily by
        # the planner, which builds thousands of candidate routes).
        self._path_cache: dict[tuple[Coord, Coord], list[Coord]] = {}
        for a in waypoints:
            for b in waypoints:
                p = reconstruct_path(self._parent[a], a, b)
                if p is not None:
                    self._path_cache[(a, b)] = p

    def dist(self, a: Coord, b: Coord) -> int:
        return self._dist[a].get(b, 10**9)

    def path(self, a: Coord, b: Coord) -> list[Coord]:
        """Unique cell path a..b (inclusive of both endpoints)."""
        p = self._path_cache.get((a, b))
        if p is None:
            p = reconstruct_path(self._parent[a], a, b)
            if p is None:
                raise ValueError(f"no path between {a} and {b}")
            self._path_cache[(a, b)] = p
        return p
