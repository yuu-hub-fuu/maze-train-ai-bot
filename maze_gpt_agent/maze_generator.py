from __future__ import annotations

from collections import deque
import random

from .constants import BOSS, COIN, EMPTY, EXIT, START, TRAP, WALL
from .maze_env import Coord, MazeSpec
from .pathfinding import bfs_path, neighbors

SCENARIOS = (
    "shortest",
    "on_path_gold",
    "detour_gold",
    "avoid_trap",
    "trap_required",
    "boss_gate",
)


def generate_course_maze(size: int = 11, seed: int | None = None, scenario: str = "mixed") -> MazeSpec:
    rng = random.Random(seed)
    if size < 7:
        raise ValueError("size must be at least 7")
    if size % 2 == 0:
        size += 1
    scenario = rng.choice(SCENARIOS) if scenario == "mixed" else scenario
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")

    rows = [[WALL for _ in range(size)] for _ in range(size)]
    start = (1, 1)
    rows[start[0]][start[1]] = EMPTY
    stack = [start]
    dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
    while stack:
        cur = stack[-1]
        cand: list[Coord] = []
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nxt = (cur[0] + dr, cur[1] + dc)
            if 1 <= nxt[0] < size - 1 and 1 <= nxt[1] < size - 1 and rows[nxt[0]][nxt[1]] == WALL:
                cand.append(nxt)
        if not cand:
            stack.pop()
            continue
        nxt = cand[0]
        wall = ((cur[0] + nxt[0]) // 2, (cur[1] + nxt[1]) // 2)
        rows[wall[0]][wall[1]] = EMPTY
        rows[nxt[0]][nxt[1]] = EMPTY
        stack.append(nxt)

    spec0 = MazeSpec.from_rows(["".join(row) for row in rows], name=f"{scenario}-{seed}", scenario=scenario)
    exit_pos = farthest_cell(spec0, start)
    path_actions = bfs_path(spec0, start, exit_pos) or []
    path = replay_positions(start, path_actions)
    boss_idx = max(1, int(len(path) * 0.78))
    boss = path[min(boss_idx, len(path) - 2)]

    grid = [list(row) for row in rows]
    grid[start[0]][start[1]] = START
    grid[exit_pos[0]][exit_pos[1]] = EXIT
    grid[boss[0]][boss[1]] = BOSS

    open_cells = [(r, c) for r in range(size) for c in range(size) if grid[r][c] == EMPTY]
    path_set = set(path)
    path_cells = [p for p in path[2:-2] if grid[p[0]][p[1]] == EMPTY]
    off_path = [p for p in open_cells if p not in path_set]
    rng.shuffle(path_cells)
    rng.shuffle(off_path)

    boss_cost = 100
    if scenario == "shortest":
        add_many(grid, path_cells, COIN, 2)
        add_many(grid, off_path, TRAP, 2)
        boss_cost = 50
    elif scenario == "on_path_gold":
        add_many(grid, path_cells, COIN, 4)
        add_many(grid, off_path, TRAP, 3)
        boss_cost = 100
    elif scenario == "detour_gold":
        add_many(grid, off_path, COIN, 5)
        add_many(grid, path_cells, TRAP, 1)
        boss_cost = 150
    elif scenario == "avoid_trap":
        add_many(grid, path_cells, COIN, 3)
        add_many(grid, open_cells, TRAP, 6)
        boss_cost = 100
    elif scenario == "trap_required":
        # Put the richest coins behind a likely detour and raise the BOSS gate so
        # an expert sometimes accepts trap damage to satisfy the resource gate.
        add_many(grid, path_cells, COIN, 2)
        add_many(grid, off_path, COIN, 5)
        add_many(grid, off_path, TRAP, 4)
        boss_cost = 200
    elif scenario == "boss_gate":
        add_many(grid, path_cells, COIN, 2)
        add_many(grid, off_path, COIN, 4)
        add_many(grid, open_cells, TRAP, 3)
        boss_cost = 150

    return MazeSpec.from_rows(
        ["".join(row) for row in grid],
        boss_cost=boss_cost,
        name=f"{scenario}-{seed}",
        scenario=scenario,
    )


def farthest_cell(spec: MazeSpec, start: Coord) -> Coord:
    q: deque[Coord] = deque([start])
    dist = {start: 0}
    far = start
    while q:
        cur = q.popleft()
        if dist[cur] > dist[far]:
            far = cur
        for _, nxt in neighbors(spec, cur, boss_defeated=True):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return far


def replay_positions(start: Coord, actions: list[str]) -> list[Coord]:
    from .constants import MOVE_DELTAS

    pos = start
    out = [pos]
    for action in actions:
        dr, dc = MOVE_DELTAS[action]
        pos = (pos[0] + dr, pos[1] + dc)
        out.append(pos)
    return out


def add_many(grid: list[list[str]], cells: list[Coord], tile: str, count: int) -> None:
    added = 0
    for r, c in cells:
        if added >= count:
            return
        if grid[r][c] == EMPTY:
            grid[r][c] = tile
            added += 1
