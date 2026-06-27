"""Perfect-maze test-fixture generator (for evaluating the AI player).

This is a helper for building cross-test corpora, not the team's deliverable
(the group's task is the AI player). It carves a perfect maze with a randomized
DFS backtracker (guarantees a tree: no isolated regions, unique path between any
two cells), then places start/exit/boss and scatters coins and traps.
"""

from __future__ import annotations

import random

from .spec import MazeSpec, Skill


def generate_maze(n: int = 15, seed: int = 0, coin_ratio: float = 0.14, trap_ratio: float = 0.12) -> MazeSpec:
    if n % 2 == 0:
        n += 1
    rng = random.Random(seed)
    grid = [["#"] * n for _ in range(n)]

    def carve(r: int, c: int) -> None:
        grid[r][c] = " "
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < n - 1 and 1 <= nc < n - 1 and grid[nr][nc] == "#":
                grid[r + dr // 2][c + dc // 2] = " "
                carve(nr, nc)

    carve(1, 1)
    floors = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == " "]
    floor_set = set(floors)

    def degree(p):
        return sum(1 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                   if (p[0] + dr, p[1] + dc) in floor_set)

    start = (1, 1)
    # Make the exit a real dead-end guarded by the boss: pick a degree-1 leaf
    # (farthest from start) as the exit, and put the boss on its unique neighbour
    # so every route to the exit must pass through the boss.
    from collections import deque
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cur[0] + dr, cur[1] + dc)
            if nb in floor_set and nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    leaves = [p for p in floors if p != start and degree(p) == 1]
    leaves.sort(key=lambda p: dist.get(p, -1), reverse=True)
    exit_ = leaves[0] if leaves else (n - 2, n - 2)
    boss = next(((exit_[0] + dr, exit_[1] + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                 if (exit_[0] + dr, exit_[1] + dc) in floor_set), None)
    reserved = {start, exit_, boss}
    free = [p for p in floors if p not in reserved]
    rng.shuffle(free)

    n_coins = max(3, int(len(free) * coin_ratio))
    n_traps = max(2, int(len(free) * trap_ratio))
    coins = free[:n_coins]
    traps = free[n_coins:n_coins + n_traps]

    grid[start[0]][start[1]] = "S"
    grid[exit_[0]][exit_[1]] = "E"
    if boss:
        grid[boss[0]][boss[1]] = "B"
    for (r, c) in coins:
        grid[r][c] = "G"
    for (r, c) in traps:
        grid[r][c] = "T"

    rows = ["".join(row) for row in grid]
    # A small, beatable boss group with a standard skill kit.
    boss_hps = [rng.choice([9, 11, 13, 15]) for _ in range(rng.randint(2, 4))]
    skills = [Skill(8, 4), Skill(2, 0), Skill(4, 2), Skill(6, 3)]
    return MazeSpec(grid=rows, boss_hps=boss_hps, skills=skills,
                    min_rounds=20, coin_consumption=5, name=f"gen_{n}_{seed}")
