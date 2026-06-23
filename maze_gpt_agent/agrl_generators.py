from __future__ import annotations

import random

from .agrl_core import (
    BOSS,
    COIN,
    EMPTY,
    EXIT,
    START,
    TRAP,
    WALL,
    BossConfig,
    Coord,
    MazeSample,
    boss_for_difficulty,
    default_skills,
    farthest_cell,
    find_tiles,
    grid_neighbors_grid,
    neighbors_of_tiles,
    place_many,
    positions_from_actions,
    shortest_path,
    validate_sample,
)


def generate_agrl_maze_v2(
    size: int = 11,
    seed: int = 42,
    difficulty: str = "Medium",
    split: str = "train",
    algorithm: str = "dfs",
) -> MazeSample:
    if size % 2 == 0:
        size += 1
    size = max(5, size)
    algorithm = algorithm.lower()
    if algorithm == "mixed":
        algorithm = ["dfs", "prim", "kruskal", "division"][seed % 4]
    rng = random.Random(seed)
    if algorithm == "dfs":
        rows = dfs_backtracker(size, rng)
    elif algorithm == "prim":
        rows = randomized_prim(size, rng)
    elif algorithm == "kruskal":
        rows = randomized_kruskal(size, rng)
    elif algorithm in {"division", "recursive_division"}:
        rows = recursive_division(size, rng)
    else:
        raise ValueError(f"unknown maze algorithm: {algorithm}")
    sample = decorate_course_rules(rows, seed, difficulty, split, algorithm, rng)
    if not validate_sample(sample)["ok"]:
        return generate_agrl_maze_v2(size, seed + 99991, difficulty, split, algorithm)
    return sample


def dfs_backtracker(size: int, rng: random.Random) -> list[str]:
    grid = [[WALL for _ in range(size)] for _ in range(size)]
    start = (1, 1)
    grid[start[0]][start[1]] = EMPTY
    stack = [start]
    dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
    while stack:
        cur = stack[-1]
        candidates: list[Coord] = []
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nxt = (cur[0] + dr, cur[1] + dc)
            if 1 <= nxt[0] < size - 1 and 1 <= nxt[1] < size - 1 and grid[nxt[0]][nxt[1]] == WALL:
                candidates.append(nxt)
        if not candidates:
            stack.pop()
            continue
        nxt = candidates[0]
        wall = ((cur[0] + nxt[0]) // 2, (cur[1] + nxt[1]) // 2)
        grid[wall[0]][wall[1]] = EMPTY
        grid[nxt[0]][nxt[1]] = EMPTY
        stack.append(nxt)
    return ["".join(row) for row in grid]


def randomized_prim(size: int, rng: random.Random) -> list[str]:
    grid = [[WALL for _ in range(size)] for _ in range(size)]
    start = (1, 1)
    grid[start[0]][start[1]] = EMPTY
    frontier: list[tuple[Coord, Coord]] = []

    def add_frontier(cell: Coord) -> None:
        for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nxt = (cell[0] + dr, cell[1] + dc)
            if 1 <= nxt[0] < size - 1 and 1 <= nxt[1] < size - 1 and grid[nxt[0]][nxt[1]] == WALL:
                frontier.append((cell, nxt))

    add_frontier(start)
    while frontier:
        idx = rng.randrange(len(frontier))
        src, dst = frontier.pop(idx)
        if grid[dst[0]][dst[1]] != WALL:
            continue
        wall = ((src[0] + dst[0]) // 2, (src[1] + dst[1]) // 2)
        grid[wall[0]][wall[1]] = EMPTY
        grid[dst[0]][dst[1]] = EMPTY
        add_frontier(dst)
    return ["".join(row) for row in grid]


class DSU:
    def __init__(self, items: list[Coord]):
        self.parent = {item: item for item in items}

    def find(self, item: Coord) -> Coord:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: Coord, b: Coord) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def randomized_kruskal(size: int, rng: random.Random) -> list[str]:
    grid = [[WALL for _ in range(size)] for _ in range(size)]
    cells = [(r, c) for r in range(1, size - 1, 2) for c in range(1, size - 1, 2)]
    for r, c in cells:
        grid[r][c] = EMPTY
    edges: list[tuple[Coord, Coord, Coord]] = []
    for r, c in cells:
        for dr, dc in ((2, 0), (0, 2)):
            nxt = (r + dr, c + dc)
            if nxt in cells:
                wall = (r + dr // 2, c + dc // 2)
                edges.append(((r, c), nxt, wall))
    rng.shuffle(edges)
    dsu = DSU(cells)
    for a, b, wall in edges:
        if dsu.union(a, b):
            grid[wall[0]][wall[1]] = EMPTY
    return ["".join(row) for row in grid]


def recursive_division(size: int, rng: random.Random) -> list[str]:
    grid = [[EMPTY for _ in range(size)] for _ in range(size)]
    for i in range(size):
        grid[0][i] = grid[size - 1][i] = WALL
        grid[i][0] = grid[i][size - 1] = WALL

    def divide(r1: int, c1: int, r2: int, c2: int) -> None:
        height = r2 - r1
        width = c2 - c1
        if height < 4 or width < 4:
            return
        horizontal = height >= width
        if horizontal:
            wall_r = rng.randrange(r1 + 2, r2 - 1, 2)
            passage_c = rng.randrange(c1 + 1, c2, 2)
            for c in range(c1 + 1, c2):
                if c != passage_c:
                    grid[wall_r][c] = WALL
            divide(r1, c1, wall_r, c2)
            divide(wall_r, c1, r2, c2)
        else:
            wall_c = rng.randrange(c1 + 2, c2 - 1, 2)
            passage_r = rng.randrange(r1 + 1, r2, 2)
            for r in range(r1 + 1, r2):
                if r != passage_r:
                    grid[r][wall_c] = WALL
            divide(r1, c1, r2, wall_c)
            divide(r1, wall_c, r2, c2)

    divide(0, 0, size - 1, size - 1)
    grid[1][1] = EMPTY
    return ["".join(row) for row in grid]


def decorate_course_rules(
    rows: list[str],
    seed: int,
    difficulty: str,
    split: str,
    algorithm: str,
    rng: random.Random,
) -> MazeSample:
    size = len(rows)
    grid = [list(row) for row in rows]
    start = nearest_open(rows, (1, 1))
    end = farthest_cell(rows, start)
    path_actions = shortest_path(rows, start, end) or []
    path = positions_from_actions(start, path_actions)
    if len(path) < 5:
        rows = dfs_backtracker(size, rng)
        grid = [list(row) for row in rows]
        start = (1, 1)
        end = farthest_cell(rows, start)
        path = positions_from_actions(start, shortest_path(rows, start, end) or [])
    boss_idx = min(max(1, len(path) - rng.randint(2, 4)), len(path) - 2)
    boss = path[boss_idx]
    grid[start[0]][start[1]] = START
    grid[end[0]][end[1]] = EXIT
    grid[boss[0]][boss[1]] = BOSS

    path_set = set(path)
    empty_cells = [(r, c) for r in range(size) for c in range(size) if grid[r][c] == EMPTY]
    main_cells = [p for p in path[1:-1] if grid[p[0]][p[1]] == EMPTY]
    branch_cells = [p for p in empty_cells if p not in path_set]
    rng.shuffle(main_cells)
    rng.shuffle(branch_cells)
    cfg = {
        "Easy": (4, 1, 0, 1, 0),
        "Medium": (3, 3, 1, 2, 1),
        "Hard": (2, 4, 2, 4, 1),
        "Extreme": (1, 5, 3, 5, 2),
    }.get(difficulty, (3, 3, 1, 2, 1))
    main_gold, branch_gold, bait_gold, light_traps, required_traps = cfg
    place_many(grid, main_cells, COIN, main_gold)
    place_many(grid, branch_cells, COIN, branch_gold)
    bait_neighbors = neighbors_of_tiles(grid, COIN)
    rng.shuffle(bait_neighbors)
    place_many(grid, bait_neighbors, TRAP, bait_gold)
    place_many(grid, branch_cells, TRAP, light_traps)
    required_pool = [p for p in main_cells if p not in (start, boss, end)]
    place_many(grid, required_pool, TRAP, required_traps)

    final_rows = ["".join(row) for row in grid]
    return MazeSample(
        sample_id=f"{split}-{algorithm}-{difficulty}-{seed}",
        seed=seed,
        rows=size,
        cols=size,
        grid=final_rows,
        start=start,
        end=end,
        boss=boss,
        coins=find_tiles(final_rows, COIN),
        traps=find_tiles(final_rows, TRAP),
        difficulty=difficulty,
        boss_config=boss_for_difficulty(difficulty),
        skill_config=default_skills(),
        train_split=split,
    )


def nearest_open(rows: list[str], origin: Coord) -> Coord:
    q = [origin]
    seen = {origin}
    while q:
        cur = q.pop(0)
        r, c = cur
        if rows[r][c] != WALL:
            return cur
        for _, nxt in grid_neighbors_grid(rows, cur):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return (1, 1)
