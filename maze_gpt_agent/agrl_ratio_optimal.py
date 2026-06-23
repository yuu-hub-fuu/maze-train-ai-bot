from __future__ import annotations

from collections import deque
from dataclasses import asdict
import random
from typing import Any

from .agrl_core import (
    BOSS,
    COIN,
    COIN_VALUE,
    EMPTY,
    EXIT,
    MOVES,
    START,
    TRAP,
    TRAP_DAMAGE,
    WALL,
    BossConfig,
    Coord,
    ExpertSolution,
    MazeSample,
    PathResult,
    boss_for_difficulty,
    default_skills,
    find_tiles,
    grid_neighbors_grid,
    positions_from_actions,
    shortest_path,
    solve_boss_battle,
)
from .agrl_generators import dfs_backtracker, randomized_kruskal, randomized_prim, recursive_division


def generate_unbiased_maze(
    size: int = 11,
    seed: int = 42,
    difficulty: str = "Medium",
    split: str = "train",
    algorithm: str = "mixed",
) -> MazeSample:
    if size % 2 == 0:
        size += 1
    size = max(5, size)
    rng = random.Random(seed)
    chosen = algorithm.lower()
    if chosen == "mixed":
        chosen = rng.choice(["dfs", "prim", "kruskal", "division"])
    if chosen == "dfs":
        rows = dfs_backtracker(size, rng)
    elif chosen == "prim":
        rows = randomized_prim(size, rng)
    elif chosen == "kruskal":
        rows = randomized_kruskal(size, rng)
    elif chosen in {"division", "recursive_division"}:
        rows = recursive_division(size, rng)
    else:
        raise ValueError(f"unknown maze algorithm: {algorithm}")
    rows = braid_maze(rows, rng, rate=0.10 if size >= 11 else 0.06)
    grid = [list(row) for row in rows]
    open_cells = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1) if grid[r][c] == EMPTY]
    start, end, boss = choose_random_specials(rows, open_cells, rng, size)
    for pos, token in ((start, START), (end, EXIT), (boss, BOSS)):
        grid[pos[0]][pos[1]] = token
    free = [p for p in open_cells if p not in {start, end, boss}]
    rng.shuffle(free)
    coin_count, trap_count = counts_for(size, difficulty)
    coins = free[:coin_count]
    traps = free[coin_count : coin_count + trap_count]
    for r, c in coins:
        grid[r][c] = COIN
    for r, c in traps:
        grid[r][c] = TRAP
    final_rows = ["".join(row) for row in grid]
    return MazeSample(
        sample_id=f"{split}-ratio-{chosen}-{difficulty}-{seed}",
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


def braid_maze(rows: list[str], rng: random.Random, rate: float) -> list[str]:
    grid = [list(row) for row in rows]
    candidates: list[Coord] = []
    for r in range(1, len(grid) - 1):
        for c in range(1, len(grid[0]) - 1):
            if grid[r][c] != WALL:
                continue
            vertical = grid[r - 1][c] != WALL and grid[r + 1][c] != WALL
            horizontal = grid[r][c - 1] != WALL and grid[r][c + 1] != WALL
            if vertical or horizontal:
                candidates.append((r, c))
    rng.shuffle(candidates)
    for r, c in candidates[: int(len(candidates) * rate)]:
        grid[r][c] = EMPTY
    return ["".join(row) for row in grid]


def choose_random_specials(rows: list[str], open_cells: list[Coord], rng: random.Random, size: int) -> tuple[Coord, Coord, Coord]:
    min_dist = max(4, size // 2)
    for _ in range(500):
        start, end, boss = rng.sample(open_cells, 3)
        sb = shortest_path(rows, start, boss)
        be = shortest_path(rows, boss, end)
        se = shortest_path(rows, start, end)
        if not sb or not be or not se:
            continue
        if len(sb) >= min_dist and len(be) >= max(3, min_dist // 2) and len(se) >= min_dist:
            return start, end, boss
    start, end = rng.sample(open_cells, 2)
    boss = rng.choice([p for p in open_cells if p not in {start, end}])
    return start, end, boss


def counts_for(size: int, difficulty: str) -> tuple[int, int]:
    scale = 1 if size <= 11 else 2
    table = {
        "Easy": (4 + scale, 1 + scale),
        "Medium": (5 + scale, 3 + scale),
        "Hard": (6 + scale, 5 + scale),
        "Extreme": (7 + scale, 7 + scale),
    }
    return table.get(difficulty, table["Medium"])


def ratio_optimal_solution(sample: MazeSample, max_steps: int | None = None, max_expansions: int = 10000) -> ExpertSolution | None:
    coin_index = {pos: idx for idx, pos in enumerate(sample.coins)}
    trap_index = {pos: idx for idx, pos in enumerate(sample.traps)}
    max_steps = max_steps or sample.rows * sample.cols * 4
    expansions = 0
    start_state = (sample.start, 0, 0, False)
    q = deque([(sample.start, 0, 0, False, 0, 0)])
    parent: dict[tuple[Coord, int, int, bool, int], tuple[tuple[Coord, int, int, bool, int], str]] = {}
    best_at: dict[tuple[Coord, int, int, bool], list[tuple[int, int]]] = {start_state: [(0, 0)]}
    best_terminal: tuple[float, int, int, Coord, int, int, bool] | None = None

    while q:
        expansions += 1
        if expansions > max_expansions:
            return None
        pos, coin_mask, trap_mask, boss_done, steps, resource = q.popleft()
        if steps >= max_steps:
            continue
        for action, nxt in grid_neighbors_grid(sample.grid, pos):
            tile = sample.char_at(nxt)
            if tile == EXIT and not boss_done:
                continue
            new_coin_mask = coin_mask
            new_trap_mask = trap_mask
            new_boss_done = boss_done
            new_resource = resource
            if nxt in coin_index and not (coin_mask & (1 << coin_index[nxt])):
                new_coin_mask |= 1 << coin_index[nxt]
                new_resource += COIN_VALUE
            if nxt in trap_index and not (trap_mask & (1 << trap_index[nxt])):
                new_trap_mask |= 1 << trap_index[nxt]
                new_resource -= TRAP_DAMAGE
            if tile == BOSS and not boss_done:
                boss = solve_boss_battle(sample.boss_config, sample.skill_config, new_resource)
                if boss.success and new_resource >= sample.boss_config.revive_cost:
                    new_boss_done = True
                else:
                    continue
            new_steps = steps + 1
            key = (nxt, new_coin_mask, new_trap_mask, new_boss_done)
            if dominated(best_at.get(key, []), new_steps, new_resource):
                continue
            best_at.setdefault(key, []).append((new_steps, new_resource))
            best_at[key] = prune_labels(best_at[key])
            parent[(nxt, new_coin_mask, new_trap_mask, new_boss_done, new_steps)] = (
                (pos, coin_mask, trap_mask, boss_done, steps),
                action,
            )
            if nxt == sample.end and new_boss_done:
                ratio = new_resource / max(1, new_steps)
                cand = (ratio, new_resource, -new_steps, nxt, new_coin_mask, new_trap_mask, new_boss_done)
                if best_terminal is None or cand > best_terminal:
                    best_terminal = cand
            q.append((nxt, new_coin_mask, new_trap_mask, new_boss_done, new_steps, new_resource))

    if best_terminal is None:
        return None
    ratio, final_resource, neg_steps, pos, coin_mask, trap_mask, boss_done = best_terminal
    total_steps = -neg_steps
    state_key = (pos, coin_mask, trap_mask, boss_done, total_steps)
    actions: list[str] = []
    while state_key[0] != sample.start or state_key[4] != 0:
        prev, action = parent[state_key]
        actions.append(action)
        state_key = prev
    actions.reverse()
    path = positions_from_actions(sample.start, actions)
    collected = [coin for coin, idx in coin_index.items() if coin_mask & (1 << idx)]
    boss = solve_boss_battle(sample.boss_config, sample.skill_config, final_resource)
    return ExpertSolution(
        recommended_targets=[],
        recommended_path=path,
        collected_coins=collected,
        boss_skill_sequence=boss.skill_sequence,
        final_resource=final_resource,
        total_steps=total_steps,
        final_score=ratio,
        success=True,
    )


def dominated(labels: list[tuple[int, int]], steps: int, resource: int) -> bool:
    return any(old_steps <= steps and old_resource >= resource for old_steps, old_resource in labels)


def prune_labels(labels: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for steps, resource in sorted(labels):
        if not any(s <= steps and r >= resource for s, r in out):
            out.append((steps, resource))
    return out


def solution_to_dict(solution: ExpertSolution) -> dict[str, Any]:
    data = asdict(solution)
    data["recommended_path"] = [list(p) for p in solution.recommended_path]
    data["collected_coins"] = [list(p) for p in solution.collected_coins]
    return data



def ratio_optimal_solution_from_state(
    sample: MazeSample,
    state: PlayerState,
    max_extra_steps: int | None = None,
    allowed_known_only: bool = False,
    max_expansions: int = 10000,
) -> ExpertSolution | None:
    """Continue planning from the current executed state and optimize final resource/total steps."""
    coin_index = {pos: idx for idx, pos in enumerate(sample.coins)}
    trap_index = {pos: idx for idx, pos in enumerate(sample.traps)}
    coin_mask0 = 0
    trap_mask0 = 0
    for pos in state.collected_coins:
        if pos in coin_index:
            coin_mask0 |= 1 << coin_index[pos]
    for pos in state.triggered_traps:
        if pos in trap_index:
            trap_mask0 |= 1 << trap_index[pos]
    max_extra_steps = max_extra_steps or sample.rows * sample.cols * 4
    expansions = 0
    start_state = (state.position, coin_mask0, trap_mask0, state.boss_defeated)
    q = deque([(state.position, coin_mask0, trap_mask0, state.boss_defeated, 0, state.resource)])
    parent: dict[tuple[Coord, int, int, bool, int], tuple[tuple[Coord, int, int, bool, int], str]] = {}
    best_at: dict[tuple[Coord, int, int, bool], list[tuple[int, int]]] = {start_state: [(0, state.resource)]}
    best_terminal: tuple[float, int, int, Coord, int, int, bool] | None = None

    def usable(pos: Coord) -> bool:
        if not allowed_known_only:
            return True
        return pos in state.known and state.known.get(pos) != WALL

    while q:
        expansions += 1
        if expansions > max_expansions:
            return None
        pos, coin_mask, trap_mask, boss_done, extra_steps, resource = q.popleft()
        if extra_steps >= max_extra_steps:
            continue
        for action, nxt in grid_neighbors_grid(sample.grid, pos):
            if not usable(nxt):
                continue
            tile = sample.char_at(nxt)
            if tile == EXIT and not boss_done:
                continue
            new_coin_mask = coin_mask
            new_trap_mask = trap_mask
            new_boss_done = boss_done
            new_resource = resource
            if nxt in coin_index and not (coin_mask & (1 << coin_index[nxt])):
                new_coin_mask |= 1 << coin_index[nxt]
                new_resource += COIN_VALUE
            if nxt in trap_index and not (trap_mask & (1 << trap_index[nxt])):
                new_trap_mask |= 1 << trap_index[nxt]
                new_resource -= TRAP_DAMAGE
            if tile == BOSS and not boss_done:
                boss = solve_boss_battle(sample.boss_config, sample.skill_config, new_resource)
                if boss.success and new_resource >= sample.boss_config.revive_cost:
                    new_boss_done = True
                else:
                    continue
            new_extra_steps = extra_steps + 1
            key = (nxt, new_coin_mask, new_trap_mask, new_boss_done)
            if dominated(best_at.get(key, []), new_extra_steps, new_resource):
                continue
            best_at.setdefault(key, []).append((new_extra_steps, new_resource))
            best_at[key] = prune_labels(best_at[key])
            parent[(nxt, new_coin_mask, new_trap_mask, new_boss_done, new_extra_steps)] = (
                (pos, coin_mask, trap_mask, boss_done, extra_steps),
                action,
            )
            if nxt == sample.end and new_boss_done:
                total_steps = state.steps + new_extra_steps
                ratio = new_resource / max(1, total_steps)
                cand = (ratio, new_resource, -total_steps, nxt, new_coin_mask, new_trap_mask, new_boss_done)
                if best_terminal is None or cand > best_terminal:
                    best_terminal = cand
            q.append((nxt, new_coin_mask, new_trap_mask, new_boss_done, new_extra_steps, new_resource))

    if best_terminal is None:
        return None
    ratio, final_resource, neg_total_steps, pos, coin_mask, trap_mask, boss_done = best_terminal
    total_steps = -neg_total_steps
    extra_steps = total_steps - state.steps
    state_key = (pos, coin_mask, trap_mask, boss_done, extra_steps)
    actions: list[str] = []
    while state_key[0] != state.position or state_key[4] != 0:
        prev, action = parent[state_key]
        actions.append(action)
        state_key = prev
    actions.reverse()
    path = positions_from_actions(state.position, actions)
    collected = [coin for coin, idx in coin_index.items() if coin_mask & (1 << idx)]
    boss = solve_boss_battle(sample.boss_config, sample.skill_config, final_resource)
    return ExpertSolution(
        recommended_targets=[],
        recommended_path=path,
        collected_coins=collected,
        boss_skill_sequence=boss.skill_sequence,
        final_resource=final_resource,
        total_steps=total_steps,
        final_score=ratio,
        success=True,
    )



