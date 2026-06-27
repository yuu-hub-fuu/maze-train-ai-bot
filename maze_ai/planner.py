"""Optimal full-maze exploration planner (AI-player task 2).

The maze is fully observable, so we compute a strong global route instead of
reacting step by step. Method:

1. Build a waypoint graph over {start, coins, boss, exit} with one BFS per
   waypoint (the maze is a tree, so each pair has a unique path).
2. Bitmask DP (Held-Karp style): ``dp[mask][i]`` = minimum steps of a route that
   starts at S, deliberately collects exactly the coins in ``mask`` and ends at
   coin ``i``. This is the classic dynamic-programming core the assignment asks
   for, here applied to resource collection.
3. Every candidate route is closed with ``... -> boss -> exit`` and scored
   *exactly* (incidental coins/traps along corridors, traps counted once, boss
   cleared for free under optimal skill play). We keep the route with the best
   ``resource / steps`` ratio.

For mazes with more coins than ``MAX_DP_COINS`` we fall back to a greedy
nearest-positive-ratio insertion heuristic so the planner never blows up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .boss import BossPlan, plan_boss_fight
from .grid import WaypointGraph, path_to_moves
from .spec import (
    BOSS,
    COIN,
    COIN_VALUE,
    EXIT,
    TRAP,
    TRAP_DAMAGE,
    Coord,
    MazeSpec,
)

MAX_DP_COINS = 15
INF = 10**9


@dataclass
class OptimalPlan:
    moves: list[str]
    coord_path: list[Coord]
    resource: int
    steps: int
    score: float
    coins_collected: int
    traps_triggered: int
    chosen_coins: list[Coord]
    boss_plan: BossPlan
    method: str = "bitmask_dp"
    notes: dict[str, Any] = field(default_factory=dict)


def plan_full_exploration(spec: MazeSpec) -> OptimalPlan:
    coins = list(spec.coins)
    boss_plan = plan_boss_fight(spec, available_coins=len(coins) * COIN_VALUE)
    boss_coins_spent = boss_plan.coins_spent

    # Boundary: too many coins for the exponential bitmask DP. If the maze is a
    # perfect maze (tree), switch to the polynomial-time tree-orienteering DP,
    # which stays exact; otherwise fall back to the greedy heuristic below.
    if len(coins) > MAX_DP_COINS:
        from .tree_orienteering import is_perfect_maze, plan_tree_orienteering
        if is_perfect_maze(spec):
            tr = plan_tree_orienteering(spec)
            res, steps, ncoins, ntraps, _ = _score_path(spec, tr.coord_path, boss_coins_spent)
            chosen = set(tr.coord_path)
            return OptimalPlan(
                moves=path_to_moves(tr.coord_path),
                coord_path=tr.coord_path,
                resource=res,
                steps=steps,
                score=(res / steps if steps else 0.0),
                coins_collected=ncoins,
                traps_triggered=ntraps,
                chosen_coins=[c for c in coins if c in chosen],
                boss_plan=boss_plan,
                method="tree_orienteering",
                notes={"num_coins": len(coins), "candidates_evaluated": 0,
                       "boss_within_limit": boss_plan.feasible_within_limit},
            )

    # Waypoint graph over start, coins, boss, exit.
    terminals = [spec.boss, spec.exit] if spec.boss is not None else [spec.exit]
    waypoints = [spec.start] + coins + [t for t in terminals]
    graph = WaypointGraph(spec, waypoints)

    close_to = terminals[0]  # boss if present, else exit: route closes toward it
    if len(coins) <= MAX_DP_COINS:
        order_candidates = _dp_orders(spec, coins, graph, close_to)
        method = "bitmask_dp"
    else:
        order_candidates = _greedy_orders(spec, coins, graph)
        method = "greedy_fallback"

    # Score every candidate cheaply; only materialise the winning route once.
    best_key: tuple | None = None
    best_order: list[int] = []
    best_coords: list[Coord] = []
    best_stats = (0, 1, 0, 0)
    for order in order_candidates:
        coord_path = _build_coord_path(graph, spec, order, terminals)
        res, steps, ncoins, ntraps, reaches_exit = _score_path(spec, coord_path, boss_coins_spent)
        if steps <= 0:
            continue
        key = (reaches_exit, res / steps, res, -steps)
        if best_key is None or key > best_key:
            best_key = key
            best_order = order
            best_coords = coord_path
            best_stats = (res, steps, ncoins, ntraps)

    if best_key is None:
        best_coords = _build_coord_path(graph, spec, [], terminals)
        best_stats = _score_path(spec, best_coords, boss_coins_spent)[:4]
        method += "_direct"

    res, steps, ncoins, ntraps = best_stats
    return OptimalPlan(
        moves=path_to_moves(best_coords),
        coord_path=best_coords,
        resource=res,
        steps=steps,
        score=(res / steps if steps else 0.0),
        coins_collected=ncoins,
        traps_triggered=ntraps,
        chosen_coins=[coins[i] for i in best_order],
        boss_plan=boss_plan,
        method=method,
        notes={
            "num_coins": len(coins),
            "candidates_evaluated": len(order_candidates),
            "boss_within_limit": boss_plan.feasible_within_limit,
        },
    )


# ----------------------------------------------------------------------- DP
def _dp_orders(spec: MazeSpec, coins: list[Coord], graph: WaypointGraph, close_to: Coord) -> list[list[int]]:
    """Held-Karp min-steps DP, then one best-closing order per coin-subset.

    For each subset ``mask`` we keep the single visiting order whose total walk
    (collect mask, then head for the closing waypoint) is shortest. That yields
    2^n candidate routes instead of 2^n * n, each the min-step realization of its
    subset; incidental coins/traps along corridors are captured later by exact
    scoring.
    """
    n = len(coins)
    if n == 0:
        return [[]]
    start = spec.start
    dp = [[INF] * n for _ in range(1 << n)]
    par = [[-2] * n for _ in range(1 << n)]
    for i in range(n):
        d = graph.dist(start, coins[i])
        if d < INF:
            dp[1 << i][i] = d
            par[1 << i][i] = -1
    full = 1 << n
    for mask in range(full):
        row = dp[mask]
        for i in range(n):
            cur = row[i]
            if cur >= INF or not (mask >> i) & 1:
                continue
            ci = coins[i]
            for j in range(n):
                if (mask >> j) & 1:
                    continue
                nd = cur + graph.dist(ci, coins[j])
                nmask = mask | (1 << j)
                if nd < dp[nmask][j]:
                    dp[nmask][j] = nd
                    par[nmask][j] = i

    orders: list[list[int]] = [[]]  # always consider collecting nothing
    for mask in range(1, full):
        best_last, best_cost = -1, INF
        for i in range(n):
            if not (mask >> i) & 1 or dp[mask][i] >= INF:
                continue
            cost = dp[mask][i] + graph.dist(coins[i], close_to)
            if cost < best_cost:
                best_cost, best_last = cost, i
        if best_last >= 0:
            orders.append(_backtrack(par, mask, best_last))
    return orders


def _backtrack(par: list[list[int]], mask: int, last: int) -> list[int]:
    order: list[int] = []
    i = last
    while i >= 0:
        order.append(i)
        p = par[mask][i]
        mask ^= (1 << i)
        i = p
    order.reverse()
    return order


# ------------------------------------------------------------- greedy fallback
def _greedy_orders(spec: MazeSpec, coins: list[Coord], graph: WaypointGraph) -> list[list[int]]:
    """Greedy insertion: repeatedly append the coin with the best ratio gain."""
    start = spec.start
    remaining = set(range(len(coins)))
    order: list[int] = []
    cur = start
    orders: list[list[int]] = [[]]
    while remaining:
        best_j, best_gain = None, 0.0
        for j in remaining:
            d = graph.dist(cur, coins[j])
            if d <= 0 or d >= INF:
                continue
            gain = COIN_VALUE / d
            if best_j is None or gain > best_gain:
                best_j, best_gain = j, gain
        if best_j is None:
            break
        order.append(best_j)
        remaining.discard(best_j)
        cur = coins[best_j]
        orders.append(list(order))
    return orders


# -------------------------------------------------------------------- scoring
def _build_coord_path(graph: WaypointGraph, spec: MazeSpec, order: list[int], terminals: list[Coord]) -> list[Coord]:
    coins = spec.coins
    seq = [spec.start] + [coins[i] for i in order] + list(terminals)
    coords: list[Coord] = [spec.start]
    for a, b in zip(seq, seq[1:]):
        seg = graph.path(a, b)
        coords.extend(seg[1:])
    return coords


def _score_path(spec: MazeSpec, coords: list[Coord], boss_coins_spent: int) -> tuple[int, int, int, int, bool]:
    """Replicate simulate's accounting cheaply (no frames, boss cost precomputed).

    Returns (resource, steps, coins, traps, reached_exit).
    """
    resource = 0
    steps = 0
    collected: set[Coord] = set()
    triggered: set[Coord] = set()
    boss_dead = spec.boss is None
    reached_exit = False
    for pos in coords[1:]:
        ch = spec.char_at(pos)
        if ch == EXIT and not boss_dead:
            break  # exit sealed; route is invalid past here
        steps += 1
        if ch == BOSS and not boss_dead:
            boss_dead = True
            resource -= boss_coins_spent
        elif ch == COIN and pos not in collected:
            collected.add(pos)
            resource += COIN_VALUE
        elif ch == TRAP and pos not in triggered:
            triggered.add(pos)
            resource -= TRAP_DAMAGE
        elif ch == EXIT:
            reached_exit = True
            break
    return resource, steps, len(collected), len(triggered), reached_exit


def _better(a: OptimalPlan, b: OptimalPlan) -> bool:
    ka = (a.score, a.resource, -a.steps)
    kb = (b.score, b.resource, -b.steps)
    return ka > kb
