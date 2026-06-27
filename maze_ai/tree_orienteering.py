"""Polynomial-time optimal resource route on a perfect maze (tree).

The bitmask DP in ``planner.py`` is exact but costs ``O(2^coins)``. On a perfect
maze the walkable cells form a tree, and that structure lets us solve the same
"max resource / step" route in time polynomial in the map size, for *any* number
of coins.

Key fact: any walk from S to E on a tree traverses a connected subtree
``T ⊇ {S, E, boss}``; every edge of T is walked twice except those on the unique
S–E path, so

    steps(T)  = 2·|edges(T)| − dist(S, E)
    resource(T) = Σ node prizes in T        (coin +50, trap −30, once each)

We want to maximise ``resource(T) / steps(T)`` over connected subtrees containing
the mandatory terminals {S, E, boss}. That ratio is handled by Dinkelbach's
parametric method: for a fixed λ,

    maximise  resource(T) − λ·steps(T)
            = [ Σ prizes − 2λ·|edges(T)| ] + λ·dist(S,E)

The bracket is a max-profit connected subtree (node prizes, uniform edge cost
2λ, mandatory nodes forced) — a classic O(V) tree DP. Iterating λ ← ratio(T)
converges to the optimal ratio in a handful of rounds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .grid import bfs_from, neighbors, path_to_moves
from .spec import COIN_VALUE, TRAP_DAMAGE, COIN, TRAP, Coord, MazeSpec


@dataclass
class TreeRoute:
    coord_path: list[Coord]
    moves: list[str]
    resource: int
    steps: int
    score: float


def is_perfect_maze(spec: MazeSpec) -> bool:
    """True iff the walkable cells form a tree (connected, |E| == |V| − 1)."""
    dist, _ = bfs_from(spec, spec.start)
    nodes = sum(1 for r in range(spec.height) for c in range(spec.width) if spec.is_walkable((r, c)))
    if len(dist) != nodes:
        return False  # not connected
    edges = 0
    for (r, c) in dist:
        for nb in neighbors(spec, (r, c)):
            edges += 1
    return edges // 2 == nodes - 1


def plan_tree_orienteering(spec: MazeSpec) -> TreeRoute:
    S, E, B = spec.start, spec.exit, spec.boss
    distSE = bfs_from(spec, S)[0][E]

    # Root the tree at S; build parent/children over walkable cells.
    dist, parent = bfs_from(spec, S)
    nodes = list(dist.keys())
    children: dict[Coord, list[Coord]] = {u: [] for u in nodes}
    for u in nodes:
        if u in parent:
            children[parent[u]].append(u)
    post_order = sorted(nodes, key=lambda u: dist[u], reverse=True)  # children before parents

    mandatory = {S, E} | ({B} if B is not None else set())
    # subtree_has_mandatory[u]: does u's subtree contain a mandatory node?
    has_mand: dict[Coord, bool] = {}
    for u in post_order:
        flag = u in mandatory
        for c in children[u]:
            flag = flag or has_mand[c]
        has_mand[u] = flag

    coin_set, trap_set = set(spec.coins), set(spec.traps)

    def prize(u: Coord) -> int:
        if u in coin_set:
            return COIN_VALUE
        if u in trap_set:
            return -TRAP_DAMAGE
        return 0

    def solve_fixed_lambda(lam: float):
        """Max-profit connected subtree containing the terminals; edge cost 2λ."""
        edge_cost = 2.0 * lam
        dp: dict[Coord, float] = {}
        include: dict[Coord, set[Coord]] = {u: set() for u in nodes}
        for u in post_order:
            total = float(prize(u))
            for c in children[u]:
                val = dp[c] - edge_cost
                if has_mand[c]:
                    total += val            # forced: terminal lives below c
                    include[u].add(c)
                elif val > 0:
                    total += val            # optional: only if profitable
                    include[u].add(c)
            dp[u] = total

        # reconstruct the chosen connected subtree from S
        chosen: set[Coord] = set()
        stack = [S]
        while stack:
            u = stack.pop()
            chosen.add(u)
            for c in include[u]:
                stack.append(c)
        edges = len(chosen) - 1
        res = sum(prize(u) for u in chosen)
        return chosen, edges, res

    # Dinkelbach iteration on the ratio resource/steps.
    lam = 0.0
    chosen, edges, res = solve_fixed_lambda(lam)
    for _ in range(64):
        steps = 2 * edges - distSE
        if steps <= 0:
            break
        new_lam = res / steps
        if abs(new_lam - lam) < 1e-12:
            lam = new_lam
            break
        lam = new_lam
        chosen, edges, res = solve_fixed_lambda(lam)

    coord_path = _build_walk(spec, chosen, children, parent, S, E)
    moves = path_to_moves(coord_path)
    steps = len(coord_path) - 1
    return TreeRoute(coord_path=coord_path, moves=moves, resource=res, steps=steps,
                     score=(res / steps if steps else 0.0))


def _build_walk(spec, chosen: set, children, parent, S: Coord, E: Coord) -> list[Coord]:
    """DFS walk over the chosen subtree from S, ending at E (E-branch deferred)."""
    # nodes on the S->E path (within the tree) get their E-ward child visited last.
    on_path = set()
    cur = E
    on_path.add(E)
    while cur != S:
        cur = parent[cur]
        on_path.add(cur)
    # next node toward E from each on-path node
    e_child: dict[Coord, Coord] = {}
    cur = E
    while cur != S:
        e_child[parent[cur]] = cur
        cur = parent[cur]

    walk: list[Coord] = []

    def dfs(u: Coord) -> None:
        walk.append(u)
        kids = [c for c in children[u] if c in chosen]
        last = e_child.get(u)
        ordered = [c for c in kids if c != last] + ([last] if last in kids else [])
        for c in ordered:
            dfs(c)
            if c is not last:
                walk.append(u)  # backtrack to u after a side excursion
    dfs(S)
    return walk
