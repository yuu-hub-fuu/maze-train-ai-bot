from __future__ import annotations

from collections import deque
from dataclasses import asdict
import heapq
import time

from .agrl_core import (
    BOSS,
    COIN,
    EXIT,
    MOVES,
    TRAP,
    TRAP_DAMAGE,
    WALL,
    BossResult,
    Coord,
    MazeSample,
    PlayerState,
    RunResult,
    Target,
    apply_move,
    frame,
    known_positions,
    observe_3x3,
    solve_boss_battle,
    tile_event,
)
from .agrl_ratio_optimal import ratio_optimal_solution_from_state


def safe_memory_path(
    sample: MazeSample,
    state: PlayerState,
    goal: Coord,
    allow_boss: bool = False,
    allow_exit: bool = False,
) -> tuple[list[str], list[Coord]] | None:
    heap: list[tuple[float, int, int, Coord]] = [(0.0, 0, 0, state.position)]
    parent: dict[Coord, tuple[Coord, str]] = {}
    best_cost: dict[Coord, float] = {state.position: 0.0}
    counter = 0
    reached: Coord | None = None
    while heap:
        cost, steps, _, cur = heapq.heappop(heap)
        if cost > best_cost.get(cur, float("inf")):
            continue
        if cur == goal:
            reached = cur
            break
        for action, (dr, dc) in MOVES.items():
            nxt = (cur[0] + dr, cur[1] + dc)
            ch = state.known.get(nxt)
            if not ch or ch == WALL:
                continue
            if ch == EXIT and not state.boss_defeated and not allow_exit:
                continue
            if ch == BOSS and not state.boss_defeated and not allow_boss:
                continue
            add_cost = 1.0
            if ch == TRAP and nxt not in state.triggered_traps:
                add_cost += 75.0
            # Coin value is scored at target selection time; Dijkstra edge costs must stay non-negative.
            add_cost += 0.0
            new_cost = cost + add_cost
            if new_cost >= best_cost.get(nxt, float("inf")):
                continue
            best_cost[nxt] = new_cost
            parent[nxt] = (cur, action)
            counter += 1
            heapq.heappush(heap, (new_cost, steps + 1, counter, nxt))
    if reached is None:
        return None
    actions: list[str] = []
    path: list[Coord] = [reached]
    cur = reached
    while cur != state.position:
        prev, action = parent[cur]
        actions.append(action)
        cur = prev
        path.append(cur)
    actions.reverse()
    path.reverse()
    return actions, path


def safe_frontier_target(sample: MazeSample, state: PlayerState) -> tuple[Target, list[str], list[Coord]] | None:
    best: tuple[int, int, Coord, list[str], list[Coord]] | None = None
    for pos, ch in state.known.items():
        if ch == WALL:
            continue
        unknown_adj = 0
        for dr, dc in MOVES.values():
            nxt = (pos[0] + dr, pos[1] + dc)
            if 0 <= nxt[0] < sample.rows and 0 <= nxt[1] < sample.cols and nxt not in state.known:
                unknown_adj += 1
        if unknown_adj <= 0:
            continue
        found = safe_memory_path(sample, state, pos, allow_boss=False)
        if found is None:
            continue
        actions, path = found
        if not actions:
            continue
        cand = (len(actions), -unknown_adj, pos, actions, path)
        if best is None or cand[:3] < best[:3]:
            best = cand
    if best is None:
        return None
    dist, unknown_bonus, pos, actions, path = best
    target = Target(f"frontier-{pos[0]}-{pos[1]}", "explore", pos, 0, dist, 0, 0, unknown_bonus - dist, True)
    return target, actions, path


def best_safe_coin(sample: MazeSample, state: PlayerState) -> tuple[Target, list[str], list[Coord]] | None:
    best: tuple[float, Target, list[str], list[Coord]] | None = None
    for coin in known_positions(state, COIN):
        if coin in state.collected_coins:
            continue
        found = safe_memory_path(sample, state, coin, allow_boss=False)
        if found is None:
            continue
        actions, path = found
        if not actions:
            continue
        trap_loss = sum(TRAP_DAMAGE for p in path[1:] if sample.char_at(p) == TRAP and p not in state.triggered_traps)
        projected_resource = state.resource + 50 - trap_loss
        projected_ratio = projected_resource / max(1, state.steps + len(actions))
        current_ratio = state.resource / max(1, state.steps)
        need_bonus = 2.0 if state.resource < sample.boss_config.revive_cost else 1.0
        score = (projected_ratio - current_ratio) * need_bonus
        target = Target(f"coin-{coin[0]}-{coin[1]}", "coin", coin, 50, len(actions), trap_loss, 0, score, True)
        cand = (score, target, actions, path)
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None or (best[0] <= 0 and state.resource >= sample.boss_config.revive_cost):
        return None
    _, target, actions, path = best
    return target, actions, path


def execute_path(sample: MazeSample, state: PlayerState, frames: list[dict], actions: list[str], path: list[Coord], target: Target | None, reason: str) -> BossResult:
    boss_result = BossResult(False, 0, [], False, state.resource)
    state.decision_history.append({"action": "SAFE_RATIO_STEP", "reason": reason, "target": asdict(target) if target else None, "path_len": len(actions)})
    for action, pos in zip(actions, path[1:]):
        apply_move(sample, state, pos)
        frames.append(frame(sample, state, action, tile_event(sample, state, pos), target))
        if state.position == sample.boss and not state.boss_defeated:
            boss_result = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
            if boss_result.success and state.resource >= sample.boss_config.revive_cost:
                state.boss_defeated = True
                frames.append(frame(sample, state, "BOSS_FIGHT", "boss_clear:" + ",".join(boss_result.skill_sequence), target))
            else:
                state.alive = False
                state.done = True
                frames.append(frame(sample, state, "BOSS_FIGHT", "boss_fail", target))
                break
        if state.position == sample.end:
            state.done = True
            if not state.boss_defeated:
                state.alive = False
            break
    return boss_result


def plan_to_actions(path: list[Coord]) -> list[str]:
    actions: list[str] = []
    reverse = {delta: action for action, delta in MOVES.items()}
    for a, b in zip(path, path[1:]):
        actions.append(reverse[(b[0] - a[0], b[1] - a[1])])
    return actions


def run_safe_ratio_planner(sample: MazeSample, max_steps: int | None = None) -> RunResult:
    started = time.perf_counter()
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    frames = [frame(sample, state, "START", "start", None)]
    max_steps = max_steps or sample.rows * sample.cols * 8
    boss_result = BossResult(False, 0, [], False, state.resource)

    while state.alive and not state.done and state.steps < max_steps:
        plan = None
        boss_known_or_done = state.known.get(sample.boss) == BOSS or state.boss_defeated
        exit_known = state.known.get(sample.end) == EXIT
        if boss_known_or_done and exit_known:
            plan = ratio_optimal_solution_from_state(sample, state, allowed_known_only=True, max_expansions=2000)
        if plan is not None and len(plan.recommended_path) > 1:
            actions = plan_to_actions(plan.recommended_path)
            target = Target("known-ratio-optimal-exit", "exit", plan.recommended_path[-1], 0, len(actions), 0, 0, plan.final_score, True)
            boss_result = execute_path(sample, state, frames, actions, plan.recommended_path, target, "known_memory_ratio_optimal_plan")
            continue

        coin = best_safe_coin(sample, state)
        if coin is not None:
            target, actions, path = coin
            boss_result = execute_path(sample, state, frames, actions, path, target, "known_coin_improves_ratio_or_resource_gate")
            continue

        if state.known.get(sample.boss) == BOSS and not state.boss_defeated and state.resource >= sample.boss_config.revive_cost:
            found = safe_memory_path(sample, state, sample.boss, allow_boss=True)
            if found is not None:
                actions, path = found
                target = Target("boss", "boss", sample.boss, 0, len(actions), 0, 0, 0, True)
                boss_result = execute_path(sample, state, frames, actions, path, target, "boss_known_and_resource_gate_satisfied")
                continue

        frontier = safe_frontier_target(sample, state)
        if frontier is not None:
            target, actions, path = frontier
            boss_result = execute_path(sample, state, frames, actions, path, target, "systematic_safe_frontier_exploration")
            continue

        if state.known.get(sample.boss) == BOSS and not state.boss_defeated and state.resource >= sample.boss_config.revive_cost:
            found = safe_memory_path(sample, state, sample.boss, allow_boss=True)
            if found is not None:
                actions, path = found
                target = Target("boss", "boss", sample.boss, 0, len(actions), 0, 0, 0, True)
                boss_result = execute_path(sample, state, frames, actions, path, target, "no_frontier_force_known_boss")
                continue

        state.alive = False
        state.done = True
        break

    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    return RunResult(
        strategy="safe_ratio_planner",
        sample_id=sample.sample_id,
        difficulty=sample.difficulty,
        success=success,
        boss_success=state.boss_defeated,
        final_resource=state.resource,
        total_steps=state.steps,
        final_score=state.resource / max(1, state.steps),
        trap_count=len(state.triggered_traps),
        coin_count=len(state.collected_coins),
        boss_rounds=boss_result.min_rounds,
        runtime_ms=(time.perf_counter() - started) * 1000,
        frames=frames,
    )

