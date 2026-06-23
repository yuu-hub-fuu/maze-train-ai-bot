from __future__ import annotations

from dataclasses import asdict
import time

from .agrl_core import (
    BOSS,
    COIN,
    EXIT,
    BossResult,
    MazeSample,
    PlayerState,
    RunResult,
    Target,
    apply_move,
    frame,
    frontier_target,
    known_positions,
    observe_3x3,
    rcspp_path,
    solve_boss_battle,
    tile_event,
)


def choose_ratio_aware_target(sample: MazeSample, state: PlayerState) -> tuple[Target | None, str, str]:
    if state.boss_defeated:
        if state.known.get(sample.end) == EXIT:
            return Target("exit", "exit", sample.end, 0, 0, 0, 0, 0, True), "boss_done_go_exit", "GO_EXIT"
        return frontier_target(sample, state), "boss_done_explore_exit", "EXPLORE"

    known_coins = [p for p in known_positions(state, COIN) if p not in state.collected_coins]
    if state.known.get(sample.boss) == BOSS and state.resource >= sample.boss_config.revive_cost:
        boss_path = rcspp_path(sample, state, sample.boss, require_boss_resource=True)
        if boss_path.feasible and boss_path.actions:
            best_coin = best_known_coin(sample, state, known_coins)
            if best_coin is None or not coin_has_ratio_upside(sample, state, best_coin, boss_path.steps):
                return Target("boss", "boss", sample.boss, 0, boss_path.steps, boss_path.trap_loss, 0, 999, True), "resource_enough_ratio_go_boss", "GO_BOSS"

    coin = best_known_coin(sample, state, known_coins)
    if coin is not None:
        path = rcspp_path(sample, state, coin, require_boss_resource=False)
        return (
            Target(f"coin-{coin[0]}-{coin[1]}", "coin", coin, 50, path.steps, path.trap_loss, 0, 50 - path.steps - path.trap_loss, True),
            "known_coin_ratio_or_resource_need",
            "BEST_VALUE_GOLD",
        )

    explore = frontier_target(sample, state)
    if explore is not None:
        return explore, "explore_until_enough_information", "EXPLORE"

    if state.known.get(sample.boss) == BOSS:
        path = rcspp_path(sample, state, sample.boss, require_boss_resource=False)
        if path.feasible and path.actions:
            return Target("boss", "boss", sample.boss, 0, path.steps, path.trap_loss, 0, -path.steps, True), "no_more_frontier_force_boss", "GO_BOSS"
    return None, "no_feasible_target", "STOP"


def best_known_coin(sample: MazeSample, state: PlayerState, coins: list[tuple[int, int]]) -> tuple[int, int] | None:
    best: tuple[float, tuple[int, int]] | None = None
    for coin in coins:
        path = rcspp_path(sample, state, coin, require_boss_resource=False)
        if not path.feasible or not path.actions:
            continue
        score = 50 - path.trap_loss - 0.75 * path.steps
        if state.resource < sample.boss_config.revive_cost:
            score += 30
        cand = (score, coin)
        if best is None or cand > best:
            best = cand
    return best[1] if best else None


def coin_has_ratio_upside(sample: MazeSample, state: PlayerState, coin: tuple[int, int], boss_steps: int) -> bool:
    path = rcspp_path(sample, state, coin, require_boss_resource=False)
    if not path.feasible or not path.actions:
        return False
    current_ratio = state.resource / max(1, state.steps + boss_steps)
    projected_ratio = (state.resource + path.coin_gain - path.trap_loss) / max(1, state.steps + path.steps + boss_steps)
    return projected_ratio > current_ratio * 1.03


def run_online_planner(sample: MazeSample, max_steps: int | None = None) -> RunResult:
    started = time.perf_counter()
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    frames = [frame(sample, state, "START", "start", None)]
    max_steps = max_steps or sample.rows * sample.cols * 6
    boss_result = BossResult(False, 0, [], False, state.resource)
    seen_decisions: dict[tuple[tuple[int, int], int, bool, int], int] = {}

    while state.alive and not state.done and state.steps < max_steps:
        target, reason, high_action = choose_ratio_aware_target(sample, state)
        if target is None:
            state.alive = False
            state.done = True
            break
        key = (state.position, state.resource, state.boss_defeated, len(state.known))
        seen_decisions[key] = seen_decisions.get(key, 0) + 1
        if seen_decisions[key] > 4:
            state.alive = False
            state.done = True
            break
        path = rcspp_path(sample, state, target.position, require_boss_resource=(target.target_type == "boss"))
        if not path.feasible or not path.actions:
            if high_action != "EXPLORE":
                explore = frontier_target(sample, state)
                if explore is not None:
                    target = explore
                    reason = "fallback_explore_after_infeasible"
                    path = rcspp_path(sample, state, target.position)
            if not path.feasible or not path.actions:
                state.alive = False
                state.done = True
                break
        state.decision_history.append({"action": high_action, "reason": reason, "target": asdict(target), "path_score": path.score})
        for action, pos in zip(path.actions, path.path[1:]):
            apply_move(sample, state, pos)
            frames.append(frame(sample, state, action, tile_event(sample, state, pos), target))
            if state.steps >= max_steps:
                break
        if state.position == sample.boss and not state.boss_defeated:
            boss_result = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
            if boss_result.success and state.resource >= sample.boss_config.revive_cost:
                state.boss_defeated = True
                frames.append(frame(sample, state, "BOSS_FIGHT", "boss_clear:" + ",".join(boss_result.skill_sequence), target))
            else:
                state.alive = False
                state.done = True
                frames.append(frame(sample, state, "BOSS_FIGHT", "boss_fail", target))
        if state.position == sample.end:
            state.done = True
            if not state.boss_defeated:
                state.alive = False

    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    return RunResult(
        strategy="online_ratio_planner",
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
