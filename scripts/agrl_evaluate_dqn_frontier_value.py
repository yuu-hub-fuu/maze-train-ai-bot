from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import (
    BOSS,
    EXIT,
    WALL,
    MOVES,
    BossResult,
    PlayerState,
    RunResult,
    Target,
    aggregate_results,
    apply_move,
    load_samples,
    observe_3x3,
    save_json,
    solve_boss_battle,
    tile_event,
)
from maze_gpt_agent.agrl_dqn import choose_dqn_action, load_dqn, safe_target_path_from_high_action, step_high_action
from maze_gpt_agent.agrl_safe_ratio_planner import safe_memory_path


def unknown_adjacent(sample, state: PlayerState, pos: tuple[int, int]) -> int:
    total = 0
    for dr, dc in MOVES.values():
        nxt = (pos[0] + dr, pos[1] + dc)
        if 0 <= nxt[0] < sample.rows and 0 <= nxt[1] < sample.cols and nxt not in state.known:
            total += 1
    return total


def path_trap_loss(sample, state: PlayerState, path: list[tuple[int, int]]) -> int:
    return sum(30 for p in path[1:] if sample.char_at(p) == "T" and p not in state.triggered_traps)


def known_distance(sample, state: PlayerState, pos: tuple[int, int]) -> int | None:
    found = safe_memory_path(sample, state, pos, allow_boss=(pos == sample.boss), allow_exit=(pos == sample.end))
    if found is None:
        return None
    return len(found[0])


def cashout_distance_after(sample, state: PlayerState, frontier: tuple[int, int], frontier_path: list[tuple[int, int]]) -> int:
    sim = state.clone()
    for pos in frontier_path[1:]:
        apply_move(sample, sim, pos)
    dist = 0
    if not sim.boss_defeated and sim.known.get(sample.boss) == BOSS and sim.resource >= sample.boss_config.revive_cost:
        found = safe_memory_path(sample, sim, sample.boss, allow_boss=True)
        if found is not None:
            dist += len(found[0])
            for pos in found[1][1:]:
                apply_move(sample, sim, pos)
            sim.boss_defeated = True
    if sim.boss_defeated and sim.known.get(sample.end) == EXIT:
        found = safe_memory_path(sample, sim, sample.end, allow_exit=True)
        if found is not None:
            dist += len(found[0])
    return dist


def frontier_candidates(sample, state: PlayerState):
    out = []
    for pos, ch in state.known.items():
        if ch == WALL:
            continue
        unk = unknown_adjacent(sample, state, pos)
        if unk <= 0:
            continue
        found = safe_memory_path(sample, state, pos)
        if found is None:
            continue
        actions, path = found
        if not actions:
            continue
        risk = path_trap_loss(sample, state, path)
        target = Target(f"frontier-{pos[0]}-{pos[1]}", "explore", pos, 0, len(actions), risk, 0, unk - len(actions), True)
        out.append((target, actions, path, unk, risk))
    return out


def select_frontier(sample, state: PlayerState, policy: str):
    cands = frontier_candidates(sample, state)
    if not cands:
        return None
    nearest = min(cands, key=lambda x: (len(x[1]), -x[3], x[0].position))
    density = max(cands, key=lambda x: (x[3] / max(1, len(x[1])), -x[4], -len(x[1])))
    cashout = max(cands, key=lambda x: (2.0 * x[3] - 1.0 * len(x[1]) - 0.35 * cashout_distance_after(sample, state, x[0].position, x[2]) - 0.05 * x[4]))
    if policy == "nearest":
        return nearest[:3]
    if policy == "info_density":
        return density[:3]
    if policy == "info_density_guarded":
        nearest_len = max(1, len(nearest[1]))
        density_len = len(density[1])
        nearest_density = nearest[3] / nearest_len
        chosen_density = density[3] / max(1, density_len)
        if (
            density_len <= nearest_len + 4
            and density_len <= max(2, nearest_len * 2)
            and density[4] <= nearest[4]
            and chosen_density >= nearest_density + 0.20
        ):
            return density[:3]
        return nearest[:3]
    if policy == "info_density_tight":
        nearest_len = max(1, len(nearest[1]))
        density_len = len(density[1])
        nearest_density = nearest[3] / nearest_len
        chosen_density = density[3] / max(1, density_len)
        if density_len <= nearest_len + 2 and density[4] <= nearest[4] and chosen_density >= nearest_density + 0.25:
            return density[:3]
        return nearest[:3]
    if policy == "info_budget":
        # Prefer high information but cap long detours; this is the value-of-information gate.
        return max(cands, key=lambda x: (2.5 * x[3] - 1.15 * len(x[1]) - 0.04 * x[4], -len(x[1])))[:3]
    if policy == "cashout_aware":
        return cashout[:3]
    if policy == "cashout_guarded":
        nearest_len = max(1, len(nearest[1]))
        cashout_len = len(cashout[1])
        nearest_density = nearest[3] / nearest_len
        chosen_density = cashout[3] / max(1, cashout_len)
        if (
            cashout_len <= nearest_len + 3
            and cashout_len <= max(2, nearest_len * 2)
            and cashout[4] <= nearest[4]
            and chosen_density >= nearest_density
        ):
            return cashout[:3]
        return nearest[:3]
    if policy == "far_then_cashout":
        # Keep as an intentionally aggressive comparator.
        return max(cands, key=lambda x: (x[3], len(x[1]) * 0.15 - 0.2 * cashout_distance_after(sample, state, x[0].position, x[2])))[:3]
    raise ValueError(policy)


def execute_custom_path(sample, state: PlayerState, action: str, target: Target, actions: list[str], path: list[tuple[int, int]]):
    before_resource = state.resource
    before_steps = state.steps
    before_known = len(state.known)
    boss_result = BossResult(False, 0, [], False, state.resource)
    for pos in path[1:]:
        apply_move(sample, state, pos)
        if state.position == sample.boss and not state.boss_defeated:
            boss_result = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
            if boss_result.success and state.resource >= sample.boss_config.revive_cost:
                state.boss_defeated = True
            else:
                state.alive = False
                state.done = True
                break
        if state.position == sample.end:
            state.done = True
            if not state.boss_defeated:
                state.alive = False
            break
    step_cost = state.steps - before_steps
    known_gain = len(state.known) - before_known
    reward = (state.resource - before_resource) - step_cost + 0.1 * known_gain
    done = bool(state.done or not state.alive)
    if state.position == sample.end and state.boss_defeated and state.alive:
        reward += 100 + 80 * state.resource / max(1, state.steps)
        done = True
    return reward, done, {"event": "custom_explore", "action": action, "target": asdict(target), "known_gain": known_gain, "path_len": len(actions)}


def run_dqn_frontier_value(sample, model, frontier_policy: str, max_decisions: int | None = None) -> RunResult:
    started = time.perf_counter()
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    max_decisions = max_decisions or sample.rows * sample.cols
    decisions: list[dict[str, Any]] = []
    for _ in range(max_decisions):
        if not state.alive or state.done:
            break
        high_action = choose_dqn_action(model, sample, state)
        if high_action == "EXPLORE":
            picked = select_frontier(sample, state, frontier_policy)
            if picked is None:
                reward, done, info = step_high_action(sample, state, high_action)
            else:
                target, actions, path = picked
                reward, done, info = execute_custom_path(sample, state, high_action, target, actions, path)
        else:
            reward, done, info = step_high_action(sample, state, high_action)
        row = dict(info)
        row.update({"raw_action": high_action, "reward": reward, "done": done, "pos": list(state.position), "resource": state.resource, "steps": state.steps})
        decisions.append(row)
        if info.get("event") in {"no_safe_target", "empty_path"}:
            state.alive = False
            state.done = True
            break
        if done:
            break
    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    return RunResult(
        strategy=f"dqn_frontier_{frontier_policy}",
        sample_id=sample.sample_id,
        difficulty=sample.difficulty,
        success=success,
        boss_success=state.boss_defeated,
        final_resource=state.resource,
        total_steps=state.steps,
        final_score=state.resource / max(1, state.steps),
        trap_count=len(state.triggered_traps),
        coin_count=len(state.collected_coins),
        boss_rounds=0,
        runtime_ms=(time.perf_counter() - started) * 1000,
        frames=decisions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate value-of-information frontier selectors under stable DQN.")
    parser.add_argument("--test", default="artifacts/agrl_oracle_ratio_15/test.json")
    parser.add_argument("--dqn", default="artifacts/agrl_oracle_ratio_15/dqn_policy.pt")
    parser.add_argument("--out", default="artifacts/agrl_oracle_ratio_15/evaluation_dqn_frontier_value.json")
    parser.add_argument("--policies", default="nearest,info_density,info_density_guarded,info_density_tight,info_budget,cashout_aware,cashout_guarded,far_then_cashout")
    args = parser.parse_args()
    samples = load_samples(args.test)
    model, metrics = load_dqn(args.dqn)
    results = []
    for policy in [x.strip() for x in args.policies.split(",") if x.strip()]:
        for idx, sample in enumerate(samples, 1):
            results.append(run_dqn_frontier_value(sample, model, policy))
            if idx % 50 == 0 or idx == len(samples):
                print({"policy": policy, "evaluated": idx, "total": len(samples)}, flush=True)
    summary = {
        "model_metrics": metrics,
        "aggregate": aggregate_results(results),
        "runs": [r.summary() for r in results],
        "note": "Only changes DQN EXPLORE frontier selection; model weights and other actions are unchanged.",
    }
    save_json(args.out, summary)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2), flush=True)
    print(f"summary: {args.out}", flush=True)


if __name__ == "__main__":
    main()
