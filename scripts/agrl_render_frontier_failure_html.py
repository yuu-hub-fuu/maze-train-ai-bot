from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import (
    BOSS,
    BossResult,
    PlayerState,
    apply_move,
    frame,
    load_samples,
    observe_3x3,
    solve_boss_battle,
    target_from_high_action,
    tile_event,
)
from maze_gpt_agent.agrl_dqn import choose_dqn_action, load_dqn, safe_target_path_from_high_action
from maze_gpt_agent.agrl_safe_ratio_planner import safe_memory_path
from maze_gpt_agent.visualizer import render_run_html
from scripts.agrl_evaluate_dqn_frontier_value import select_frontier


def legacy_unsafe_boss_target(sample, state, high_action: str):
    if high_action != "GO_BOSS":
        return None
    if state.boss_defeated or state.known.get(sample.boss) != BOSS:
        return None
    found = safe_memory_path(sample, state, sample.boss, allow_boss=True)
    if found is None:
        return None
    actions, path = found
    if not actions:
        return None
    target = target_from_high_action(sample, state, high_action)
    return target, actions, path


def oracle_metrics(sample, score: float) -> dict:
    expert = sample.expert_solution or {}
    oracle_score = float(expert.get("final_score") or 0.0)
    oracle_gold = expert.get("final_resource")
    oracle_steps = expert.get("total_steps")
    if oracle_score > 0:
        score_of_oracle_pct = score / oracle_score * 100.0
        score_gap_pct = (oracle_score - score) / oracle_score * 100.0
    else:
        score_of_oracle_pct = 0.0
        score_gap_pct = 0.0
    return {
        "oracle_score": oracle_score,
        "oracle_gold": oracle_gold,
        "oracle_steps": oracle_steps,
        "score_of_oracle_pct": score_of_oracle_pct,
        "score_gap_pct": score_gap_pct,
    }


def run_with_frames(sample, model, frontier_policy: str, legacy_unsafe_boss: bool = False):
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    mode = "legacy unsafe boss gate" if legacy_unsafe_boss else "fixed boss resource gate"
    frames = [frame(sample, state, "START", f"sample={sample.sample_id}; policy={frontier_policy}; {mode}", None)]
    decisions = []
    boss_result = BossResult(False, 0, [], False, state.resource)
    max_decisions = sample.rows * sample.cols

    for _ in range(max_decisions):
        if not state.alive or state.done:
            break
        high_action = choose_dqn_action(model, sample, state)
        decision_note = high_action
        if high_action == "EXPLORE":
            chosen = select_frontier(sample, state, frontier_policy)
            decision_note = f"EXPLORE/{frontier_policy}"
        else:
            chosen = safe_target_path_from_high_action(sample, state, high_action)
            if chosen is None and legacy_unsafe_boss:
                chosen = legacy_unsafe_boss_target(sample, state, high_action)
                if chosen is not None:
                    decision_note = "GO_BOSS/legacy_unsafe_resource_gate"
        if chosen is None:
            state.alive = False
            state.done = True
            frames.append(frame(sample, state, high_action, "no_safe_target", None))
            decisions.append({"action": high_action, "event": "no_safe_target", "resource": state.resource, "steps": state.steps})
            break
        target, actions, path = chosen
        if not actions:
            state.alive = False
            state.done = True
            frames.append(frame(sample, state, high_action, "empty_path", target))
            decisions.append({"action": high_action, "event": "empty_path", "resource": state.resource, "steps": state.steps})
            break

        before_resource = state.resource
        before_steps = state.steps
        before_known = len(state.known)
        for action, pos in zip(actions, path[1:]):
            apply_move(sample, state, pos)
            event = tile_event(sample, state, pos)
            frames.append(frame(sample, state, action, f"{decision_note}; {event}; target={target.target_id}", target))
            if state.position == sample.boss and not state.boss_defeated:
                boss_result = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
                if boss_result.success and state.resource >= sample.boss_config.revive_cost:
                    state.boss_defeated = True
                    frames.append(frame(sample, state, "BOSS_FIGHT", "boss_clear:" + ",".join(boss_result.skill_sequence), target))
                else:
                    state.alive = False
                    state.done = True
                    frames.append(frame(sample, state, "BOSS_FIGHT", f"boss_fail: resource={state.resource}, required={sample.boss_config.revive_cost}", target))
                    break
            if state.position == sample.end:
                state.done = True
                if not state.boss_defeated:
                    state.alive = False
                    frames.append(frame(sample, state, "EXIT", "exit_before_boss_fail", target))
                else:
                    frames.append(frame(sample, state, "EXIT", "success", target))
                break
        decisions.append(
            {
                "raw_action": high_action,
                "decision": decision_note,
                "target": asdict(target) if target else None,
                "path_len": len(actions),
                "resource_delta": state.resource - before_resource,
                "steps_delta": state.steps - before_steps,
                "known_gain": len(state.known) - before_known,
                "resource": state.resource,
                "steps": state.steps,
                "boss_defeated": state.boss_defeated,
            }
        )

    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    score = state.resource / max(1, state.steps)
    summary = {
        "agent": f"DQN + {frontier_policy} frontier",
        "sample_id": sample.sample_id,
        "difficulty": sample.difficulty,
        "success": success,
        "boss_clear": state.boss_defeated,
        "gold": state.resource,
        "steps": state.steps,
        "score": score,
        "trap_count": len(state.triggered_traps),
        "coin_count": len(state.collected_coins),
        "final_pos": list(state.position),
        "failure_reason": "resource below boss revive cost when BOSS reached" if (state.position == sample.boss and not state.boss_defeated) else ("ended without clearing boss/exit" if not success else "success"),
        "boss_gate_mode": mode,
        "decisions": decisions,
    }
    summary.update(oracle_metrics(sample, score))
    return frames, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the failed info-density frontier trajectory as HTML.")
    parser.add_argument("--test", default="artifacts/agrl_oracle_ratio_15/test.json")
    parser.add_argument("--dqn", default="artifacts/agrl_oracle_ratio_15/dqn_policy.pt")
    parser.add_argument("--sample-id", default="test-ratio-prim-Medium-241137")
    parser.add_argument("--policy", default="info_density")
    parser.add_argument("--legacy-unsafe-boss", action="store_true")
    parser.add_argument("--out", default="artifacts/agrl_oracle_ratio_15/html_frontier_failure/test-ratio-prim-Medium-241137_info_density_failure.html")
    args = parser.parse_args()

    samples = load_samples(args.test)
    sample = next((x for x in samples if x.sample_id == args.sample_id), None)
    if sample is None:
        raise SystemExit(f"sample not found: {args.sample_id}")
    model, _metrics = load_dqn(args.dqn)
    frames, summary = run_with_frames(sample, model, args.policy, legacy_unsafe_boss=args.legacy_unsafe_boss)
    suffix = "legacy unsafe boss" if args.legacy_unsafe_boss else "fixed boss gate"
    title = f"Frontier Failure - {args.sample_id} - {args.policy} - {suffix}"
    render_run_html(args.out, title, frames, summary)
    print(summary)
    print(f"html: {args.out}")


if __name__ == "__main__":
    main()