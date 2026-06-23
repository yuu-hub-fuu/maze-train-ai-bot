from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import BossResult, PlayerState, RunResult, aggregate_results, load_samples, observe_3x3, save_json
from maze_gpt_agent.agrl_dqn import choose_dqn_action, load_dqn, step_high_action
from maze_gpt_agent.agrl_safe_ratio_planner import run_safe_ratio_planner


def run_dqn_safe(sample, model, max_decisions: int | None = None) -> RunResult:
    started = time.perf_counter()
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    max_decisions = max_decisions or sample.rows * sample.cols
    decisions: list[dict[str, Any]] = []
    for _ in range(max_decisions):
        if not state.alive or state.done:
            break
        high_action = choose_dqn_action(model, sample, state)
        reward, done, info = step_high_action(sample, state, high_action)
        info = dict(info)
        info["reward"] = reward
        info["done"] = done
        info["pos"] = list(state.position)
        info["resource"] = state.resource
        info["steps"] = state.steps
        decisions.append(info)
        if info.get("event") in {"no_safe_target", "empty_path"}:
            state.alive = False
            state.done = True
            break
        if done:
            break
    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    return RunResult(
        strategy="dqn_safe_aligned_oracle_train",
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


def oracle_reference(samples) -> dict[str, Any]:
    vals = []
    teacher_vals = []
    coverages = []
    for sample in samples:
        sol = sample.expert_solution or {}
        if sol.get("label_source") == "bounded_full_map_ratio_oracle":
            vals.append(float(sol.get("final_score", 0.0)))
            teacher_vals.append(float(sol.get("teacher_score", 0.0)))
            if sol.get("oracle_path_coverage") is not None:
                coverages.append(float(sol["oracle_path_coverage"]))
    return {
        "count": len(vals),
        "avg_oracle_score": sum(vals) / max(1, len(vals)),
        "avg_teacher_score": sum(teacher_vals) / max(1, len(teacher_vals)),
        "avg_teacher_to_oracle": (sum(teacher_vals) / max(1, len(teacher_vals))) / max(1e-9, sum(vals) / max(1, len(vals))),
        "avg_oracle_path_coverage": sum(coverages) / max(1, len(coverages)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate safe-aligned DQN trained with full-map oracle labels.")
    parser.add_argument("--test", default="artifacts/agrl_oracle_ratio_15/test.json")
    parser.add_argument("--dqn", default="artifacts/agrl_oracle_ratio_15/dqn_policy.pt")
    parser.add_argument("--out", default="artifacts/agrl_oracle_ratio_15/evaluation_dqn_safe_oracle.json")
    parser.add_argument("--include-teacher", action="store_true")
    args = parser.parse_args()

    samples = load_samples(args.test)
    model, metrics = load_dqn(args.dqn)
    results = []
    for idx, sample in enumerate(samples, 1):
        results.append(run_dqn_safe(sample, model))
        if args.include_teacher:
            results.append(run_safe_ratio_planner(sample))
        if idx % 25 == 0 or idx == len(samples):
            print({"evaluated": idx, "total": len(samples)}, flush=True)
    summary = {
        "model_metrics": metrics,
        "aggregate": aggregate_results(results),
        "oracle_reference": oracle_reference(samples),
        "runs": [r.summary() for r in results],
        "vision_rule": "DQN observes only current 3x3 plus remembered cells; oracle labels are used only during training/warm-start and evaluation reference.",
    }
    save_json(args.out, summary)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps({"oracle_reference": summary["oracle_reference"]}, ensure_ascii=False, indent=2), flush=True)
    print(f"summary: {args.out}", flush=True)


if __name__ == "__main__":
    main()
