from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import PlayerState, RunResult, aggregate_results, load_samples, observe_3x3, save_json
from maze_gpt_agent.agrl_frontier_dqn import choose_action, load_frontier_dqn, step_action


def run_frontier_dqn(sample, model, max_decisions: int | None = None) -> RunResult:
    started = time.perf_counter()
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    max_decisions = max_decisions or sample.rows * sample.cols
    decisions: list[dict[str, Any]] = []
    for _ in range(max_decisions):
        if not state.alive or state.done:
            break
        action = choose_action(model, sample, state)
        reward, done, info = step_action(sample, state, action)
        row = dict(info)
        row.update({"reward": reward, "done": done, "pos": list(state.position), "resource": state.resource, "steps": state.steps})
        decisions.append(row)
        if info.get("event") in {"no_safe_target", "empty_path"}:
            state.alive = False
            state.done = True
            break
        if done:
            break
    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    return RunResult(
        strategy="frontier_dqn",
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
    parser = argparse.ArgumentParser(description="Evaluate explicit-frontier DQN.")
    parser.add_argument("--test", default="artifacts/agrl_oracle_ratio_15/test.json")
    parser.add_argument("--model", default="artifacts/agrl_oracle_ratio_15/frontier_dqn.pt")
    parser.add_argument("--out", default="artifacts/agrl_oracle_ratio_15/evaluation_frontier_dqn.json")
    args = parser.parse_args()

    samples = load_samples(args.test)
    model, metrics = load_frontier_dqn(args.model)
    results = []
    for idx, sample in enumerate(samples, 1):
        results.append(run_frontier_dqn(sample, model))
        if idx % 50 == 0 or idx == len(samples):
            print({"evaluated": idx, "total": len(samples)}, flush=True)
    summary = {
        "model_metrics": metrics,
        "aggregate": aggregate_results(results),
        "runs": [r.summary() for r in results],
    }
    save_json(args.out, summary)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2), flush=True)
    print(f"summary: {args.out}", flush=True)


if __name__ == "__main__":
    main()