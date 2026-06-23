from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import PlayerState, RunResult, aggregate_results, load_samples, observe_3x3, save_json
from maze_gpt_agent.agrl_dqn import choose_dqn_action, load_dqn, safe_target_path_from_high_action, step_high_action


def projected_cashout(sample, state: PlayerState) -> tuple[float | None, str | None, int | None, int | None]:
    sim = state.clone()
    first_action = None
    if not sim.boss_defeated:
        if safe_target_path_from_high_action(sample, sim, "GO_BOSS") is None:
            return None, None, None, None
        first_action = "GO_BOSS"
        _reward, done, info = step_high_action(sample, sim, "GO_BOSS")
        if info.get("event") in {"no_safe_target", "empty_path"} or not sim.alive:
            return None, None, None, None
        if done and sim.position != sample.end:
            return None, None, None, None
    if safe_target_path_from_high_action(sample, sim, "GO_EXIT") is None:
        return None, first_action, None, None
    if first_action is None:
        first_action = "GO_EXIT"
    _reward, done, info = step_high_action(sample, sim, "GO_EXIT")
    if info.get("event") in {"no_safe_target", "empty_path"} or not sim.alive:
        return None, first_action, None, None
    if sim.position == sample.end and sim.boss_defeated:
        return sim.resource / max(1, sim.steps), first_action, sim.resource, sim.steps
    return None, first_action, None, None


def projected_action_then_cashout(sample, state: PlayerState, action: str):
    sim = state.clone()
    before_pos = sim.position
    _reward, done, info = step_high_action(sample, sim, action)
    if info.get("event") in {"no_safe_target", "empty_path"} or not sim.alive:
        return None
    if done:
        if sim.position == sample.end and sim.boss_defeated:
            return sim.resource / max(1, sim.steps)
        return None
    score, _first, _res, _steps = projected_cashout(sample, sim)
    return score


def gated_action(sample, state: PlayerState, dqn_action: str, margin: float, force_boss_known: bool) -> tuple[str, dict[str, Any]]:
    gate_info: dict[str, Any] = {"raw_action": dqn_action, "override": False}
    cash_score, first_cash, cash_resource, cash_steps = projected_cashout(sample, state)
    gate_info.update({"cash_score": cash_score, "first_cash": first_cash, "cash_resource": cash_resource, "cash_steps": cash_steps})

    if dqn_action in {"GO_BOSS", "GO_EXIT"}:
        return dqn_action, gate_info

    if force_boss_known and not state.boss_defeated and safe_target_path_from_high_action(sample, state, "GO_BOSS") is not None:
        projected = projected_action_then_cashout(sample, state, dqn_action)
        # If the proposed action cannot prove a better later cashout, cash BOSS now.
        if projected is None or (cash_score is not None and projected < cash_score + margin):
            gate_info.update({"override": True, "reason": "force_known_boss", "projected_action_score": projected})
            return "GO_BOSS", gate_info

    if cash_score is None or first_cash is None:
        return dqn_action, gate_info

    if dqn_action in {"BEST_VALUE_GOLD", "NEAREST_GOLD", "MAIN_PATH_GOLD", "AVOID_TRAP", "EXPLORE"}:
        projected = projected_action_then_cashout(sample, state, dqn_action)
        gate_info["projected_action_score"] = projected
        if projected is None or projected < cash_score + margin:
            gate_info.update({"override": True, "reason": "cashout_ratio_better"})
            return first_cash, gate_info
    return dqn_action, gate_info


def run_dqn_ratio_gate(sample, model, margin: float, force_boss_known: bool, max_decisions: int | None = None) -> RunResult:
    started = time.perf_counter()
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    max_decisions = max_decisions or sample.rows * sample.cols
    decisions: list[dict[str, Any]] = []
    for _ in range(max_decisions):
        if not state.alive or state.done:
            break
        raw = choose_dqn_action(model, sample, state)
        action, gate = gated_action(sample, state, raw, margin=margin, force_boss_known=force_boss_known)
        reward, done, info = step_high_action(sample, state, action)
        row = dict(info)
        row.update(gate)
        row["action"] = action
        row["reward"] = reward
        row["done"] = done
        row["pos"] = list(state.position)
        row["resource"] = state.resource
        row["steps"] = state.steps
        decisions.append(row)
        if info.get("event") in {"no_safe_target", "empty_path"}:
            state.alive = False
            state.done = True
            break
        if done:
            break
    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    suffix = f"m{margin}_boss{int(force_boss_known)}"
    return RunResult(
        strategy=f"dqn_ratio_gate_{suffix}",
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
    parser = argparse.ArgumentParser(description="Evaluate conservative ratio gate around stable DQN.")
    parser.add_argument("--test", default="artifacts/agrl_oracle_ratio_15/test.json")
    parser.add_argument("--dqn", default="artifacts/agrl_oracle_ratio_15/dqn_policy.pt")
    parser.add_argument("--out", default="artifacts/agrl_oracle_ratio_15/evaluation_dqn_ratio_gate.json")
    parser.add_argument("--margins", default="0,0.03,0.06,0.10")
    args = parser.parse_args()
    samples = load_samples(args.test)
    model, metrics = load_dqn(args.dqn)
    results = []
    margins = [float(x) for x in args.margins.split(",") if x.strip()]
    for force_boss in [False, True]:
        for margin in margins:
            for idx, sample in enumerate(samples, 1):
                results.append(run_dqn_ratio_gate(sample, model, margin, force_boss))
                if idx % 50 == 0 or idx == len(samples):
                    print({"force_boss": force_boss, "margin": margin, "evaluated": idx, "total": len(samples)}, flush=True)
    summary = {
        "model_metrics": metrics,
        "aggregate": aggregate_results(results),
        "runs": [r.summary() for r in results],
        "gate": "Conservative analytic ratio gate around stable DQN; overrides exploration/coin only when immediate cashout has better projected final resource/steps.",
    }
    save_json(args.out, summary)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2), flush=True)
    print(f"summary: {args.out}", flush=True)


if __name__ == "__main__":
    main()
