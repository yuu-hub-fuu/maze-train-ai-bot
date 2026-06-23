from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import (
    BossResult,
    PlayerState,
    RunResult,
    aggregate_results,
    apply_move,
    frame,
    load_samples,
    observe_3x3,
    rcspp_path,
    run_strategy,
    save_json,
    solve_boss_battle,
    target_from_high_action,
    tile_event,
)
from maze_gpt_agent.agrl_dqn import choose_dqn_action, load_dqn
from maze_gpt_agent.visualizer import render_run_html


def run_dqn_strategy(sample, model, max_steps: int | None = None) -> RunResult:
    started = time.perf_counter()
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    frames = [frame(sample, state, "START", "start", None)]
    max_steps = max_steps or sample.rows * sample.cols * 4
    boss_result = BossResult(False, 0, [], False, state.resource)
    while state.alive and not state.done and state.steps < max_steps:
        high_action = choose_dqn_action(model, sample, state)
        target = target_from_high_action(sample, state, high_action)
        if target is None:
            state.done = True
            state.alive = False
            break
        path = rcspp_path(sample, state, target.position, require_boss_resource=(target.target_type == "boss"))
        if not path.feasible or not path.actions:
            state.done = True
            state.alive = False
            break
        state.decision_history.append({"action": high_action, "reason": "dqn_high_level_policy", "target": target.__dict__, "path_score": path.score})
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
        strategy="dqn",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AGRL strategies including Q-learning and DQN.")
    parser.add_argument("--test", default="artifacts/agrl_large/test.json")
    parser.add_argument("--q-table", default="artifacts/agrl_large/q_table.json")
    parser.add_argument("--dqn", default="artifacts/agrl_large/dqn_policy.pt")
    parser.add_argument("--out", default="artifacts/agrl_large/evaluation_summary.json")
    parser.add_argument("--html", default="artifacts/agrl_large/demo_dqn.html")
    args = parser.parse_args()

    samples = load_samples(args.test)
    q_table = json.loads(Path(args.q_table).read_text(encoding="utf-8")) if Path(args.q_table).exists() else {}
    dqn_model = None
    if Path(args.dqn).exists():
        dqn_model, dqn_metrics = load_dqn(args.dqn)
        print(json.dumps({"loaded_dqn": args.dqn, "metrics": dqn_metrics}, ensure_ascii=False, indent=2))
    results = []
    demo = None
    for sample in samples:
        for strategy in ["shortest", "greedy3x3", "classic", "rl"]:
            result = run_strategy(sample, strategy, q_table=q_table if strategy == "rl" else None)
            results.append(result)
        if dqn_model is not None:
            dqn_result = run_dqn_strategy(sample, dqn_model)
            results.append(dqn_result)
            if demo is None:
                demo = dqn_result
    summary = {
        "aggregate": aggregate_results(results),
        "runs": [r.summary() for r in results],
        "vision_rule": "all evaluated strategies observe only current 3x3 plus remembered cells; unknown cells are hidden",
    }
    save_json(args.out, summary)
    if demo is not None:
        render_run_html(args.html, "AGRL-Maze DQN 3x3 Memory Demo", demo.frames, demo.summary())
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"summary: {args.out}")
    print(f"demo_html: {args.html}")


if __name__ == "__main__":
    main()
