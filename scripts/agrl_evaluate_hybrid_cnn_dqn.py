from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import BossResult, MOVES, PlayerState, RunResult, aggregate_results, apply_move, frame, load_samples, observe_3x3, save_json, solve_boss_battle, tile_event
from maze_gpt_agent.agrl_cnn_bc import encode_state, load_model as load_cnn
from maze_gpt_agent.agrl_lowlevel_bc import ID_TO_ACTION, valid_move_mask
from maze_gpt_agent.agrl_dqn import choose_dqn_action, load_dqn, step_high_action


def cnn_action_conf(model, sample, state, max_size=15):
    g, s = encode_state(sample, state, max_size)
    with torch.no_grad():
        logits = model(torch.tensor(g[None], dtype=torch.float32), torch.tensor(s[None], dtype=torch.float32)).squeeze(0)
    mask = valid_move_mask(sample, state)
    if not mask.any():
        return None, 0.0, 0.0
    mask_t = torch.tensor(mask, dtype=torch.bool)
    masked = logits.masked_fill(~mask_t, -1e9)
    probs = F.softmax(masked, dim=0)
    vals, ids = probs.sort(descending=True)
    action = ID_TO_ACTION[int(ids[0].item())]
    conf = float(vals[0].item())
    margin = float((vals[0] - vals[1]).item()) if len(vals) > 1 else conf
    return action, conf, margin


def run_hybrid(sample, cnn_model, dqn_model, threshold=0.88, margin_threshold=0.25, max_size=15):
    started = time.perf_counter()
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    frames = [frame(sample, state, "START", "start", None)]
    boss_result = BossResult(False, 0, [], False, state.resource)
    visits = {}
    cnn_steps = 0
    dqn_steps = 0
    max_decisions = sample.rows * sample.cols
    for _ in range(max_decisions):
        if not state.alive or state.done:
            break
        visits[state.position] = visits.get(state.position, 0) + 1
        action, conf, margin = cnn_action_conf(cnn_model, sample, state, max_size)
        use_cnn = action is not None and conf >= threshold and margin >= margin_threshold and visits.get(state.position, 0) <= 2
        if use_cnn:
            dr, dc = MOVES[action]
            nxt = (state.position[0] + dr, state.position[1] + dc)
            apply_move(sample, state, nxt)
            cnn_steps += 1
            frames.append(frame(sample, state, action, f"cnn_conf={conf:.3f},margin={margin:.3f};" + tile_event(sample, state, nxt), None))
            if visits.get(state.position, 0) > 6:
                use_cnn = False
            if state.position == sample.boss and not state.boss_defeated:
                boss_result = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
                if boss_result.success and state.resource >= sample.boss_config.revive_cost:
                    state.boss_defeated = True
                    frames.append(frame(sample, state, "BOSS_FIGHT", "boss_clear:" + ",".join(boss_result.skill_sequence), None))
                else:
                    state.alive = False
                    state.done = True
                    frames.append(frame(sample, state, "BOSS_FIGHT", "boss_fail", None))
            if state.position == sample.end:
                state.done = True
                if not state.boss_defeated:
                    state.alive = False
            continue
        high_action = choose_dqn_action(dqn_model, sample, state)
        reward, done, info = step_high_action(sample, state, high_action)
        dqn_steps += 1
        frames.append(frame(sample, state, "DQN_" + high_action, f"fallback reward={reward:.2f} event={info.get('event')}", None))
        if info.get("event") in {"no_safe_target", "empty_path"}:
            state.alive = False
            state.done = True
            break
        if done:
            break
    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    result = RunResult(
        strategy=f"hybrid_cnn_dqn_t{threshold}",
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
    result.frames[0]["hybrid_counts"] = {"cnn_steps": cnn_steps, "dqn_steps": dqn_steps}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='artifacts/agrl_oracle_ratio_15/test.json')
    parser.add_argument('--cnn', default='artifacts/agrl_oracle_ratio_15/cnn_lowlevel_oracle_bc.pt')
    parser.add_argument('--dqn', default='artifacts/agrl_oracle_ratio_15/dqn_policy.pt')
    parser.add_argument('--out', default='artifacts/agrl_oracle_ratio_15/evaluation_hybrid_cnn_dqn.json')
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--margin-threshold', type=float, default=0.3)
    args = parser.parse_args()
    samples = load_samples(args.test)
    cnn, cnn_metrics = load_cnn(args.cnn)
    dqn, dqn_metrics = load_dqn(args.dqn)
    results = []
    for idx, sample in enumerate(samples, 1):
        results.append(run_hybrid(sample, cnn, dqn, threshold=args.threshold, margin_threshold=args.margin_threshold))
        if idx % 25 == 0 or idx == len(samples):
            print({'evaluated': idx, 'total': len(samples)}, flush=True)
    summary = {
        'aggregate': aggregate_results(results),
        'runs': [r.summary() for r in results],
        'cnn_metrics': cnn_metrics,
        'dqn_metrics': dqn_metrics,
        'threshold': args.threshold,
        'margin_threshold': args.margin_threshold,
    }
    save_json(args.out, summary)
    print(json.dumps(summary['aggregate'], ensure_ascii=False, indent=2))
    print(f"summary: {args.out}")


if __name__ == '__main__':
    main()
