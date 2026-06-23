from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import MazeSample, PlayerState, apply_move, load_samples, observe_3x3, solve_boss_battle
from maze_gpt_agent.agrl_dqn import state_vector as dqn_state_vector
from maze_gpt_agent.agrl_lowlevel_bc import action_between, state_vector as low_state_vector


def oracle_path(sample: MazeSample):
    raw = (sample.expert_solution or {}).get('recommended_path') or []
    return [tuple(p) for p in raw]


def analyze(samples, limit=None):
    dqn_labels = defaultdict(Counter)
    dqn_count = 0
    low_labels = defaultdict(Counter)
    low_count = 0
    for sample in samples[: limit or len(samples)]:
        path = oracle_path(sample)
        if len(path) < 2:
            continue
        state = PlayerState(position=sample.start, path_history=[sample.start])
        observe_3x3(sample, state)
        for cur, nxt in zip(path, path[1:]):
            if state.position != cur:
                break
            action = action_between(cur, nxt)
            # DQN's compact state aliases many different absolute/memory states.
            dkey = tuple(round(float(x), 2) for x in dqn_state_vector(sample, state))
            dqn_labels[dkey][action] += 1
            dqn_count += 1
            # Low-level BC's full remembered-map state should have fewer exact collisions.
            lkey = tuple(low_state_vector(sample, state, 15).astype('int8').tolist())
            low_labels[lkey][action] += 1
            low_count += 1
            apply_move(sample, state, nxt)
            if state.position == sample.boss and not state.boss_defeated:
                boss = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
                if boss.success and state.resource >= sample.boss_config.revive_cost:
                    state.boss_defeated = True
                else:
                    break
            if state.position == sample.end:
                break

    def summarize(table, total):
        conflict_keys = {k: c for k, c in table.items() if len(c) > 1}
        conflict_steps = sum(sum(c.values()) for c in conflict_keys.values())
        entropy_examples = sorted(
            [(sum(c.values()), dict(c)) for c in conflict_keys.values()],
            key=lambda x: x[0], reverse=True,
        )[:10]
        return {
            'steps': total,
            'unique_state_keys': len(table),
            'conflict_keys': len(conflict_keys),
            'conflict_steps': conflict_steps,
            'conflict_step_rate': conflict_steps / max(1, total),
            'top_conflicts': entropy_examples,
        }

    return {'dqn_compact_state': summarize(dqn_labels, dqn_count), 'lowlevel_memory_state': summarize(low_labels, low_count)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='artifacts/agrl_oracle_ratio_15/train.json')
    parser.add_argument('--out', default='artifacts/agrl_oracle_ratio_15/state_alias_analysis.json')
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args()
    result = analyze(load_samples(args.data), args.limit)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
