from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import load_samples, save_json
from maze_gpt_agent.agrl_waypoint_q import save_model, train_waypoint_q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='artifacts/agrl_oracle_ratio_15/train.json')
    parser.add_argument('--episodes', type=int, default=2500)
    parser.add_argument('--out', default='artifacts/agrl_oracle_ratio_15/waypoint_q.pt')
    parser.add_argument('--metrics', default='artifacts/agrl_oracle_ratio_15/waypoint_q_metrics.json')
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--teacher-start', type=float, default=0.7)
    args = parser.parse_args()
    model, metrics = train_waypoint_q(load_samples(args.train), episodes=args.episodes, seed=args.seed, teacher_start=args.teacher_start)
    save_model(args.out, model, metrics)
    save_json(args.metrics, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f'waypoint_q: {args.out}')

if __name__ == '__main__':
    main()
