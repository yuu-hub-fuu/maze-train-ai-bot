from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import load_samples, save_json
from maze_gpt_agent.agrl_dqn import save_dqn, train_dqn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN high-level AGRL policy.")
    parser.add_argument("--train", default="artifacts/agrl_large/train.json")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--out", default="artifacts/agrl_large/dqn_policy.pt")
    parser.add_argument("--metrics", default="artifacts/agrl_large/dqn_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-start", type=float, default=0.75)
    args = parser.parse_args()

    samples = load_samples(args.train)
    model, metrics = train_dqn(
        samples,
        episodes=args.episodes,
        seed=args.seed,
        log_every=max(50, args.episodes // 20),
        teacher_start=args.teacher_start,
    )
    save_dqn(args.out, model, metrics)
    save_json(args.metrics, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"dqn_policy: {args.out}")


if __name__ == "__main__":
    main()
