from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import load_samples
from maze_gpt_agent.agrl_frontier_dqn import save_frontier_dqn, train_frontier_dqn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN with explicit frontier exploration options.")
    parser.add_argument("--train", default="artifacts/agrl_oracle_ratio_15/train.json")
    parser.add_argument("--out", default="artifacts/agrl_oracle_ratio_15/frontier_dqn.pt")
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument("--teacher-start", type=float, default=0.70)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    samples = load_samples(args.train)
    model, metrics = train_frontier_dqn(
        samples,
        episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        log_every=args.log_every,
        teacher_start=args.teacher_start,
    )
    save_frontier_dqn(args.out, model, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()