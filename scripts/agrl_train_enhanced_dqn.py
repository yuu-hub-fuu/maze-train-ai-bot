from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import load_samples, save_json
from maze_gpt_agent.agrl_enhanced_dqn import save_model, train_enhanced_dqn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN Dueling Double DQN for AGRL online maze agent.")
    parser.add_argument("--train", default="artifacts/agrl_oracle_ratio_15/train.json")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--out", default="artifacts/agrl_oracle_ratio_15/enhanced_dqn.pt")
    parser.add_argument("--metrics", default="artifacts/agrl_oracle_ratio_15/enhanced_dqn_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-start", type=float, default=0.6)
    args = parser.parse_args()
    model, metrics = train_enhanced_dqn(load_samples(args.train), episodes=args.episodes, seed=args.seed, teacher_start=args.teacher_start, log_every=max(100, args.episodes // 20))
    save_model(args.out, model, metrics)
    save_json(args.metrics, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"enhanced_dqn: {args.out}")


if __name__ == "__main__":
    main()
