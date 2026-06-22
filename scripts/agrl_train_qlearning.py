from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import load_samples, run_strategy, save_json, train_q_learning


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AGRL high-level Q-learning policy.")
    parser.add_argument("--train", default="artifacts/agrl/train.json")
    parser.add_argument("--val", default="artifacts/agrl/val.json")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--out", default="artifacts/agrl/q_table.json")
    parser.add_argument("--curve", default="artifacts/agrl/qlearning_curve.json")
    args = parser.parse_args()

    train_samples = load_samples(args.train)
    val_samples = load_samples(args.val)
    q_table = train_q_learning(train_samples, episodes=args.episodes)
    save_json(args.out, q_table)

    val_results = [run_strategy(s, "rl", q_table) for s in val_samples]
    curve = {
        "episodes": args.episodes,
        "q_states": len(q_table),
        "val_success_rate": sum(r.success for r in val_results) / max(1, len(val_results)),
        "val_avg_score": sum(r.final_score for r in val_results) / max(1, len(val_results)),
        "val_avg_steps": sum(r.total_steps for r in val_results) / max(1, len(val_results)),
    }
    save_json(args.curve, curve)
    print(json.dumps(curve, ensure_ascii=False, indent=2))
    print(f"q_table: {args.out}")


if __name__ == "__main__":
    main()
