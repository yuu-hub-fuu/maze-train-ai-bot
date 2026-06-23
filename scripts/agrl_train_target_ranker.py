from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import load_samples, save_json
from maze_gpt_agent.agrl_target_ranker import save_model, train_ranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural target ranker from oracle paths.")
    parser.add_argument("--train", default="artifacts/agrl_oracle_ratio_15/train.json")
    parser.add_argument("--val", default="artifacts/agrl_oracle_ratio_15/val.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="artifacts/agrl_oracle_ratio_15/target_ranker.pt")
    parser.add_argument("--metrics", default="artifacts/agrl_oracle_ratio_15/target_ranker_metrics.json")
    args = parser.parse_args()

    model, metrics = train_ranker(load_samples(args.train), load_samples(args.val), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    save_model(args.out, model, metrics)
    save_json(args.metrics, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"target_ranker: {args.out}")


if __name__ == "__main__":
    main()
