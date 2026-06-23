from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import load_samples, save_json
from maze_gpt_agent.agrl_cnn_bc import save_model, train_cnn_bc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN low-level oracle BC policy.")
    parser.add_argument("--train", default="artifacts/agrl_oracle_ratio_15/train.json")
    parser.add_argument("--val", default="artifacts/agrl_oracle_ratio_15/val.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--max-size", type=int, default=15)
    parser.add_argument("--out", default="artifacts/agrl_oracle_ratio_15/cnn_lowlevel_oracle_bc.pt")
    parser.add_argument("--metrics", default="artifacts/agrl_oracle_ratio_15/cnn_lowlevel_oracle_bc_metrics.json")
    args = parser.parse_args()
    model, metrics = train_cnn_bc(load_samples(args.train), load_samples(args.val), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_size=args.max_size)
    save_model(args.out, model, metrics)
    save_json(args.metrics, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"cnn_lowlevel_bc: {args.out}")


if __name__ == "__main__":
    main()
