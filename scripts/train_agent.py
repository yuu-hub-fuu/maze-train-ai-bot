from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agents import train_perceptron
from maze_gpt_agent.dataset_builder import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-train the local MazeGPT policy head from expert SFT records.")
    parser.add_argument("--data", default="artifacts/train/sft_records.jsonl")
    parser.add_argument("--out", default="artifacts/models/maze_policy.json")
    parser.add_argument("--epochs", type=int, default=7)
    args = parser.parse_args()

    records = read_jsonl(args.data, with_state=True)
    agent = train_perceptron(records, epochs=args.epochs)
    agent.save(args.out)
    print(f"trained on {len(records)} states")
    print(f"saved model: {args.out}")


if __name__ == "__main__":
    main()
