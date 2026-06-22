from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.dataset_builder import build_records, write_jsonl
from maze_gpt_agent.maze_env import save_json
from maze_gpt_agent.maze_generator import SCENARIOS, generate_course_maze


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate course-rule maze SFT samples.")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="artifacts/train/sft_records.jsonl")
    parser.add_argument("--maze-out", default="artifacts/mazes/train_mazes.json")
    args = parser.parse_args()

    records = []
    mazes = []
    for i in range(args.episodes):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        spec = generate_course_maze(size=args.size, seed=args.seed + i, scenario=scenario)
        episode_records = build_records(spec, target_return="high")
        if not episode_records:
            continue
        records.extend(episode_records)
        mazes.append(spec.to_dict())

    write_jsonl(args.out, records)
    save_json(args.maze_out, mazes)
    hf_path = Path(args.out).with_name("hf_sft_messages.jsonl")
    with hf_path.open("w", encoding="utf-8") as f:
        for rec in records:
            row = {
                "messages": [
                    {"role": "user", "content": rec["prompt"]},
                    {"role": "assistant", "content": rec["action"]},
                ],
                "scenario": rec["scenario"],
                "target_return": rec["target_return"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records from {len(mazes)} mazes")
    print(f"local records: {args.out}")
    print(f"huggingface SFT messages: {hf_path}")


if __name__ == "__main__":
    main()
