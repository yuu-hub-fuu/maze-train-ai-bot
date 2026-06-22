from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import generate_agrl_maze, run_strategy, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AGRL-Maze train/val/test datasets.")
    parser.add_argument("--train", type=int, default=120)
    parser.add_argument("--val", type=int, default=30)
    parser.add_argument("--test", type=int, default=30)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="artifacts/agrl")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = ["Easy"] * 30 + ["Medium"] * 40 + ["Hard"] * 25 + ["Extreme"] * 5

    def build(split: str, count: int, seed_offset: int) -> list[dict]:
        rows = []
        for i in range(count):
            difficulty = ratios[i % len(ratios)]
            sample = generate_agrl_maze(size=args.size, seed=args.seed + seed_offset + i, difficulty=difficulty, split=split)
            expert = run_strategy(sample, "classic")
            sample.expert_solution = {
                "recommended_path": [f.get("pos") for f in expert.frames if f.get("pos") is not None],
                "final_resource": expert.final_resource,
                "total_steps": expert.total_steps,
                "final_score": expert.final_score,
                "success": expert.success,
            }
            rows.append(sample.to_dict())
        return rows

    train = build("train", args.train, 0)
    val = build("val", args.val, 100000)
    test = build("test", args.test, 200000)
    save_json(out_dir / "train.json", train)
    save_json(out_dir / "val.json", val)
    save_json(out_dir / "test.json", test)
    manifest = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "size": args.size,
        "seed": args.seed,
        "vision_rule": "agent observes only 3x3 around current position; previously observed cells are stored in memory; unknown cells are hidden as ?",
    }
    save_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"dataset_dir: {out_dir}")


if __name__ == "__main__":
    main()
