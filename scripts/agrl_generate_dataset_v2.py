from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import run_strategy, save_json
from maze_gpt_agent.agrl_generators import generate_agrl_maze_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate large AGRL-Maze datasets with DFS/Prim/Kruskal/recursive division.")
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithm", default="mixed", choices=["dfs", "prim", "kruskal", "division", "mixed"])
    parser.add_argument("--out-dir", default="artifacts/agrl_large")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = ["Easy"] * 30 + ["Medium"] * 40 + ["Hard"] * 25 + ["Extreme"] * 5

    def build(split: str, count: int, offset: int) -> list[dict]:
        rows = []
        for i in range(count):
            difficulty = ratios[i % len(ratios)]
            sample = generate_agrl_maze_v2(args.size, args.seed + offset + i, difficulty, split, args.algorithm)
            expert = run_strategy(sample, "classic")
            sample.expert_solution = {
                "recommended_path": [f.get("pos") for f in expert.frames if f.get("pos") is not None],
                "final_resource": expert.final_resource,
                "total_steps": expert.total_steps,
                "final_score": expert.final_score,
                "success": expert.success,
            }
            rows.append(sample.to_dict())
            if (i + 1) % 100 == 0 or i + 1 == count:
                print(f"{split}: {i+1}/{count}", flush=True)
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
        "algorithm": args.algorithm,
        "generators": ["dfs", "prim", "kruskal", "division"] if args.algorithm == "mixed" else [args.algorithm],
        "vision_rule": "agent observes only 3x3 around current position; previously observed cells are stored in memory; unknown cells are hidden as ?",
    }
    save_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
