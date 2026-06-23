from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import run_strategy, save_json
from maze_gpt_agent.agrl_generators import generate_agrl_maze_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AGRL datasets filtered by successful classic expert runs.")
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithm", default="mixed")
    parser.add_argument("--out-dir", default="artifacts/agrl_large_valid")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = ["Easy"] * 30 + ["Medium"] * 40 + ["Hard"] * 25 + ["Extreme"] * 5

    def build(split: str, count: int, offset: int) -> list[dict]:
        rows = []
        attempts = 0
        while len(rows) < count:
            difficulty = ratios[attempts % len(ratios)]
            seed = args.seed + offset + attempts
            attempts += 1
            sample = generate_agrl_maze_v2(args.size, seed, difficulty, split, args.algorithm)
            expert = run_strategy(sample, "classic")
            if attempts % 50 == 0:
                print(
                    f"{split}: attempts={attempts}, kept={len(rows)}/{count}, "
                    f"last_success={expert.success}, last_seed={seed}, last_difficulty={difficulty}",
                    flush=True,
                )
            if not expert.success:
                continue
            sample.expert_solution = {
                "recommended_path": [f.get("pos") for f in expert.frames if f.get("pos") is not None],
                "final_resource": expert.final_resource,
                "total_steps": expert.total_steps,
                "final_score": expert.final_score,
                "success": expert.success,
            }
            rows.append(sample.to_dict())
            if len(rows) % 100 == 0 or len(rows) == count:
                print(f"{split}: kept {len(rows)}/{count}, attempts={attempts}", flush=True)
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
        "filter": "classic expert success only",
        "vision_rule": "agent observes only 3x3 plus remembered cells; unknown cells are hidden as ?",
    }
    save_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
