from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import save_json
from maze_gpt_agent.agrl_ratio_optimal import generate_unbiased_maze, ratio_optimal_solution, solution_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unbiased AGRL mazes labeled by ratio-optimal expert solutions.")
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--algorithm", default="mixed")
    parser.add_argument("--out-dir", default="artifacts/agrl_ratio_v4")
    parser.add_argument("--max-attempt-mult", type=int, default=40)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = ["Easy"] * 30 + ["Medium"] * 40 + ["Hard"] * 25 + ["Extreme"] * 5

    def build(split: str, count: int, offset: int) -> tuple[list[dict], int]:
        rows: list[dict] = []
        attempts = 0
        max_attempts = max(1, count * args.max_attempt_mult)
        while len(rows) < count and attempts < max_attempts:
            difficulty = ratios[attempts % len(ratios)]
            seed = args.seed + offset + attempts
            attempts += 1
            sample = generate_unbiased_maze(args.size, seed, difficulty, split, args.algorithm)
            solution = ratio_optimal_solution(sample)
            if attempts % 20 == 0:
                print(
                    f"{split}: attempts={attempts}, kept={len(rows)}/{count}, "
                    f"last_optimal={solution is not None}, seed={seed}, difficulty={difficulty}",
                    flush=True,
                )
            if solution is None:
                continue
            sample.expert_solution = solution_to_dict(solution)
            rows.append(sample.to_dict())
            if len(rows) % 50 == 0 or len(rows) == count:
                print(
                    f"{split}: kept {len(rows)}/{count}, attempts={attempts}, "
                    f"last_score={solution.final_score:.4f}, steps={solution.total_steps}, resource={solution.final_resource}",
                    flush=True,
                )
        if len(rows) < count:
            raise RuntimeError(f"failed to generate enough {split}: kept={len(rows)} attempts={attempts}")
        return rows, attempts

    train, train_attempts = build("train", args.train, 0)
    val, val_attempts = build("val", args.val, 100000)
    test, test_attempts = build("test", args.test, 200000)
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
        "generator": "unbiased random S/E/B/coin/trap placement with braided mazes",
        "teacher": "ratio_optimal_solution maximizes final_resource / total_steps",
        "metric": "final_resource / total_steps is the only expert-selection objective",
        "attempts": {"train": train_attempts, "val": val_attempts, "test": test_attempts},
        "vision_rule": "student agents observe only current 3x3 plus remembered cells; optimal teacher has full-map access only for label generation",
    }
    save_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
