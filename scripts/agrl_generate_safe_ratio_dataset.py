from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import ExpertSolution, MazeSample, RunResult, save_json
from maze_gpt_agent.agrl_ratio_optimal import generate_unbiased_maze, ratio_optimal_solution, solution_to_dict
from maze_gpt_agent.agrl_safe_ratio_planner import run_safe_ratio_planner


def run_to_solution(run: RunResult) -> dict[str, Any]:
    path = []
    for fr in run.frames:
        pos = tuple(fr["pos"])
        if not path or path[-1] != pos:
            path.append(pos)
    sol = ExpertSolution(
        recommended_targets=[],
        recommended_path=path,
        collected_coins=[],
        boss_skill_sequence=[],
        final_resource=run.final_resource,
        total_steps=run.total_steps,
        final_score=run.final_score,
        success=run.success,
    )
    return solution_to_dict(sol)


def traversable_count(sample: MazeSample) -> int:
    return sum(ch != "#" for row in sample.grid for ch in row)


def unique_path_coverage(sample: MazeSample, path: list) -> float:
    total = max(1, traversable_count(sample))
    return len({tuple(pos) for pos in path}) / total


def run_path_coverage(sample: MazeSample, run: RunResult) -> float:
    path = [tuple(fr["pos"]) for fr in run.frames if "pos" in fr]
    return unique_path_coverage(sample, path)


def attach_oracle_label(sample: MazeSample, teacher: RunResult, oracle) -> None:
    teacher_solution = run_to_solution(teacher)
    if oracle is None:
        sample.expert_solution = teacher_solution
        sample.expert_solution["label_source"] = "online_safe_ratio_teacher"
        sample.expert_solution["teacher_path_coverage"] = run_path_coverage(sample, teacher)
        return

    sample.expert_solution = solution_to_dict(oracle)
    sample.expert_solution["label_source"] = "bounded_full_map_ratio_oracle"
    sample.expert_solution["teacher_score"] = teacher.final_score
    sample.expert_solution["teacher_steps"] = teacher.total_steps
    sample.expert_solution["teacher_resource"] = teacher.final_resource
    sample.expert_solution["teacher_path_coverage"] = run_path_coverage(sample, teacher)
    sample.expert_solution["oracle_path_coverage"] = unique_path_coverage(sample, oracle.recommended_path)
    sample.expert_solution["teacher_score_ratio_to_oracle"] = teacher.final_score / max(1e-9, oracle.final_score)


def generation_audit(sample: MazeSample, teacher: RunResult, oracle) -> dict[str, Any]:
    return {
        "traversable_cells": traversable_count(sample),
        "teacher_path_coverage": run_path_coverage(sample, teacher),
        "oracle_available": oracle is not None,
        "oracle_path_coverage": unique_path_coverage(sample, oracle.recommended_path) if oracle is not None else None,
        "oracle_score": oracle.final_score if oracle is not None else None,
        "oracle_steps": oracle.total_steps if oracle is not None else None,
        "oracle_resource": oracle.final_resource if oracle is not None else None,
        "teacher_score": teacher.final_score,
        "teacher_steps": teacher.total_steps,
        "teacher_resource": teacher.final_resource,
        "teacher_score_ratio_to_oracle": teacher.final_score / max(1e-9, oracle.final_score) if oracle is not None else None,
        "coin_count": len(sample.coins),
        "trap_count": len(sample.traps),
    }


def build_split(
    split: str,
    count: int,
    size: int,
    seed: int,
    algorithm: str,
    max_attempt_mult: int,
    oracle_limit: int,
    oracle_mode: str,
) -> tuple[list[dict], dict]:
    diffs = ["Easy"] * 25 + ["Medium"] * 35 + ["Hard"] * 30 + ["Extreme"] * 10
    rows = []
    attempts = 0
    stats = {
        "attempts": 0,
        "teacher_success": 0,
        "oracle_success": 0,
        "oracle_skipped": 0,
        "oracle_required_rejects": 0,
    }
    max_attempts = max(1, count * max_attempt_mult)
    while len(rows) < count and attempts < max_attempts:
        difficulty = diffs[attempts % len(diffs)]
        sample_seed = seed + attempts
        attempts += 1
        sample = generate_unbiased_maze(size=size, seed=sample_seed, difficulty=difficulty, split=split, algorithm=algorithm)
        teacher = run_safe_ratio_planner(sample)
        stats["attempts"] = attempts
        if not teacher.success:
            if attempts % 25 == 0:
                print({"split": split, "attempts": attempts, "kept": len(rows), "last_teacher_success": False}, flush=True)
            continue
        stats["teacher_success"] += 1
        oracle = ratio_optimal_solution(sample, max_expansions=oracle_limit) if oracle_limit > 0 else None
        if oracle is None:
            stats["oracle_skipped"] += 1
            if oracle_mode == "required":
                stats["oracle_required_rejects"] += 1
                if attempts % 25 == 0:
                    print(
                        {
                            "split": split,
                            "attempts": attempts,
                            "kept": len(rows),
                            "last_teacher_score": teacher.final_score,
                            "oracle": "missing_required",
                            "oracle_required_rejects": stats["oracle_required_rejects"],
                        },
                        flush=True,
                    )
                continue
        else:
            stats["oracle_success"] += 1

        attach_oracle_label(sample, teacher, oracle)
        row = sample.to_dict()
        row["teacher_run_summary"] = teacher.summary()
        row["generation_audit"] = generation_audit(sample, teacher, oracle)
        rows.append(row)
        if len(rows) % 25 == 0 or len(rows) == count:
            print(
                {
                    "split": split,
                    "kept": len(rows),
                    "target": count,
                    "attempts": attempts,
                    "last_teacher_score": teacher.final_score,
                    "last_teacher_steps": teacher.total_steps,
                    "last_teacher_resource": teacher.final_resource,
                    "oracle_success": stats["oracle_success"],
                    "oracle_required_rejects": stats["oracle_required_rejects"],
                    "label_source": sample.expert_solution.get("label_source") if sample.expert_solution else None,
                },
                flush=True,
            )
    if len(rows) < count:
        raise RuntimeError(f"failed to build {split}: kept={len(rows)} attempts={attempts} target={count}")
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unbiased mazes labeled by full-map ratio oracle plus 3x3-memory online teacher audit.")
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--algorithm", default="mixed")
    parser.add_argument("--out-dir", default="artifacts/agrl_safe_ratio_15")
    parser.add_argument("--max-attempt-mult", type=int, default=80)
    parser.add_argument("--oracle-limit", type=int, default=200000)
    parser.add_argument(
        "--oracle-mode",
        choices=["required", "optional", "off"],
        default="required",
        help="required keeps only maps whose bounded full-map ratio oracle succeeds; optional stores it when available.",
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oracle_limit = 0 if args.oracle_mode == "off" else args.oracle_limit
    train, train_stats = build_split("train", args.train, args.size, args.seed, args.algorithm, args.max_attempt_mult, oracle_limit, args.oracle_mode)
    val, val_stats = build_split("val", args.val, args.size, args.seed + 100000, args.algorithm, args.max_attempt_mult, oracle_limit, args.oracle_mode)
    test, test_stats = build_split("test", args.test, args.size, args.seed + 200000, args.algorithm, args.max_attempt_mult, oracle_limit, args.oracle_mode)
    save_json(out / "train.json", train)
    save_json(out / "val.json", val)
    save_json(out / "test.json", test)
    manifest = {
        "generator": "unbiased randomized maze with random S/E/BOSS/coin/trap placement",
        "teacher": "safe_ratio_planner under strict 3x3 observation plus memory",
        "label_source": "bounded full-map ratio oracle" if args.oracle_mode == "required" else "oracle when available, otherwise online teacher",
        "metric": "final_remaining_resource / total_steps",
        "size": args.size,
        "seed": args.seed,
        "algorithm": args.algorithm,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "stats": {"train": train_stats, "val": val_stats, "test": test_stats},
        "oracle_mode": args.oracle_mode,
        "oracle_limit": oracle_limit,
        "oracle_note": "generated samples carry the bounded full-map ratio-optimal path as the primary expert label when available; online 3x3 teacher is retained as an execution/audit baseline",
    }
    save_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
