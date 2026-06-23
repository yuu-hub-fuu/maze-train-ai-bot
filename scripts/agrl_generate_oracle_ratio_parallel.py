from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import ExpertSolution, MazeSample, RunResult, save_json
from maze_gpt_agent.agrl_ratio_optimal import generate_unbiased_maze, ratio_optimal_solution, solution_to_dict
from maze_gpt_agent.agrl_safe_ratio_planner import run_safe_ratio_planner


DIFFICULTIES = ["Easy"] * 25 + ["Medium"] * 35 + ["Hard"] * 30 + ["Extreme"] * 10


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
    if oracle is None:
        sample.expert_solution = run_to_solution(teacher)
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


def build_attempt(args: tuple[str, int, int, str, int, str, int]) -> dict[str, Any]:
    split, attempt, size, algorithm, seed_base, oracle_mode, oracle_limit = args
    difficulty = DIFFICULTIES[attempt % len(DIFFICULTIES)]
    sample_seed = seed_base + attempt
    try:
        sample = generate_unbiased_maze(size=size, seed=sample_seed, difficulty=difficulty, split=split, algorithm=algorithm)
        teacher = run_safe_ratio_planner(sample)
        if not teacher.success:
            return {"status": "teacher_fail", "attempt": attempt, "difficulty": difficulty}
        oracle = ratio_optimal_solution(sample, max_expansions=oracle_limit) if oracle_limit > 0 else None
        if oracle is None and oracle_mode == "required":
            return {
                "status": "oracle_fail",
                "attempt": attempt,
                "difficulty": difficulty,
                "teacher_score": teacher.final_score,
            }
        attach_oracle_label(sample, teacher, oracle)
        row = sample.to_dict()
        row["teacher_run_summary"] = teacher.summary()
        row["generation_audit"] = generation_audit(sample, teacher, oracle)
        return {"status": "accepted", "attempt": attempt, "row": row}
    except Exception as exc:
        return {"status": "error", "attempt": attempt, "error": repr(exc)}


def build_split_parallel(
    split: str,
    count: int,
    size: int,
    seed: int,
    algorithm: str,
    max_attempt_mult: int,
    oracle_limit: int,
    oracle_mode: str,
    workers: int,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    attempts_submitted = 0
    max_attempts = max(1, count * max_attempt_mult)
    stats: dict[str, Any] = {
        "attempts_submitted": 0,
        "accepted": 0,
        "teacher_fail": 0,
        "oracle_fail": 0,
        "error": 0,
        "workers": workers,
    }
    pending = set()
    task_common = (split, size, algorithm, seed, oracle_mode, oracle_limit)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while (len(rows) < count and attempts_submitted < max_attempts) or pending:
            while len(rows) < count and attempts_submitted < max_attempts and len(pending) < workers * 3:
                task = (task_common[0], attempts_submitted, task_common[1], task_common[2], task_common[3], task_common[4], task_common[5])
                pending.add(executor.submit(build_attempt, task))
                attempts_submitted += 1
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                res = fut.result()
                status = res.get("status", "error")
                if status == "accepted":
                    rows.append(res["row"])
                    stats["accepted"] = len(rows)
                    if len(rows) % 25 == 0 or len(rows) == count:
                        partial = out_dir / f"{split}.partial.json"
                        save_json(partial, rows)
                        audit = res["row"].get("generation_audit", {})
                        print(
                            {
                                "split": split,
                                "kept": len(rows),
                                "target": count,
                                "attempts_submitted": attempts_submitted,
                                "teacher_fail": stats["teacher_fail"],
                                "oracle_fail": stats["oracle_fail"],
                                "last_oracle_score": audit.get("oracle_score"),
                                "last_teacher_score": audit.get("teacher_score"),
                                "teacher_to_oracle": audit.get("teacher_score_ratio_to_oracle"),
                            },
                            flush=True,
                        )
                elif status == "teacher_fail":
                    stats["teacher_fail"] += 1
                elif status == "oracle_fail":
                    stats["oracle_fail"] += 1
                else:
                    stats["error"] += 1
                    print({"split": split, "status": status, "attempt": res.get("attempt"), "error": res.get("error")}, flush=True)
                stats["attempts_submitted"] = attempts_submitted
                if len(rows) >= count:
                    for item in pending:
                        item.cancel()
                    pending.clear()
                    break
    if len(rows) < count:
        raise RuntimeError(f"failed to build {split}: kept={len(rows)} attempts={attempts_submitted} target={count} stats={stats}")
    rows.sort(key=lambda x: x["sample_id"])
    return rows[:count], stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel full-map oracle-ratio AGRL dataset generation.")
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--algorithm", default="mixed")
    parser.add_argument("--out-dir", default="artifacts/agrl_oracle_ratio_15")
    parser.add_argument("--max-attempt-mult", type=int, default=250)
    parser.add_argument("--oracle-limit", type=int, default=200000)
    parser.add_argument("--oracle-mode", choices=["required", "optional", "off"], default="required")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oracle_limit = 0 if args.oracle_mode == "off" else args.oracle_limit
    counts = {"train": args.train, "val": args.val, "test": args.test}
    offsets = {"train": 0, "val": 100000, "test": 200000}
    all_stats: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        rows, stats = build_split_parallel(
            split=split,
            count=counts[split],
            size=args.size,
            seed=args.seed + offsets[split],
            algorithm=args.algorithm,
            max_attempt_mult=args.max_attempt_mult,
            oracle_limit=oracle_limit,
            oracle_mode=args.oracle_mode,
            workers=args.workers,
            out_dir=out,
        )
        save_json(out / f"{split}.json", rows)
        all_stats[split] = stats
    manifest = {
        "generator": "parallel unbiased randomized maze with random S/E/BOSS/coin/trap placement",
        "teacher": "safe_ratio_planner under strict 3x3 observation plus memory",
        "label_source": "bounded full-map ratio oracle" if args.oracle_mode == "required" else "oracle when available, otherwise online teacher",
        "metric": "final_remaining_resource / total_steps",
        "size": args.size,
        "seed": args.seed,
        "algorithm": args.algorithm,
        "counts": counts,
        "stats": all_stats,
        "oracle_mode": args.oracle_mode,
        "oracle_limit": oracle_limit,
        "workers": args.workers,
    }
    save_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
