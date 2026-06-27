"""CLI: solve a course maze with the classic-algorithm AI player.

Usage:
    python -m maze_ai.run path/to/maze.json [--html out.html] [--greedy]

Outputs the BOSS battle plan, the optimal exploration route summary
(remaining resource value, steps, value/step ratio), and optionally an
interactive HTML visualization of the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .greedy3x3 import greedy_3x3_run
from .planner import plan_full_exploration
from .simulate import simulate
from .spec import MazeSpec
from .visualize import render_html


def solve_and_report(path: str, html_path: str | None = None, greedy: bool = False) -> dict:
    spec = MazeSpec.from_json(path)
    plan = plan_full_exploration(spec)
    run = simulate(spec, plan.moves)

    bp = plan.boss_plan
    summary = {
        "maze": spec.name,
        "size": f"{spec.height}x{spec.width}",
        "boss": {
            "hp_group": spec.boss_hps,
            "min_rounds": bp.min_rounds,
            "round_limit": spec.min_rounds,
            "within_limit": bp.feasible_within_limit,
            "skill_sequence": bp.skill_names,
            "revives": bp.revives,
            "coins_spent": bp.coins_spent,
        },
        "exploration": {
            "success": run.success,
            "remaining_resource_value": run.resource,
            "steps": run.steps,
            "score_value_per_step": round(run.score, 4),
            "coins_collected": f"{run.coins_collected}/{len(spec.coins)}",
            "traps_triggered": f"{run.traps_triggered}/{len(spec.traps)}",
            "method": plan.method,
            "candidates_evaluated": plan.notes.get("candidates_evaluated"),
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if greedy:
        gr = greedy_3x3_run(spec)
        summary["greedy_3x3"] = {
            "picked_value": gr.picked_value,
            "steps": gr.steps,
            "avg_per_step": round(gr.avg_per_step, 4),
            "coins_collected": gr.coins_collected,
            "traps_triggered": gr.traps_triggered,
            "reached_exit": gr.reached_exit,
        }
        print("greedy_3x3:", json.dumps(summary["greedy_3x3"], ensure_ascii=False))

    if html_path:
        render_html(f"{spec.name} — AI player (DP optimal route)", run.frames, summary, html_path)
        print(f"HTML written to {html_path}")

    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Classic-algorithm maze AI player")
    ap.add_argument("maze", help="path to maze JSON (official format)")
    ap.add_argument("--html", default=None, help="write interactive HTML visualization here")
    ap.add_argument("--greedy", action="store_true", help="also run the 3x3 greedy pickup (task 1)")
    args = ap.parse_args(argv)
    solve_and_report(args.maze, args.html, args.greedy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
