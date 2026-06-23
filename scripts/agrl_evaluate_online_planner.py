from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import aggregate_results, load_samples, run_strategy, save_json
from maze_gpt_agent.agrl_online_planner import run_online_planner
from maze_gpt_agent.visualizer import render_run_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate online ratio-aware planner.")
    parser.add_argument("--test", default="artifacts/agrl_ratio_v4_probe/test.json")
    parser.add_argument("--out", default="artifacts/agrl_ratio_v4_probe/evaluation_online_planner.json")
    parser.add_argument("--html", default="artifacts/agrl_ratio_v4_probe/demo_online_planner.html")
    parser.add_argument("--include-baselines", action="store_true")
    args = parser.parse_args()

    results = []
    demo = None
    for sample in load_samples(args.test):
        if args.include_baselines:
            for strategy in ["shortest", "greedy3x3", "classic"]:
                results.append(run_strategy(sample, strategy))
        result = run_online_planner(sample)
        results.append(result)
        if demo is None:
            demo = result
    summary = {
        "aggregate": aggregate_results(results),
        "runs": [r.summary() for r in results],
        "vision_rule": "online planner observes only 3x3-updated memory; full map is used only by environment",
    }
    save_json(args.out, summary)
    if demo:
        render_run_html(args.html, "AGRL-Maze Online Ratio Planner Demo", demo.frames, demo.summary())
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"summary: {args.out}")
    print(f"demo_html: {args.html}")


if __name__ == "__main__":
    main()
