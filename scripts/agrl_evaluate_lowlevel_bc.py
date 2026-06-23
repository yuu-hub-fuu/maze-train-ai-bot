from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import aggregate_results, load_samples, run_strategy, save_json
from maze_gpt_agent.agrl_lowlevel_bc import load_model, run_bc_strategy
from maze_gpt_agent.visualizer import render_run_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate low-level BC policy against AGRL baselines.")
    parser.add_argument("--test", default="artifacts/agrl_ratio_v4/test.json")
    parser.add_argument("--model", default="artifacts/agrl_ratio_v4/lowlevel_bc.pt")
    parser.add_argument("--max-size", type=int, default=15)
    parser.add_argument("--out", default="artifacts/agrl_ratio_v4/evaluation_lowlevel_bc.json")
    parser.add_argument("--html", default="artifacts/agrl_ratio_v4/demo_lowlevel_bc.html")
    parser.add_argument("--include-baselines", action="store_true")
    args = parser.parse_args()

    samples = load_samples(args.test)
    model, metrics = load_model(args.model)
    print(json.dumps({"loaded_lowlevel_bc": args.model, "metrics": metrics}, ensure_ascii=False, indent=2))
    results = []
    demo = None
    for sample in samples:
        if args.include_baselines:
            for strategy in ["shortest", "greedy3x3", "classic"]:
                results.append(run_strategy(sample, strategy))
        result = run_bc_strategy(sample, model, args.max_size)
        results.append(result)
        if demo is None:
            demo = result
    summary = {
        "aggregate": aggregate_results(results),
        "runs": [r.summary() for r in results],
        "model_metrics": metrics,
        "vision_rule": "low-level model sees only 3x3-updated memory map; unknown cells are ?",
    }
    save_json(args.out, summary)
    if demo:
        render_run_html(args.html, "AGRL-Maze Low-Level BC Demo", demo.frames, demo.summary())
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"summary: {args.out}")
    print(f"demo_html: {args.html}")


if __name__ == "__main__":
    main()
