from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import aggregate_results, load_samples, save_json
from maze_gpt_agent.agrl_cnn_bc import load_model, run_cnn_bc_strategy
from maze_gpt_agent.visualizer import render_run_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CNN low-level oracle BC policy.")
    parser.add_argument("--test", default="artifacts/agrl_oracle_ratio_15/test.json")
    parser.add_argument("--model", default="artifacts/agrl_oracle_ratio_15/cnn_lowlevel_oracle_bc.pt")
    parser.add_argument("--max-size", type=int, default=15)
    parser.add_argument("--out", default="artifacts/agrl_oracle_ratio_15/evaluation_cnn_lowlevel_oracle_bc.json")
    parser.add_argument("--html", default="artifacts/agrl_oracle_ratio_15/demo_cnn_lowlevel_oracle_bc.html")
    args = parser.parse_args()
    samples = load_samples(args.test)
    model, metrics = load_model(args.model)
    results = []
    demo = None
    for idx, sample in enumerate(samples, 1):
        result = run_cnn_bc_strategy(sample, model, args.max_size)
        results.append(result)
        if demo is None:
            demo = result
        if idx % 25 == 0 or idx == len(samples):
            print({"evaluated": idx, "total": len(samples)}, flush=True)
    summary = {
        "aggregate": aggregate_results(results),
        "runs": [r.summary() for r in results],
        "model_metrics": metrics,
        "vision_rule": "CNN low-level policy sees only 3x3-updated memory map; unknown cells are encoded as ?."
    }
    save_json(args.out, summary)
    if demo:
        render_run_html(args.html, "AGRL-Maze CNN Low-Level Oracle BC Demo", demo.frames, demo.summary())
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"summary: {args.out}")
    print(f"demo_html: {args.html}")


if __name__ == "__main__":
    main()
