from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import aggregate_results, load_samples, run_strategy, save_json
from maze_gpt_agent.visualizer import render_run_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AGRL-Maze strategies.")
    parser.add_argument("--test", default="artifacts/agrl/test.json")
    parser.add_argument("--q-table", default="artifacts/agrl/q_table.json")
    parser.add_argument("--out", default="artifacts/agrl/evaluation_summary.json")
    parser.add_argument("--html", default="artifacts/agrl/demo_run.html")
    args = parser.parse_args()

    samples = load_samples(args.test)
    q_table = json.loads(Path(args.q_table).read_text(encoding="utf-8")) if Path(args.q_table).exists() else {}
    strategies = ["shortest", "greedy3x3", "classic", "rl"]
    results = []
    demo = None
    for sample in samples:
        for strategy in strategies:
            result = run_strategy(sample, strategy, q_table=q_table if strategy == "rl" else None)
            results.append(result)
            if demo is None and strategy == "rl":
                demo = result
    summary = {
        "aggregate": aggregate_results(results),
        "runs": [r.summary() for r in results],
        "vision_rule": "all evaluated strategies observe only current 3x3 plus remembered cells; unknown cells are hidden",
    }
    save_json(args.out, summary)
    if demo:
        render_run_html(args.html, "AGRL-Maze 3x3 Memory Demo", demo.frames, demo.summary())
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"summary: {args.out}")
    print(f"demo_html: {args.html}")


if __name__ == "__main__":
    main()
