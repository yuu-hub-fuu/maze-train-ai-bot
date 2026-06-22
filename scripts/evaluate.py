from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agents import AStarResourceAgent, BFSToEndAgent, Greedy3x3Agent, PerceptronMazeAgent
from maze_gpt_agent.evaluator import aggregate, run_agent
from maze_gpt_agent.maze_env import MazeSpec, save_json
from maze_gpt_agent.maze_generator import SCENARIOS, generate_course_maze
from maze_gpt_agent.visualizer import render_run_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baselines and the post-trained MazeGPT agent.")
    parser.add_argument("--model", default="artifacts/models/maze_policy.json")
    parser.add_argument("--episodes", type=int, default=36)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--maze-file", default="")
    parser.add_argument("--out", default="artifacts/eval/evaluation_summary.json")
    parser.add_argument("--html", default="artifacts/eval/demo_run.html")
    args = parser.parse_args()

    if args.maze_file:
        raw = json.loads(Path(args.maze_file).read_text(encoding="utf-8"))
        mazes = [MazeSpec.from_dict(item) for item in raw]
    else:
        mazes = [
            generate_course_maze(size=args.size, seed=args.seed + i, scenario=SCENARIOS[i % len(SCENARIOS)])
            for i in range(args.episodes)
        ]

    agents = [Greedy3x3Agent(), BFSToEndAgent(), AStarResourceAgent()]
    if Path(args.model).exists():
        agents.append(PerceptronMazeAgent.load(args.model))
    else:
        print(f"warning: model not found: {args.model}; skipping Ours")

    results = []
    demo = None
    for spec in mazes:
        for agent in agents:
            result = run_agent(spec, agent)
            results.append(result)
            if demo is None and agent.name.startswith("Ours"):
                demo = result

    summary = {
        "aggregate": aggregate(results),
        "runs": [r.summary() for r in results],
        "mazes": [m.to_dict() for m in mazes],
    }
    save_json(args.out, summary)
    if demo is not None:
        render_run_html(args.html, "MazeGPT-Agent Demo Run", demo.frames, demo.summary())
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"summary: {args.out}")
    if demo is not None:
        print(f"demo html: {args.html}")


if __name__ == "__main__":
    main()
