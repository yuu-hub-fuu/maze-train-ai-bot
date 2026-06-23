from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import (
    HIGH_LEVEL_ACTIONS,
    PlayerState,
    apply_move,
    choose_target,
    load_samples,
    observe_3x3,
    rcspp_path,
    solve_boss_battle,
    vision_3x3,
)


def memory_rows(sample, state: PlayerState) -> list[str]:
    rows = []
    for r in range(sample.rows):
        chars = []
        for c in range(sample.cols):
            pos = (r, c)
            if pos == state.position:
                chars.append("@")
            else:
                chars.append(state.known.get(pos, "?"))
        rows.append("".join(chars))
    return rows


def prompt(sample, state: PlayerState) -> str:
    return "\n".join(
        [
            "You are AGRL-MazeGPT. The player can observe only the current 3x3 view and remembered cells.",
            "Unknown cells are '?'. Choose exactly one high-level action:",
            ", ".join(HIGH_LEVEL_ACTIONS),
            f"difficulty={sample.difficulty} resource={state.resource} steps={state.steps}",
            f"boss_defeated={state.boss_defeated} boss_revive_cost={sample.boss_config.revive_cost}",
            f"position={state.position} known_cells={len(state.known)}",
            "vision_3x3:",
            *vision_3x3(sample, state),
            "memory_map:",
            *memory_rows(sample, state),
            "Answer with the next high-level action only.",
        ]
    )


def export_records(samples, max_steps: int | None = None):
    rows = []
    for sample in samples:
        state = PlayerState(position=sample.start, path_history=[sample.start])
        observe_3x3(sample, state)
        limit = max_steps or sample.rows * sample.cols * 4
        for step_idx in range(limit):
            target, reason, high_action = choose_target(sample, state, "classic")
            if target is None:
                break
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt(sample, state)},
                        {"role": "assistant", "content": high_action},
                    ],
                    "sample_id": sample.sample_id,
                    "difficulty": sample.difficulty,
                    "reason": reason,
                    "target_type": target.target_type,
                    "target": list(target.position),
                }
            )
            path = rcspp_path(sample, state, target.position, require_boss_resource=(target.target_type == "boss"))
            if not path.feasible or not path.actions:
                break
            for pos in path.path[1:]:
                apply_move(sample, state, pos)
            if state.position == sample.boss and not state.boss_defeated:
                boss = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
                if boss.success and state.resource >= sample.boss_config.revive_cost:
                    state.boss_defeated = True
                else:
                    break
            if state.position == sample.end:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AGRL 3x3-memory expert decisions as HuggingFace chat SFT JSONL.")
    parser.add_argument("--input", default="artifacts/agrl_large/train.json")
    parser.add_argument("--out", default="artifacts/agrl_large/hf_sft_messages.jsonl")
    args = parser.parse_args()

    samples = load_samples(args.input)
    rows = export_records(samples)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"samples": len(samples), "records": len(rows), "out": args.out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
