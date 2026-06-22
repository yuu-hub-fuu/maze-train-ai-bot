from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .expert_solver import solve_expert
from .maze_env import MazeSpec, MazeState


def build_records(spec: MazeSpec, target_return: str = "high") -> list[dict[str, Any]]:
    traj = solve_expert(spec)
    state = MazeState(spec)
    records: list[dict[str, Any]] = []
    for idx, action in enumerate(traj.actions):
        rec = {
            "id": f"{spec.name}-{idx}",
            "maze": spec.to_dict(),
            "state": state_to_dict(state),
            "prompt": state.prompt(target_return),
            "target_return": target_return,
            "action": action,
            "expert_score": traj.score,
            "expert_final_gold": traj.final_gold,
            "expert_steps": traj.steps,
            "scenario": spec.scenario,
        }
        records.append(rec)
        state.step(action)
    return records


def state_to_dict(state: MazeState) -> dict[str, Any]:
    return {
        "pos": state.pos,
        "gold": state.gold,
        "steps": state.steps,
        "collected": sorted(state.collected),
        "triggered": sorted(state.triggered),
        "boss_defeated": state.boss_defeated,
        "done": state.done,
        "failed": state.failed,
    }


def state_from_record(rec: dict[str, Any]) -> MazeState:
    spec = MazeSpec.from_dict(rec["maze"])
    s = rec["state"]
    return MazeState(
        spec=spec,
        pos=tuple(s["pos"]),
        gold=int(s["gold"]),
        steps=int(s["steps"]),
        collected={tuple(x) for x in s.get("collected", [])},
        triggered={tuple(x) for x in s.get("triggered", [])},
        boss_defeated=bool(s.get("boss_defeated", False)),
        done=bool(s.get("done", False)),
        failed=bool(s.get("failed", False)),
    )


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(strip_runtime(rec), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path, with_state: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if with_state:
                rec["_state"] = state_from_record(rec)
            records.append(rec)
    return records


def strip_runtime(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rec.items() if not k.startswith("_")}
