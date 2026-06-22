from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agents import Agent
from .expert_solver import frame_record
from .maze_env import MazeSpec, MazeState


@dataclass
class RunResult:
    agent: str
    maze: str
    scenario: str
    success: bool
    boss_clear: bool
    gold: int
    steps: int
    score: float
    trap_count: int
    invalid_actions: int
    frames: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "maze": self.maze,
            "scenario": self.scenario,
            "success": self.success,
            "boss_clear": self.boss_clear,
            "gold": self.gold,
            "steps": self.steps,
            "score": self.score,
            "trap_count": self.trap_count,
            "invalid_actions": self.invalid_actions,
        }


def run_agent(spec: MazeSpec, agent: Agent, max_steps: int | None = None, target_score: str = "high") -> RunResult:
    max_steps = max_steps or spec.height * spec.width * 4
    state = MazeState(spec)
    frames = [frame_record(state, None, "start")]
    invalid = 0
    while not state.done and state.steps < max_steps:
        action = agent.act(state, target_score)
        before = state.pos
        rec = state.step(action)
        if rec.get("invalid"):
            invalid += 1
        if before == state.pos and action not in ("ATTACK_BOSS", "STOP") and rec.get("invalid"):
            invalid += 1
        frames.append(frame_record(state, action, rec["event"]))
        if action == "STOP":
            break
    if not state.done:
        state.failed = True
        state.done = True
        frames.append(frame_record(state, "STOP", "timeout"))
    return RunResult(
        agent=agent.name,
        maze=spec.name,
        scenario=spec.scenario,
        success=(state.pos == spec.exit and not state.failed and state.done),
        boss_clear=state.boss_defeated,
        gold=state.gold,
        steps=state.steps,
        score=state.score(),
        trap_count=len(state.triggered),
        invalid_actions=invalid,
        frames=frames,
    )


def aggregate(results: list[RunResult]) -> list[dict[str, Any]]:
    by_agent: dict[str, list[RunResult]] = {}
    for result in results:
        by_agent.setdefault(result.agent, []).append(result)
    rows: list[dict[str, Any]] = []
    for agent, vals in sorted(by_agent.items()):
        n = len(vals)
        rows.append(
            {
                "agent": agent,
                "episodes": n,
                "success_rate": sum(v.success for v in vals) / n,
                "boss_clear_rate": sum(v.boss_clear for v in vals) / n,
                "avg_gold_left": sum(v.gold for v in vals) / n,
                "avg_steps": sum(v.steps for v in vals) / n,
                "avg_score_gold_per_step": sum(v.score for v in vals) / n,
                "avg_trap_count": sum(v.trap_count for v in vals) / n,
                "invalid_actions": sum(v.invalid_actions for v in vals),
            }
        )
    return rows
