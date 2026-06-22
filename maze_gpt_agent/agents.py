from __future__ import annotations

from collections import Counter, deque
import json
from pathlib import Path
from typing import Any, Protocol

from .constants import ACTIONS, COIN, EXIT, TRAP
from .features import action_features, normalize_action
from .maze_env import MazeState
from .pathfinding import astar_path, bfs_path


class Agent(Protocol):
    name: str

    def act(self, state: MazeState, target_score: str = "high") -> str:
        ...


class Greedy3x3Agent:
    name = "Greedy-3x3"

    def act(self, state: MazeState, target_score: str = "high") -> str:
        assert state.pos is not None
        best: tuple[float, str] | None = None
        for action in state.legal_actions():
            if action not in ("UP", "DOWN", "LEFT", "RIGHT"):
                continue
            nxt = next_pos(state, action)
            ch = state.visible_char(nxt)
            value = 0
            if ch == COIN:
                value = state.spec.coin_value
            elif ch == TRAP:
                value = -state.spec.trap_damage
            score = value / 1.0
            if best is None or score > best[0]:
                best = (score, action)
        if best and best[0] > 0:
            return best[1]
        return _path_following_action(state, prefer_resource=True)


class BFSToEndAgent:
    name = "BFS-to-End"

    def act(self, state: MazeState, target_score: str = "high") -> str:
        goal = state.spec.exit
        path = bfs_path(state.spec, state.pos or state.spec.start, goal, boss_defeated=True)
        return path[0] if path else "STOP"


class AStarResourceAgent:
    name = "AStar-Resource"

    def act(self, state: MazeState, target_score: str = "high") -> str:
        return _path_following_action(state, prefer_resource=True, use_astar=True)


class PerceptronMazeAgent:
    name = "Ours-PostTrained-MazeAgent"

    def __init__(self, weights: dict[str, float] | None = None, labels: list[str] | None = None):
        self.weights = weights or {}
        self.labels = labels or list(ACTIONS)
        self.invalid_raw = 0
        self.total_raw = 0
        self.recent: deque[tuple[int, int]] = deque(maxlen=8)

    def score(self, state: MazeState, action: str, target_score: str = "high") -> float:
        return sum(self.weights.get(feat, 0.0) for feat in action_features(state, action, target_score))

    def act(self, state: MazeState, target_score: str = "high") -> str:
        legal = [a for a in self.labels if a in state.legal_actions() and a != "STOP"]
        if not legal:
            return "STOP"
        scored = sorted(((self.score(state, a, target_score), a) for a in legal), reverse=True)
        best_score, raw = scored[0]
        self.total_raw += 1
        margin = best_score - (scored[1][0] if len(scored) > 1 else -999.0)
        if best_score <= 0.0 or margin < 1.0:
            self.invalid_raw += 1
            raw = self.safe_fallback(state)
        if self._would_loop(state, raw):
            raw = self.safe_fallback(state)
        return raw

    def _would_loop(self, state: MazeState, action: str) -> bool:
        if action not in ("UP", "DOWN", "LEFT", "RIGHT"):
            return False
        nxt = next_pos(state, action)
        if self.recent.count(nxt) >= 3:
            return True
        self.recent.append(nxt)
        return False

    def safe_fallback(self, state: MazeState) -> str:
        # Safety executor only constrains illegal/looping actions; it does not
        # replace normal policy selection.
        return _path_following_action(state, prefer_resource=True, use_astar=True)

    @property
    def invalid_action_rate(self) -> float:
        return self.invalid_raw / max(1, self.total_raw)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"labels": self.labels, "weights": self.weights}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "PerceptronMazeAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(weights={k: float(v) for k, v in data["weights"].items()}, labels=data["labels"])


def train_perceptron(records: list[dict[str, Any]], epochs: int = 5) -> PerceptronMazeAgent:
    agent = PerceptronMazeAgent()
    weights = Counter(agent.weights)
    labels = list(ACTIONS)
    for _ in range(epochs):
        mistakes = 0
        for rec in records:
            state = rec["_state"]
            gold = rec["action"]
            target = rec.get("target_return", "high")
            pred = max(labels, key=lambda a: sum(weights[f] for f in action_features(state, a, target)))
            if pred != gold:
                mistakes += 1
                for feat in action_features(state, gold, target):
                    weights[feat] += 1.0
                for feat in action_features(state, pred, target):
                    weights[feat] -= 1.0
        if mistakes == 0:
            break
    agent.weights = dict(weights)
    return agent


def _path_following_action(state: MazeState, prefer_resource: bool = False, use_astar: bool = False) -> str:
    assert state.pos is not None
    goals = []
    if prefer_resource:
        need = state.spec.boss_cost - state.gold
        uncollected = [p for p in state.spec.coins if p not in state.collected]
        if need > 0:
            goals.extend(uncollected)
        elif uncollected:
            # Keep collecting when it is close enough to improve the final ratio.
            goals.extend(uncollected[:])
    if state.spec.boss and not state.boss_defeated:
        goals.append(state.spec.boss)
    goals.append(state.spec.exit)

    cost_fn = lambda pos: 8.0 if state.visible_char(pos) == TRAP else 1.0
    best_path = None
    for goal in goals:
        path = astar_path(state.spec, state.pos, goal, state.boss_defeated, cost_fn) if use_astar else bfs_path(
            state.spec, state.pos, goal, state.boss_defeated
        )
        if path and (best_path is None or len(path) < len(best_path)):
            best_path = path
    return best_path[0] if best_path else first_legal_move(state)


def first_legal_move(state: MazeState) -> str:
    for action in ("UP", "DOWN", "LEFT", "RIGHT", "ATTACK_BOSS"):
        if action in state.legal_actions():
            return action
    return "STOP"


def next_pos(state: MazeState, action: str) -> tuple[int, int]:
    from .constants import MOVE_DELTAS

    assert state.pos is not None
    dr, dc = MOVE_DELTAS[action]
    return (state.pos[0] + dr, state.pos[1] + dc)

