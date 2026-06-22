from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    ACTIONS,
    BOSS,
    COIN,
    COIN_VALUE,
    DEFAULT_BOSS_COST,
    EMPTY,
    EXIT,
    MOVE_DELTAS,
    REWARD_BOSS,
    REWARD_BOSS_FAIL,
    REWARD_COIN,
    REWARD_EXIT,
    REWARD_INVALID,
    REWARD_STEP,
    REWARD_TRAP,
    START,
    TRAP,
    TRAP_DAMAGE,
    WALL,
)

Coord = tuple[int, int]


@dataclass(frozen=True)
class MazeSpec:
    grid: tuple[str, ...]
    boss_cost: int = DEFAULT_BOSS_COST
    coin_value: int = COIN_VALUE
    trap_damage: int = TRAP_DAMAGE
    name: str = "maze"
    scenario: str = "mixed"

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[str],
        boss_cost: int = DEFAULT_BOSS_COST,
        name: str = "maze",
        scenario: str = "mixed",
    ) -> "MazeSpec":
        grid = tuple(str(row) for row in rows)
        if not grid or len({len(row) for row in grid}) != 1:
            raise ValueError("maze grid must be rectangular and non-empty")
        return cls(grid=grid, boss_cost=boss_cost, name=name, scenario=scenario)

    @classmethod
    def from_json(cls, path: str | Path) -> "MazeSpec":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MazeSpec":
        return cls.from_rows(
            data["grid"],
            boss_cost=int(data.get("boss_cost", DEFAULT_BOSS_COST)),
            name=data.get("name", "maze"),
            scenario=data.get("scenario", "mixed"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scenario": self.scenario,
            "boss_cost": self.boss_cost,
            "coin_value": self.coin_value,
            "trap_damage": self.trap_damage,
            "grid": list(self.grid),
        }

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0])

    def char_at(self, pos: Coord) -> str:
        r, c = pos
        if not (0 <= r < self.height and 0 <= c < self.width):
            return WALL
        return self.grid[r][c]

    def positions(self, tile: str) -> list[Coord]:
        out: list[Coord] = []
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == tile:
                    out.append((r, c))
        return out

    @property
    def start(self) -> Coord:
        hits = self.positions(START)
        if not hits:
            raise ValueError("maze has no S start")
        return hits[0]

    @property
    def exit(self) -> Coord:
        hits = self.positions(EXIT)
        if not hits:
            raise ValueError("maze has no E exit")
        return hits[0]

    @property
    def boss(self) -> Coord | None:
        hits = self.positions(BOSS)
        return hits[0] if hits else None

    @property
    def coins(self) -> tuple[Coord, ...]:
        return tuple(self.positions(COIN))

    @property
    def traps(self) -> tuple[Coord, ...]:
        return tuple(self.positions(TRAP))

    def is_walkable(self, pos: Coord, boss_defeated: bool = False) -> bool:
        ch = self.char_at(pos)
        return ch != WALL


@dataclass
class MazeState:
    spec: MazeSpec
    pos: Coord | None = None
    gold: int = 0
    steps: int = 0
    collected: set[Coord] = field(default_factory=set)
    triggered: set[Coord] = field(default_factory=set)
    boss_defeated: bool = False
    done: bool = False
    failed: bool = False
    log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.pos is None:
            self.pos = self.spec.start

    def clone(self) -> "MazeState":
        return MazeState(
            spec=self.spec,
            pos=self.pos,
            gold=self.gold,
            steps=self.steps,
            collected=set(self.collected),
            triggered=set(self.triggered),
            boss_defeated=self.boss_defeated,
            done=self.done,
            failed=self.failed,
            log=list(self.log),
        )

    def legal_actions(self) -> list[str]:
        if self.done:
            return ["STOP"]
        actions: list[str] = []
        assert self.pos is not None
        for action, delta in MOVE_DELTAS.items():
            nr, nc = self.pos[0] + delta[0], self.pos[1] + delta[1]
            if self.spec.is_walkable((nr, nc), self.boss_defeated):
                actions.append(action)
        boss = self.spec.boss
        if boss and not self.boss_defeated:
            if manhattan(self.pos, boss) <= 1 or self.pos == boss:
                actions.append("ATTACK_BOSS")
        actions.append("STOP")
        return actions

    def step(self, action: str) -> dict[str, Any]:
        if action not in ACTIONS:
            action = "STOP"
        if self.done:
            return {"action": action, "reward": 0, "event": "already_done"}

        reward = REWARD_STEP
        event = "move"
        invalid = False
        old_pos = self.pos
        assert old_pos is not None

        if action == "STOP":
            self.done = True
            self.failed = self.pos != self.spec.exit
            event = "stop"
        elif action == "ATTACK_BOSS":
            reward += self._fight_boss()
            event = self.log[-1]["event"] if self.log and self.log[-1].get("synthetic") else "attack"
        else:
            delta = MOVE_DELTAS[action]
            new_pos = (old_pos[0] + delta[0], old_pos[1] + delta[1])
            if not self.spec.is_walkable(new_pos, self.boss_defeated):
                reward += REWARD_INVALID
                invalid = True
                event = "invalid_wall"
            else:
                self.pos = new_pos
                self.steps += 1
                cell = self.spec.char_at(new_pos)
                if cell == COIN and new_pos not in self.collected:
                    self.collected.add(new_pos)
                    self.gold += self.spec.coin_value
                    reward += REWARD_COIN
                    event = "coin"
                elif cell == TRAP and new_pos not in self.triggered:
                    self.triggered.add(new_pos)
                    self.gold -= self.spec.trap_damage
                    reward += REWARD_TRAP
                    event = "trap"
                elif cell == BOSS and not self.boss_defeated:
                    reward += self._fight_boss()
                    event = "boss_clear" if self.boss_defeated else "boss_fail"
                elif cell == EXIT:
                    if self.spec.boss is None or self.boss_defeated:
                        reward += REWARD_EXIT
                        self.done = True
                        event = "exit"
                    else:
                        reward += REWARD_BOSS_FAIL
                        self.done = True
                        self.failed = True
                        event = "exit_without_boss"

        rec = {
            "step": self.steps,
            "action": action,
            "from": old_pos,
            "to": self.pos,
            "gold": self.gold,
            "reward": reward,
            "event": event,
            "invalid": invalid,
            "done": self.done,
            "failed": self.failed,
            "score": self.score(),
        }
        self.log.append(rec)
        return rec

    def _fight_boss(self) -> int:
        if self.boss_defeated:
            return 0
        boss = self.spec.boss
        if boss is None:
            return 0
        if self.gold >= self.spec.boss_cost:
            self.gold -= self.spec.boss_cost
            self.boss_defeated = True
            self.log.append({"event": "boss_clear", "synthetic": True, "gold": self.gold})
            return REWARD_BOSS
        self.done = True
        self.failed = True
        self.log.append({"event": "boss_fail", "synthetic": True, "gold": self.gold})
        return REWARD_BOSS_FAIL

    def score(self) -> float:
        if self.steps <= 0:
            return 0.0
        return self.gold / self.steps

    def vision_3x3(self) -> list[str]:
        assert self.pos is not None
        rows: list[str] = []
        for r in range(self.pos[0] - 1, self.pos[0] + 2):
            chars: list[str] = []
            for c in range(self.pos[1] - 1, self.pos[1] + 2):
                if (r, c) == self.pos:
                    chars.append("@")
                else:
                    chars.append(self.visible_char((r, c)))
            rows.append("".join(chars))
        return rows

    def visible_char(self, pos: Coord) -> str:
        ch = self.spec.char_at(pos)
        if ch == COIN and pos in self.collected:
            return EMPTY
        if ch == TRAP and pos in self.triggered:
            return EMPTY
        if ch == BOSS and self.boss_defeated:
            return EMPTY
        return ch

    def render(self) -> list[str]:
        rows = [list(row) for row in self.spec.grid]
        for r, c in self.collected:
            if rows[r][c] == COIN:
                rows[r][c] = EMPTY
        for r, c in self.triggered:
            if rows[r][c] == TRAP:
                rows[r][c] = EMPTY
        if self.boss_defeated and self.spec.boss:
            r, c = self.spec.boss
            rows[r][c] = EMPTY
        assert self.pos is not None
        pr, pc = self.pos
        rows[pr][pc] = "@"
        return ["".join(row) for row in rows]

    def to_observation(self, target_score: str = "high") -> dict[str, Any]:
        return {
            "maze": list(self.render()),
            "player_pos": self.pos,
            "vision_3x3": self.vision_3x3(),
            "coin_now": self.gold,
            "boss_cost": self.spec.boss_cost,
            "trap_damage": self.spec.trap_damage,
            "collected_gold": sorted(self.collected),
            "triggered_traps": sorted(self.triggered),
            "boss_defeated": self.boss_defeated,
            "step_now": self.steps,
            "target_return": target_score,
        }

    def prompt(self, target_score: str = "high") -> str:
        obs = self.to_observation(target_score)
        lines = [
            "You are MazeGPT-Agent. Choose exactly one action:",
            ", ".join(ACTIONS),
            f"target_score={obs['target_return']} coin_now={obs['coin_now']} step_now={obs['step_now']}",
            f"boss_cost={obs['boss_cost']} boss_defeated={obs['boss_defeated']} trap_damage={obs['trap_damage']}",
            f"player_pos={obs['player_pos']}",
            "vision_3x3:",
            *obs["vision_3x3"],
            "maze:",
            *obs["maze"],
            "Answer with the next action only.",
        ]
        return "\n".join(lines)


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def save_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
