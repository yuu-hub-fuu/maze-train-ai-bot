"""Parse and represent the official course maze input format."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

Coord = tuple[int, int]

# Tile characters (everything except WALL is walkable).
WALL = "#"
EMPTY = " "
START = "S"
EXIT = "E"
BOSS = "B"
COIN = "G"
TRAP = "T"

# Resource values fixed by the assignment.
COIN_VALUE = 50
TRAP_DAMAGE = 30

MOVES: dict[str, Coord] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


@dataclass
class Skill:
    """A player skill: ``damage`` per use, on cooldown for ``cooldown`` rounds."""

    damage: int
    cooldown: int


@dataclass
class MazeSpec:
    """A fully-observable course maze plus its BOSS battle parameters."""

    grid: list[str]
    boss_hps: list[int]
    skills: list[Skill]
    min_rounds: int = 20
    coin_consumption: int = 5
    name: str = "maze"

    # Cached lookups (filled in __post_init__).
    start: Coord = field(init=False)
    exit: Coord = field(init=False)
    boss: Coord | None = field(init=False)
    coins: list[Coord] = field(init=False)
    traps: list[Coord] = field(init=False)

    def __post_init__(self) -> None:
        if not self.grid or len({len(r) for r in self.grid}) != 1:
            raise ValueError("maze grid must be rectangular and non-empty")
        self.coins = []
        self.traps = []
        self.boss = None
        start = exit_ = None
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                pos = (r, c)
                if ch == START:
                    start = pos
                elif ch == EXIT:
                    exit_ = pos
                elif ch == BOSS:
                    self.boss = pos
                elif ch == COIN:
                    self.coins.append(pos)
                elif ch == TRAP:
                    self.traps.append(pos)
        if start is None:
            raise ValueError("maze has no start 'S'")
        if exit_ is None:
            raise ValueError("maze has no exit 'E'")
        self.start = start
        self.exit = exit_

    # ------------------------------------------------------------------ grid
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

    def is_walkable(self, pos: Coord) -> bool:
        return self.char_at(pos) != WALL

    # ------------------------------------------------------------------- I/O
    @classmethod
    def from_dict(cls, data: dict[str, Any], name: str = "maze") -> "MazeSpec":
        rows = data["maze"]
        # Rows may be a list of single-char lists or already-joined strings.
        grid = ["".join(row) if not isinstance(row, str) else row for row in rows]
        boss_hps = [int(x) for x in data.get("B", [])]
        skills = [Skill(int(a), int(b)) for a, b in data.get("PlayerSkills", [])]
        return cls(
            grid=grid,
            boss_hps=boss_hps,
            skills=skills,
            min_rounds=int(data.get("minRouds", data.get("minRounds", 20))),
            coin_consumption=int(data.get("CoinConsumption", 5)),
            name=name,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "MazeSpec":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data, name=p.stem)

    def to_dict(self) -> dict[str, Any]:
        return {
            "maze": [list(row) for row in self.grid],
            "B": list(self.boss_hps),
            "PlayerSkills": [[s.damage, s.cooldown] for s in self.skills],
            "minRouds": self.min_rounds,
            "CoinConsumption": self.coin_consumption,
        }
