"""Execute a move sequence under the official rules and score it.

Score = remaining resource value / steps  (the assignment's AI-player metric).

Resource economy:
    resource (coins) = COIN_VALUE * coins_collected
                     - TRAP_DAMAGE * traps_triggered
                     - CoinConsumption * revives
A 'G' tile yields COIN_VALUE (=50) coins, a trap costs TRAP_DAMAGE (=30) coins,
each trap triggers once, and a BOSS revive costs ``CoinConsumption`` coins. With
optimal skill play the group is cleared within the round limit, so revives is
normally 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .boss import BossPlan, plan_boss_fight
from .spec import (
    BOSS,
    COIN,
    COIN_VALUE,
    EXIT,
    MOVES,
    TRAP,
    TRAP_DAMAGE,
    Coord,
    MazeSpec,
)


@dataclass
class RunResult:
    success: bool
    cleared_boss: bool
    resource: int
    steps: int
    score: float
    coins_collected: int
    traps_triggered: int
    boss_rounds: int
    revives: int
    moves: list[str]
    frames: list[dict[str, Any]] = field(default_factory=list)
    boss_plan: BossPlan | None = None
    reason: str = ""


def _render(spec: MazeSpec, pos: Coord, collected: set, triggered: set, boss_dead: bool) -> list[str]:
    rows = [list(row) for row in spec.grid]
    for (r, c) in collected:
        if rows[r][c] == COIN:
            rows[r][c] = "."
    for (r, c) in triggered:
        if rows[r][c] == TRAP:
            rows[r][c] = "."
    if boss_dead and spec.boss is not None:
        br, bc = spec.boss
        rows[br][bc] = "."
    rows[pos[0]][pos[1]] = "@"
    return ["".join(row) for row in rows]


def simulate(spec: MazeSpec, moves: list[str]) -> RunResult:
    pos = spec.start
    resource = 0
    steps = 0
    collected: set[Coord] = set()
    triggered: set[Coord] = set()
    boss_dead = spec.boss is None
    boss_plan: BossPlan | None = None
    boss_rounds = 0
    revives = 0
    reason = "ok"

    def frame(event: str) -> dict[str, Any]:
        return {
            "step": steps,
            "pos": list(pos),
            "resource": resource,
            "event": event,
            "boss_dead": boss_dead,
            "grid": _render(spec, pos, collected, triggered, boss_dead),
        }

    frames = [frame("start")]

    def fight_boss() -> bool:
        # Resource is denominated in coins: a 'G' tile is +COIN_VALUE coins, a
        # trap -TRAP_DAMAGE coins, and a BOSS revive costs CoinConsumption coins.
        nonlocal resource, boss_dead, boss_plan, boss_rounds, revives, reason
        boss_plan = plan_boss_fight(spec, available_coins=resource)
        boss_rounds = boss_plan.min_rounds
        revives = boss_plan.revives
        resource -= boss_plan.coins_spent
        if not boss_plan.cleared:
            reason = "boss_unbeatable_no_coins"
            return False
        boss_dead = True
        return True

    for mv in moves:
        if mv not in MOVES:
            continue
        dr, dc = MOVES[mv]
        nxt = (pos[0] + dr, pos[1] + dc)
        if not spec.is_walkable(nxt):
            reason = "hit_wall"
            break
        ch = spec.char_at(nxt)
        # The exit is sealed until the boss group is defeated.
        if ch == EXIT and not boss_dead:
            reason = "exit_locked_boss_alive"
            break
        if ch == BOSS and not boss_dead:
            if not fight_boss():
                pos = nxt
                steps += 1
                frames.append(frame("boss_fail"))
                break
            pos = nxt
            steps += 1
            frames.append(frame("boss_clear"))
            continue
        pos = nxt
        steps += 1
        event = "move"
        if ch == COIN and nxt not in collected:
            collected.add(nxt)
            resource += COIN_VALUE
            event = "coin"
        elif ch == TRAP and nxt not in triggered:
            triggered.add(nxt)
            resource -= TRAP_DAMAGE
            event = "trap"
        elif ch == EXIT:
            event = "exit"
        frames.append(frame(event))
        if ch == EXIT:
            break

    success = pos == spec.exit and boss_dead
    score = resource / steps if steps > 0 else 0.0
    return RunResult(
        success=success,
        cleared_boss=boss_dead,
        resource=resource,
        steps=steps,
        score=score,
        coins_collected=len(collected),
        traps_triggered=len(triggered),
        boss_rounds=boss_rounds,
        revives=revives,
        moves=list(moves),
        frames=frames,
        boss_plan=boss_plan,
        reason=reason,
    )
