"""Branch-and-bound BOSS battle solver.

The player faces a *group* of bosses (``B = [hp0, hp1, ...]``) defeated in a
fixed order. Each round the player uses exactly one skill that is not on
cooldown; using skill ``k`` deals ``damage[k]`` to the current boss and puts that
skill on cooldown for ``cooldown[k]`` rounds. We minimise the number of rounds to
clear the whole group, and recover the optimal skill sequence.

Search space: a state is ``(boss_index, hp_remaining, cooldowns)``. Each round
costs 1, so a breadth-first search over states expands them in non-decreasing
round order and the first time we reach "all bosses cleared" is optimal. The
state space is tiny (bounded by ``#bosses x max_hp x prod(cooldown_i+1)``), so a
plain BFS with a visited set acts as the branch-and-bound: dominated states
(already reached in fewer-or-equal rounds) are pruned.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .spec import MazeSpec, Skill


@dataclass
class BossPlan:
    min_rounds: int                 # optimal rounds to clear the group (no round limit)
    skill_sequence: list[int]       # indices into spec.skills, one per round
    feasible_within_limit: bool     # min_rounds <= round limit (one attempt is enough)
    revives: int                    # times the player had to revive (failed attempts)
    coins_spent: int                # CoinConsumption * revives
    cleared: bool                   # group eventually cleared given the coin budget
    skill_names: list[str] = field(default_factory=list)


def solve_boss_group(
    boss_hps: list[int],
    skills: list[Skill],
    start_index: int = 0,
    start_hp: int | None = None,
    carry_overkill: bool = False,
) -> tuple[int, list[int]]:
    """Return (min_rounds, skill_sequence) to clear bosses[start_index:].

    ``start_hp`` overrides the current boss's HP (for resuming after a revive).
    Cooldowns always start fresh (all available).
    """
    n = len(boss_hps)
    if n == 0:
        return 0, []
    if not skills:
        raise ValueError("at least one skill is required")
    cds = tuple(s.cooldown for s in skills)
    dmgs = tuple(s.damage for s in skills)

    init_hp = boss_hps[start_index] if start_hp is None else start_hp
    init_state = (start_index, init_hp, (0,) * len(skills))
    parent: dict[tuple, tuple] = {init_state: (None, -1)}
    q: deque[tuple] = deque([init_state])

    goal: tuple | None = None
    while q:
        state = q.popleft()
        bi, hp, cd = state
        if bi >= n:
            goal = state
            break
        for k in range(len(skills)):
            if cd[k] != 0:
                continue
            new_hp = hp - dmgs[k]
            # advance cooldowns by one round, then put skill k on cooldown
            new_cd = tuple(0 if j == k else (c - 1 if c > 0 else 0) for j, c in enumerate(cd))
            new_cd = tuple(cds[k] if j == k else new_cd[j] for j in range(len(skills)))
            if new_hp <= 0:
                nbi = bi + 1
                if nbi >= n:
                    nstate = (nbi, 0, new_cd)
                else:
                    carry = (-new_hp) if carry_overkill else 0
                    nstate = (nbi, max(1, boss_hps[nbi] - carry), new_cd)
            else:
                nstate = (bi, new_hp, new_cd)
            if nstate not in parent:
                parent[nstate] = (state, k)
                q.append(nstate)
    if goal is None:
        raise ValueError("boss group is unbeatable with the given skills")

    # reconstruct skill sequence
    seq: list[int] = []
    cur = goal
    while parent[cur][0] is not None:
        prev, k = parent[cur]
        seq.append(k)
        cur = prev
    seq.reverse()
    return len(seq), seq


def plan_boss_fight(spec: MazeSpec, available_coins: int | None = None) -> BossPlan:
    """Full BOSS-fight plan including the round limit + revive economics.

    If the optimal clear needs no more than ``min_rounds`` (the round limit), the
    group is cleared in a single attempt with zero coin cost. Otherwise the
    player revives (paying ``CoinConsumption`` coins each time); already-defeated
    bosses stay dead, so progress accumulates across attempts.
    """
    boss_hps = spec.boss_hps
    if not boss_hps:
        return BossPlan(0, [], True, 0, 0, True, [])

    limit = spec.min_rounds
    full_rounds, full_seq = solve_boss_group(boss_hps, spec.skills)
    names = _skill_names(spec)

    if full_rounds <= limit:
        return BossPlan(
            min_rounds=full_rounds,
            skill_sequence=full_seq,
            feasible_within_limit=True,
            revives=0,
            coins_spent=0,
            cleared=True,
            skill_names=[names[k] for k in full_seq],
        )

    # Round limit too tight for one attempt: simulate limited attempts + revives.
    sequence: list[int] = []
    revives = 0
    coins_left = available_coins if available_coins is not None else 10**9
    bi, hp = 0, boss_hps[0]
    cleared = False
    while True:
        _, seq = solve_boss_group(boss_hps, spec.skills, start_index=bi, start_hp=hp)
        # play at most `limit` rounds of this attempt
        played = seq[:limit]
        sequence.extend(played)
        bi, hp = _advance(boss_hps, spec.skills, bi, hp, played)
        if bi >= len(boss_hps):
            cleared = True
            break
        # attempt failed -> must revive
        if coins_left < spec.coin_consumption:
            break
        coins_left -= spec.coin_consumption
        revives += 1
        hp = boss_hps[bi]  # boss HP for the failed boss resets on a fresh life

    return BossPlan(
        min_rounds=full_rounds,
        skill_sequence=sequence,
        feasible_within_limit=False,
        revives=revives,
        coins_spent=revives * spec.coin_consumption,
        cleared=cleared,
        skill_names=[names[k] for k in sequence],
    )


def _advance(boss_hps, skills, bi, hp, seq) -> tuple[int, int]:
    """Apply a skill sequence and return the resulting (boss_index, hp)."""
    cd = [0] * len(skills)
    for k in seq:
        if cd[k] != 0:
            continue
        hp -= skills[k].damage
        for j in range(len(skills)):
            cd[j] = max(0, cd[j] - 1)
        cd[k] = skills[k].cooldown
        if hp <= 0:
            bi += 1
            if bi >= len(boss_hps):
                return bi, 0
            hp = boss_hps[bi]
    return bi, hp


def _skill_names(spec: MazeSpec) -> list[str]:
    return [f"S{i}(d{s.damage},cd{s.cooldown})" for i, s in enumerate(spec.skills)]
