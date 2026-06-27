"""Tests for the classic-algorithm maze AI player (maze_ai).

Run directly (``python tests/test_maze_ai.py``) or via pytest.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_ai.spec import MazeSpec, Skill
from maze_ai.boss import solve_boss_group, plan_boss_fight
from maze_ai.generate import generate_maze
from maze_ai.planner import plan_full_exploration
from maze_ai.simulate import simulate

EXAMPLE = Path(__file__).resolve().parents[1] / "maze_ai" / "examples" / "maze_15_15.json"


def _brute_boss(boss_hps, skills, cap=40):
    n = len(boss_hps)
    best = [cap + 1]

    def dfs(bi, hp, cd, rounds):
        if rounds >= best[0]:
            return
        if bi >= n:
            best[0] = min(best[0], rounds)
            return
        for k in range(len(skills)):
            if cd[k] != 0:
                continue
            nhp = hp - skills[k].damage
            ncd = [max(0, c - 1) for c in cd]
            ncd[k] = skills[k].cooldown
            nbi, nh = (bi, nhp) if nhp > 0 else (bi + 1, boss_hps[bi + 1] if bi + 1 < n else 0)
            dfs(nbi, nh, tuple(ncd), rounds + 1)

    dfs(0, boss_hps[0], tuple(0 for _ in skills), 0)
    return best[0]


def test_boss_solver_is_minimal():
    rng = random.Random(13)
    for _ in range(200):
        hps = [rng.randint(3, 16) for _ in range(rng.randint(1, 4))]
        skills = [Skill(rng.randint(2, 9), rng.randint(0, 4)) for _ in range(rng.randint(1, 4))]
        if all(s.cooldown > 0 for s in skills):
            skills[0] = Skill(skills[0].damage, 0)
        rounds, seq = solve_boss_group(hps, skills)
        assert rounds == _brute_boss(hps, skills)
        # the recovered sequence actually clears the group
        bi, hp, cd = 0, hps[0], [0] * len(skills)
        for k in seq:
            assert cd[k] == 0
            hp -= skills[k].damage
            cd = [max(0, c - 1) for c in cd]
            cd[k] = skills[k].cooldown
            if hp <= 0:
                bi += 1
                hp = hps[bi] if bi < len(hps) else 0
        assert bi == len(hps)


def test_official_maze_solved():
    spec = MazeSpec.from_json(EXAMPLE)
    plan = plan_full_exploration(spec)
    run = simulate(spec, plan.moves)
    assert run.success
    assert run.cleared_boss
    assert plan.boss_plan.feasible_within_limit
    # planner's internal score must equal the simulator's score
    assert abs(plan.score - run.score) < 1e-9
    assert run.score > 3.0  # comfortably strong


def test_generated_mazes_all_succeed():
    for seed in range(30):
        spec = generate_maze(15, seed=seed)
        plan = plan_full_exploration(spec)
        run = simulate(spec, plan.moves)
        assert run.success, f"seed {seed} failed: {run.reason}"
        assert abs(plan.score - run.score) < 1e-9


def test_moves_are_legal():
    spec = generate_maze(15, seed=1)
    plan = plan_full_exploration(spec)
    run = simulate(spec, plan.moves)
    assert run.reason in ("ok", "exit")  # never hit a wall / illegal move
    assert run.success


if __name__ == "__main__":
    test_boss_solver_is_minimal()
    test_official_maze_solved()
    test_generated_mazes_all_succeed()
    test_moves_are_legal()
    print("all maze_ai tests passed")
