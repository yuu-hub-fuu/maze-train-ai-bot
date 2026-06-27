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


def _spec(grid, boss_hps, mr=20, cc=5, skills=None):
    sk = skills or [Skill(8, 4), Skill(2, 0), Skill(4, 2), Skill(6, 3)]
    return MazeSpec(grid=grid, boss_hps=boss_hps, skills=sk, min_rounds=mr, coin_consumption=cc)


def _solve(spec):
    plan = plan_full_exploration(spec)
    return plan, simulate(spec, plan.moves)


def test_edge_no_coins_no_boss_no_traps():
    _, r = _solve(_spec(["#####", "#SBE#", "#####"], [10]))
    assert r.success and r.cleared_boss
    _, r = _solve(_spec(["#####", "#S E#", "#####"], []))  # no boss cell
    assert r.success
    _, r = _solve(_spec(["#######", "#SG BE#", "#######"], [10]))  # no traps
    assert r.success


def test_edge_tight_round_limit_revive_persists_progress():
    # HP 60, only 2 rounds per life -> must revive several times; damage persists.
    plan, r = _solve(_spec(["#########", "#SGGGGBE#", "#########"], [60], mr=2, cc=1))
    assert r.success, r.reason
    assert plan.boss_plan.revives > 0
    assert not plan.boss_plan.feasible_within_limit


def test_edge_unbeatable_boss_fails_gracefully():
    # All skills deal 0 damage: must report failure, not raise.
    plan, r = _solve(_spec(["#####", "#SBE#", "#####"], [10], skills=[Skill(0, 0)]))
    assert not r.success
    assert not plan.boss_plan.cleared


def test_edge_non_perfect_maze_does_not_crash():
    loop = ["#####", "#S G#", "# # #", "#GBE#", "#####"]
    _, r = _solve(_spec(loop, [10]))
    assert r.success  # still solvable (optimality not guaranteed off-tree)


def test_tree_orienteering_matches_bitmask_dp():
    # The polynomial tree DP must give the same (optimal) score as bitmask DP.
    from maze_ai.planner import _score_path
    from maze_ai.tree_orienteering import plan_tree_orienteering
    for seed in range(20):
        spec = generate_maze([7, 9, 11][seed % 3], seed=seed)
        bm, _ = _solve(spec)  # bitmask DP (<=15 coins)
        tr = plan_tree_orienteering(spec)
        res, steps, _, _, _ = _score_path(spec, tr.coord_path, 0)
        assert abs(bm.score - res / steps) < 1e-9, seed


def test_coin_dense_map_uses_tree_dp_and_succeeds():
    spec = generate_maze(31, seed=5, coin_ratio=0.10, trap_ratio=0.06)
    assert len(spec.coins) > 15
    plan, r = _solve(spec)
    assert plan.method == "tree_orienteering"
    assert r.success and r.score > 0


if __name__ == "__main__":
    test_boss_solver_is_minimal()
    test_official_maze_solved()
    test_generated_mazes_all_succeed()
    test_moves_are_legal()
    test_edge_no_coins_no_boss_no_traps()
    test_edge_tight_round_limit_revive_persists_progress()
    test_edge_unbeatable_boss_fails_gracefully()
    test_edge_non_perfect_maze_does_not_crash()
    test_tree_orienteering_matches_bitmask_dp()
    test_coin_dense_map_uses_tree_dp_and_succeeds()
    print("all maze_ai tests passed")
