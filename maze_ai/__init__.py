"""maze_ai: deterministic classic-algorithm AI player for the course maze game.

This package replaces the earlier neural/DQN/LoRA experiments. It is built
entirely from the classic algorithm paradigms the assignment grades:

- BFS / shortest path on the maze graph (the maze is a perfect maze = a tree).
- Bitmask dynamic programming for the optimal resource-collection route.
- Branch-and-bound for the minimum-round BOSS battle and optimal skill sequence.
- Greedy for the 3x3 local real-time resource pickup.

It parses the *official* task input format directly (see ``spec.py``):

    {
      "maze": [[...chars...], ...],   # S E B # ' ' G(coin) T(trap)
      "B": [hp0, hp1, ...],           # boss group HP, fixed defeat order
      "PlayerSkills": [[dmg, cd], ...],
      "minRouds": 20,                 # round limit for one BOSS attempt
      "CoinConsumption": 5            # coins paid to revive on a failed attempt
    }
"""

from .spec import MazeSpec, COIN_VALUE, TRAP_DAMAGE
from .boss import BossPlan, solve_boss_group, plan_boss_fight
from .planner import OptimalPlan, plan_full_exploration
from .greedy3x3 import greedy_3x3_run
from .simulate import RunResult, simulate

__all__ = [
    "MazeSpec",
    "COIN_VALUE",
    "TRAP_DAMAGE",
    "BossPlan",
    "solve_boss_group",
    "plan_boss_fight",
    "OptimalPlan",
    "plan_full_exploration",
    "greedy_3x3_run",
    "RunResult",
    "simulate",
]
