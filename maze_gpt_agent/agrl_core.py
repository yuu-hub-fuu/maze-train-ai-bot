from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
import heapq
import json
import math
import random
import time
from pathlib import Path
from typing import Any


WALL = "#"
EMPTY = "."
START = "S"
EXIT = "E"
BOSS = "B"
COIN = "G"
TRAP = "T"
COIN_VALUE = 50
TRAP_DAMAGE = 30
MOVES = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}
HIGH_LEVEL_ACTIONS = (
    "NEAREST_GOLD",
    "BEST_VALUE_GOLD",
    "MAIN_PATH_GOLD",
    "GO_BOSS",
    "GO_EXIT",
    "AVOID_TRAP",
    "EXPLORE",
)


Coord = tuple[int, int]


@dataclass
class BossConfig:
    hp: int = 25
    turn_limit: int = 5
    revive_cost: int = 60


@dataclass
class Skill:
    name: str
    damage: int
    cooldown: int = 0


@dataclass
class MazeSample:
    sample_id: str
    seed: int
    rows: int
    cols: int
    grid: list[str]
    start: Coord
    end: Coord
    boss: Coord
    coins: list[Coord]
    traps: list[Coord]
    difficulty: str
    boss_config: BossConfig
    skill_config: list[Skill]
    train_split: str = "train"
    expert_solution: dict[str, Any] | None = None

    def char_at(self, pos: Coord) -> str:
        r, c = pos
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return WALL
        return self.grid[r][c]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start"] = list(self.start)
        data["end"] = list(self.end)
        data["boss"] = list(self.boss)
        data["coins"] = [list(x) for x in self.coins]
        data["traps"] = [list(x) for x in self.traps]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MazeSample":
        return cls(
            sample_id=data["sample_id"],
            seed=int(data["seed"]),
            rows=int(data["rows"]),
            cols=int(data["cols"]),
            grid=list(data["grid"]),
            start=tuple(data["start"]),
            end=tuple(data["end"]),
            boss=tuple(data["boss"]),
            coins=[tuple(x) for x in data.get("coins", [])],
            traps=[tuple(x) for x in data.get("traps", [])],
            difficulty=data.get("difficulty", "Medium"),
            boss_config=BossConfig(**data.get("boss_config", {})),
            skill_config=[Skill(**x) for x in data.get("skill_config", default_skills_dict())],
            train_split=data.get("train_split", "train"),
            expert_solution=data.get("expert_solution"),
        )


@dataclass
class PlayerState:
    position: Coord
    resource: int = 0
    steps: int = 0
    collected_coins: set[Coord] = field(default_factory=set)
    triggered_traps: set[Coord] = field(default_factory=set)
    boss_defeated: bool = False
    alive: bool = True
    done: bool = False
    known: dict[Coord, str] = field(default_factory=dict)
    path_history: list[Coord] = field(default_factory=list)
    decision_history: list[dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "PlayerState":
        return PlayerState(
            position=self.position,
            resource=self.resource,
            steps=self.steps,
            collected_coins=set(self.collected_coins),
            triggered_traps=set(self.triggered_traps),
            boss_defeated=self.boss_defeated,
            alive=self.alive,
            done=self.done,
            known=dict(self.known),
            path_history=list(self.path_history),
            decision_history=list(self.decision_history),
        )


@dataclass
class Target:
    target_id: str
    target_type: str
    position: Coord
    gain: int
    distance: int
    risk: int
    future_cost: int
    score: float
    feasible: bool


@dataclass
class PathResult:
    path: list[Coord]
    actions: list[str]
    steps: int
    trap_loss: int
    coin_gain: int
    remaining_resource: int
    score: float
    feasible: bool


@dataclass
class BossResult:
    success: bool
    min_rounds: int
    skill_sequence: list[str]
    revive_needed: bool
    remaining_resource: int


@dataclass
class ExpertSolution:
    recommended_targets: list[dict[str, Any]]
    recommended_path: list[Coord]
    collected_coins: list[Coord]
    boss_skill_sequence: list[str]
    final_resource: int
    total_steps: int
    final_score: float
    success: bool


@dataclass
class RunResult:
    strategy: str
    sample_id: str
    difficulty: str
    success: bool
    boss_success: bool
    final_resource: int
    total_steps: int
    final_score: float
    trap_count: int
    coin_count: int
    boss_rounds: int
    runtime_ms: float
    frames: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "sample_id": self.sample_id,
            "difficulty": self.difficulty,
            "success": self.success,
            "boss_success": self.boss_success,
            "final_resource": self.final_resource,
            "total_steps": self.total_steps,
            "final_score": self.final_score,
            "trap_count": self.trap_count,
            "coin_count": self.coin_count,
            "boss_rounds": self.boss_rounds,
            "runtime_ms": self.runtime_ms,
        }


def default_skills() -> list[Skill]:
    return [Skill("normal_attack", 5, 0), Skill("ultimate", 10, 2)]


def default_skills_dict() -> list[dict[str, Any]]:
    return [asdict(x) for x in default_skills()]


def boss_for_difficulty(difficulty: str) -> BossConfig:
    table = {
        "Easy": BossConfig(15, 4, 40),
        "Medium": BossConfig(25, 5, 60),
        "Hard": BossConfig(35, 6, 80),
        "Extreme": BossConfig(45, 6, 100),
    }
    return table.get(difficulty, table["Medium"])


def generate_agrl_maze(size: int = 11, seed: int = 42, difficulty: str = "Medium", split: str = "train") -> MazeSample:
    rng = random.Random(seed)
    if size % 2 == 0:
        size += 1
    size = max(5, size)
    grid = [[WALL for _ in range(size)] for _ in range(size)]
    start = (1, 1)
    grid[start[0]][start[1]] = EMPTY
    stack = [start]
    dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
    while stack:
        cur = stack[-1]
        candidates: list[Coord] = []
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nxt = (cur[0] + dr, cur[1] + dc)
            if 1 <= nxt[0] < size - 1 and 1 <= nxt[1] < size - 1 and grid[nxt[0]][nxt[1]] == WALL:
                candidates.append(nxt)
        if not candidates:
            stack.pop()
            continue
        nxt = candidates[0]
        wall = ((cur[0] + nxt[0]) // 2, (cur[1] + nxt[1]) // 2)
        grid[wall[0]][wall[1]] = EMPTY
        grid[nxt[0]][nxt[1]] = EMPTY
        stack.append(nxt)

    temp = ["".join(row) for row in grid]
    end = farthest_cell(temp, start)
    path = positions_from_actions(start, shortest_path(temp, start, end) or [])
    boss_idx = min(max(1, len(path) - rng.randint(2, 4)), len(path) - 2)
    boss = path[boss_idx]
    grid[start[0]][start[1]] = START
    grid[end[0]][end[1]] = EXIT
    grid[boss[0]][boss[1]] = BOSS

    path_set = set(path)
    empty_cells = [(r, c) for r in range(size) for c in range(size) if grid[r][c] == EMPTY]
    main_cells = [p for p in path[1:-1] if grid[p[0]][p[1]] == EMPTY]
    branch_cells = [p for p in empty_cells if p not in path_set]
    rng.shuffle(main_cells)
    rng.shuffle(branch_cells)
    cfg = {
        "Easy": (4, 1, 0, 1, 0),
        "Medium": (3, 3, 1, 2, 1),
        "Hard": (2, 4, 2, 4, 1),
        "Extreme": (1, 5, 3, 5, 2),
    }.get(difficulty, (3, 3, 1, 2, 1))
    main_gold, branch_gold, bait_gold, light_traps, required_traps = cfg
    place_many(grid, main_cells, COIN, main_gold)
    place_many(grid, branch_cells, COIN, branch_gold)
    bait_neighbors = neighbors_of_tiles(grid, COIN)
    rng.shuffle(bait_neighbors)
    place_many(grid, bait_neighbors, TRAP, bait_gold)
    place_many(grid, branch_cells, TRAP, light_traps)
    required_pool = [p for p in main_cells if p not in (start, boss, end)]
    place_many(grid, required_pool, TRAP, required_traps)

    rows = ["".join(row) for row in grid]
    coins = find_tiles(rows, COIN)
    traps = find_tiles(rows, TRAP)
    sample = MazeSample(
        sample_id=f"{split}-{difficulty}-{seed}",
        seed=seed,
        rows=size,
        cols=size,
        grid=rows,
        start=start,
        end=end,
        boss=boss,
        coins=coins,
        traps=traps,
        difficulty=difficulty,
        boss_config=boss_for_difficulty(difficulty),
        skill_config=default_skills(),
        train_split=split,
    )
    if not validate_sample(sample)["ok"]:
        return generate_agrl_maze(size=size, seed=seed + 10000, difficulty=difficulty, split=split)
    return sample


def validate_sample(sample: MazeSample) -> dict[str, Any]:
    issues: list[str] = []
    if sample.start is None:
        issues.append("missing_start")
    if sample.end is None:
        issues.append("missing_end")
    if sample.boss is None:
        issues.append("missing_boss")
    if not shortest_path(sample.grid, sample.start, sample.boss):
        issues.append("start_to_boss_unreachable")
    if not shortest_path(sample.grid, sample.boss, sample.end):
        issues.append("boss_to_end_unreachable")
    if len(sample.coins) < 1:
        issues.append("too_few_coins")
    return {"ok": not issues, "issues": issues}


def local_greedy_3x3(sample: MazeSample, state: PlayerState) -> Target | None:
    best: Target | None = None
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            pos = (state.position[0] + dr, state.position[1] + dc)
            ch = visible_char(sample, state, pos)
            if ch == WALL or pos == state.position:
                continue
            dist = abs(dr) + abs(dc)
            if dist <= 0 or dist > 2:
                continue
            gain = sample_value(sample, state, pos)
            risk = TRAP_DAMAGE if ch == TRAP else 0
            score = (gain - risk) / (dist + 1)
            target = Target(f"local-{pos}", "local", pos, gain, dist, risk, 0, score, score > 0)
            if target.feasible and (best is None or target.score > best.score):
                best = target
    return best


def pctsp_targets(sample: MazeSample, state: PlayerState, k: int = 8) -> list[Target]:
    targets: list[Target] = []
    boss_need = sample.boss_config.revive_cost
    main_path = set(positions_from_actions(sample.start, shortest_path(sample.grid, sample.start, sample.boss) or []))
    for coin in known_positions(state, COIN):
        if coin in state.collected_coins:
            continue
        path = rcspp_path(sample, state, coin, require_boss_resource=False)
        if not path.feasible:
            continue
        future_to_boss = len(shortest_path(sample.grid, coin, sample.boss) or []) or 999
        future_to_end = len(shortest_path(sample.grid, sample.boss, sample.end) or []) or 999
        route_bonus = 15 if coin in main_path else 0
        need_bonus = 25 if state.resource < boss_need else -10
        score = COIN_VALUE + route_bonus + need_bonus - path.steps - path.trap_loss - 0.35 * (future_to_boss + future_to_end)
        targets.append(
            Target(
                f"coin-{coin[0]}-{coin[1]}",
                "coin",
                coin,
                COIN_VALUE,
                path.steps,
                path.trap_loss,
                future_to_boss + future_to_end,
                score,
                path.feasible,
            )
        )
    return sorted(targets, key=lambda t: t.score, reverse=True)[:k]


def rcspp_path(sample: MazeSample, state: PlayerState, goal: Coord, require_boss_resource: bool = False) -> PathResult:
    risk_weight = 2.0 if state.resource < sample.boss_config.revive_cost else 0.8
    heap: list[tuple[float, int, Coord, int, int, list[Coord], list[str]]] = [(0.0, 0, state.position, 0, 0, [state.position], [])]
    best: dict[Coord, tuple[int, int]] = {}
    counter = 0
    while heap:
        cost, steps, pos, trap_loss, coin_gain, path, actions = heapq.heappop(heap)
        prev = best.get(pos)
        if prev and prev[0] <= steps and prev[1] <= trap_loss:
            continue
        best[pos] = (steps, trap_loss)
        if pos == goal:
            remaining = state.resource + coin_gain - trap_loss
            feasible = remaining >= 0 and (not require_boss_resource or remaining >= sample.boss_config.revive_cost)
            return PathResult(path, actions, steps, trap_loss, coin_gain, remaining, cost, feasible)
        for action, nxt in memory_neighbors(sample, state, pos, goal):
            if nxt in path:
                continue
            ch = visible_char(sample, state, nxt)
            add_loss = TRAP_DAMAGE if ch == TRAP and nxt not in state.triggered_traps else 0
            add_gain = COIN_VALUE if ch == COIN and nxt not in state.collected_coins else 0
            new_loss = trap_loss + add_loss
            new_gain = coin_gain + add_gain
            remaining = state.resource + new_gain - new_loss
            if remaining < -sample.boss_config.revive_cost:
                continue
            counter += 1
            new_steps = steps + 1
            new_cost = new_steps + risk_weight * new_loss + manhattan(nxt, goal)
            heapq.heappush(heap, (new_cost, new_steps, nxt, new_loss, new_gain, path + [nxt], actions + [action]))
    return PathResult([], [], 0, 0, 0, state.resource, float("inf"), False)


def solve_boss_battle(boss: BossConfig, skills: list[Skill], resource: int) -> BossResult:
    best_seq: list[str] | None = None
    best_rounds = boss.turn_limit + 1
    max_damage = max(s.damage for s in skills)

    def dfs(round_idx: int, hp: int, cooldowns: tuple[int, ...], seq: list[str]) -> None:
        nonlocal best_seq, best_rounds
        if hp <= 0:
            if round_idx < best_rounds:
                best_rounds = round_idx
                best_seq = list(seq)
            return
        if round_idx >= boss.turn_limit:
            return
        lower = math.ceil(hp / max_damage)
        if round_idx + lower >= best_rounds:
            return
        for idx, skill in sorted(enumerate(skills), key=lambda x: x[1].damage, reverse=True):
            if cooldowns[idx] > 0:
                continue
            next_cds = []
            for j, cd in enumerate(cooldowns):
                next_cds.append(max(0, cd - 1))
            next_cds[idx] = skill.cooldown
            dfs(round_idx + 1, hp - skill.damage, tuple(next_cds), seq + [skill.name])

    dfs(0, boss.hp, tuple(0 for _ in skills), [])
    if best_seq is not None:
        return BossResult(True, best_rounds, best_seq, False, resource)
    if resource >= boss.revive_cost:
        return BossResult(False, boss.turn_limit, [], True, resource - boss.revive_cost)
    return BossResult(False, boss.turn_limit, [], False, resource)


def run_strategy(sample: MazeSample, strategy: str, q_table: dict[str, dict[str, float]] | None = None, max_steps: int | None = None) -> RunResult:
    started = time.perf_counter()
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    frames = [frame(sample, state, "START", "start", None)]
    max_steps = max_steps or sample.rows * sample.cols * 4
    boss_result = BossResult(False, 0, [], False, state.resource)
    while state.alive and not state.done and state.steps < max_steps:
        target, reason, high_action = choose_target(sample, state, strategy, q_table)
        if target is None:
            state.done = True
            break
        path = rcspp_path(sample, state, target.position, require_boss_resource=(target.target_type == "boss"))
        if not path.feasible or not path.actions:
            state.decision_history.append({"action": high_action, "reason": "infeasible_target", "target": asdict(target)})
            state.done = True
            state.alive = False
            break
        state.decision_history.append({"action": high_action, "reason": reason, "target": asdict(target), "path_score": path.score})
        for action, pos in zip(path.actions, path.path[1:]):
            apply_move(sample, state, pos)
            frames.append(frame(sample, state, action, tile_event(sample, state, pos), target))
            if state.steps >= max_steps:
                break
        if state.position == sample.boss and not state.boss_defeated:
            boss_result = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
            if boss_result.success and state.resource >= sample.boss_config.revive_cost:
                state.boss_defeated = True
                frames.append(frame(sample, state, "BOSS_FIGHT", "boss_clear:" + ",".join(boss_result.skill_sequence), target))
            elif boss_result.revive_needed:
                state.resource = boss_result.remaining_resource
                state.alive = False
                state.done = True
                frames.append(frame(sample, state, "BOSS_FIGHT", "boss_fail_revive_spent", target))
            else:
                state.alive = False
                state.done = True
                frames.append(frame(sample, state, "BOSS_FIGHT", "boss_fail_game_over", target))
        if state.position == sample.end:
            if state.boss_defeated:
                state.done = True
            else:
                state.done = True
                state.alive = False
    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    return RunResult(
        strategy=strategy,
        sample_id=sample.sample_id,
        difficulty=sample.difficulty,
        success=success,
        boss_success=state.boss_defeated,
        final_resource=state.resource,
        total_steps=state.steps,
        final_score=state.resource / max(1, state.steps),
        trap_count=len(state.triggered_traps),
        coin_count=len(state.collected_coins),
        boss_rounds=boss_result.min_rounds,
        runtime_ms=(time.perf_counter() - started) * 1000,
        frames=frames,
    )


def choose_target(
    sample: MazeSample, state: PlayerState, strategy: str, q_table: dict[str, dict[str, float]] | None = None
) -> tuple[Target | None, str, str]:
    if strategy == "shortest":
        if not state.boss_defeated:
            if state.known.get(sample.boss) != BOSS:
                explore = frontier_target(sample, state)
                return explore, "shortest_memory_explore_until_boss_seen", "EXPLORE"
            return Target("boss", "boss", sample.boss, 0, 0, 0, 0, 0, True), "shortest_to_boss", "GO_BOSS"
        if state.known.get(sample.end) != EXIT:
            explore = frontier_target(sample, state)
            return explore, "shortest_memory_explore_until_exit_seen", "EXPLORE"
        return Target("exit", "exit", sample.end, 0, 0, 0, 0, 0, True), "shortest_to_exit", "GO_EXIT"
    if strategy == "greedy3x3":
        local = local_greedy_3x3(sample, state)
        if local:
            return local, "local_3x3_best_score", "BEST_VALUE_GOLD"
    if strategy == "rl" and q_table:
        high = best_q_action(q_table, rl_state_key(sample, state), epsilon=0.0)
        target = target_from_high_action(sample, state, high)
        if target:
            return target, "q_learning_high_level_policy", high
    if strategy == "classic":
        local = local_greedy_3x3(sample, state)
        if local and local.score >= 15:
            return local, "local_3x3_positive_resource", "BEST_VALUE_GOLD"
    return classic_target(sample, state)


def classic_target(sample: MazeSample, state: PlayerState) -> tuple[Target | None, str, str]:
    if state.boss_defeated:
        if state.known.get(sample.end) != EXIT:
            explore = frontier_target(sample, state)
            if explore:
                return explore, "boss_defeated_explore_until_exit_seen", "EXPLORE"
        return Target("exit", "exit", sample.end, 0, 0, 0, 0, 0, True), "boss_defeated_go_exit", "GO_EXIT"
    targets = pctsp_targets(sample, state)
    if state.resource >= sample.boss_config.revive_cost and state.known.get(sample.boss) == BOSS:
        boss_path = rcspp_path(sample, state, sample.boss, require_boss_resource=True)
        if boss_path.feasible:
            return Target("boss", "boss", sample.boss, 0, boss_path.steps, boss_path.trap_loss, 0, 999, True), "resource_enough_go_boss", "GO_BOSS"
    if targets:
        return targets[0], "pctsp_best_gold", "BEST_VALUE_GOLD"
    explore = frontier_target(sample, state)
    if explore:
        return explore, "explore_unknown_frontier", "EXPLORE"
    return Target("boss", "boss", sample.boss, 0, 0, 0, 0, 0, True), "memory_exhausted_go_boss", "GO_BOSS"


def target_from_high_action(sample: MazeSample, state: PlayerState, action: str) -> Target | None:
    targets = pctsp_targets(sample, state)
    if action == "EXPLORE":
        return frontier_target(sample, state)
    if action == "NEAREST_GOLD":
        return min(targets, key=lambda t: t.distance, default=None)
    if action == "BEST_VALUE_GOLD":
        return targets[0] if targets else None
    if action == "MAIN_PATH_GOLD":
        main = set(positions_from_actions(sample.start, shortest_path(sample.grid, sample.start, sample.boss) or []))
        near = [t for t in targets if t.position in main]
        return near[0] if near else (targets[0] if targets else None)
    if action == "GO_BOSS":
        return Target("boss", "boss", sample.boss, 0, 0, 0, 0, 0, True) if state.known.get(sample.boss) == BOSS else frontier_target(sample, state)
    if action == "GO_EXIT":
        return Target("exit", "exit", sample.end, 0, 0, 0, 0, 0, state.boss_defeated) if state.known.get(sample.end) == EXIT else frontier_target(sample, state)
    if action == "AVOID_TRAP":
        safe = [t for t in targets if t.risk == 0]
        return safe[0] if safe else (targets[0] if targets else None)
    return None


def train_q_learning(samples: list[MazeSample], episodes: int = 200, alpha: float = 0.4, gamma: float = 0.85) -> dict[str, dict[str, float]]:
    q: dict[str, dict[str, float]] = defaultdict(lambda: {a: 0.0 for a in HIGH_LEVEL_ACTIONS})
    for ep in range(episodes):
        epsilon = max(0.05, 0.8 * (1 - ep / max(1, episodes)))
        sample = samples[ep % len(samples)]
        state = PlayerState(position=sample.start, path_history=[sample.start])
        observe_3x3(sample, state)
        for _ in range(sample.rows * sample.cols):
            key = rl_state_key(sample, state)
            action = best_q_action(q, key, epsilon)
            target = target_from_high_action(sample, state, action)
            before_score = state.resource / max(1, state.steps)
            if target is None or (target.target_type == "exit" and not state.boss_defeated):
                reward = -50
                done = False
                next_key = key
            else:
                path = rcspp_path(sample, state, target.position, require_boss_resource=(target.target_type == "boss"))
                if not path.feasible or not path.actions:
                    reward = -50
                    done = False
                    next_key = key
                else:
                    for pos in path.path[1:]:
                        apply_move(sample, state, pos)
                    reward = state.resource - before_score - path.steps
                    if target.target_type == "coin" and target.score > 0:
                        reward += 10
                    if state.position == sample.boss and not state.boss_defeated:
                        br = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
                        if br.success and state.resource >= sample.boss_config.revive_cost:
                            state.boss_defeated = True
                            reward += 100
                        else:
                            state.alive = False
                            reward -= 100
                    done = (state.position == sample.end and state.boss_defeated) or not state.alive
                    if done and state.position == sample.end and state.boss_defeated:
                        reward += 200 + state.resource - state.steps
                    next_key = rl_state_key(sample, state)
            q[key][action] += alpha * (reward + gamma * max(q[next_key].values()) - q[key][action])
            if done:
                break
    return {k: dict(v) for k, v in q.items()}


def rl_state_key(sample: MazeSample, state: PlayerState) -> str:
    nearest_gold = min((len(memory_shortest_path(sample, state, c) or []) for c in known_positions(state, COIN) if c not in state.collected_coins), default=99)
    boss_dist = len(memory_shortest_path(sample, state, sample.boss) or []) if state.known.get(sample.boss) == BOSS else 99
    exit_dist = len(memory_shortest_path(sample, state, sample.end) or []) if state.known.get(sample.end) == EXIT else 99
    local = "".join(ch for row in vision_3x3(sample, state) for ch in row)
    return "|".join(
        [
            f"d={sample.difficulty}",
            f"res={bucket(state.resource, [0, 50, 100, 150, 250])}",
            f"step={bucket(state.steps, [10, 25, 50, 100])}",
            f"boss={int(state.boss_defeated)}",
            f"ng={bucket(nearest_gold, [1, 3, 6, 10])}",
            f"bd={bucket(boss_dist, [2, 5, 10, 20])}",
            f"ed={bucket(exit_dist, [2, 5, 10, 20])}",
            f"coins={bucket(len(sample.coins) - len(state.collected_coins), [0, 2, 5, 10])}",
            f"enough={int(state.resource >= sample.boss_config.revive_cost)}",
            f"local={local}",
        ]
    )


def best_q_action(q: dict[str, dict[str, float]], key: str, epsilon: float) -> str:
    if random.random() < epsilon:
        return random.choice(HIGH_LEVEL_ACTIONS)
    values = q.setdefault(key, {a: 0.0 for a in HIGH_LEVEL_ACTIONS})
    return max(HIGH_LEVEL_ACTIONS, key=lambda a: values.get(a, 0.0))


def apply_move(sample: MazeSample, state: PlayerState, pos: Coord) -> None:
    state.position = pos
    state.steps += 1
    state.path_history.append(pos)
    ch = sample.char_at(pos)
    if ch == COIN and pos not in state.collected_coins:
        state.collected_coins.add(pos)
        state.resource += COIN_VALUE
    elif ch == TRAP and pos not in state.triggered_traps:
        state.triggered_traps.add(pos)
        state.resource -= TRAP_DAMAGE
    observe_3x3(sample, state)


def frame(sample: MazeSample, state: PlayerState, action: str, event: str, target: Target | None) -> dict[str, Any]:
    rows = [list(r) for r in sample.grid]
    for r, c in state.collected_coins:
        if rows[r][c] == COIN:
            rows[r][c] = EMPTY
    for r, c in state.triggered_traps:
        if rows[r][c] == TRAP:
            rows[r][c] = EMPTY
    if state.boss_defeated:
        r, c = sample.boss
        rows[r][c] = EMPTY
    r, c = state.position
    rows[r][c] = "@"
    return {
        "grid": ["".join(row) for row in rows],
        "action": action,
        "event": event,
        "pos": list(state.position),
        "resource": state.resource,
        "gold": state.resource,
        "steps": state.steps,
        "score": state.resource / max(1, state.steps),
        "target": asdict(target) if target else None,
        "decision": state.decision_history[-1] if state.decision_history else None,
        "boss_defeated": state.boss_defeated,
    }


def tile_event(sample: MazeSample, state: PlayerState, pos: Coord) -> str:
    ch = sample.char_at(pos)
    if ch == COIN and pos in state.collected_coins:
        return "coin"
    if ch == TRAP and pos in state.triggered_traps:
        return "trap"
    if ch == BOSS:
        return "boss"
    if ch == EXIT:
        return "exit"
    return "move"


def vision_3x3(sample: MazeSample, state: PlayerState) -> list[str]:
    out: list[str] = []
    for r in range(state.position[0] - 1, state.position[0] + 2):
        chars: list[str] = []
        for c in range(state.position[1] - 1, state.position[1] + 2):
            pos = (r, c)
            chars.append("@" if pos == state.position else visible_char(sample, state, pos))
        out.append("".join(chars))
    return out


def visible_char(sample: MazeSample, state: PlayerState, pos: Coord) -> str:
    ch = sample.char_at(pos)
    if ch == COIN and pos in state.collected_coins:
        return EMPTY
    if ch == TRAP and pos in state.triggered_traps:
        return EMPTY
    if ch == BOSS and state.boss_defeated:
        return EMPTY
    return ch



def observe_3x3(sample: MazeSample, state: PlayerState) -> None:
    for r in range(state.position[0] - 1, state.position[0] + 2):
        for c in range(state.position[1] - 1, state.position[1] + 2):
            pos = (r, c)
            if 0 <= r < sample.rows and 0 <= c < sample.cols:
                state.known[pos] = sample.char_at(pos)


def known_positions(state: PlayerState, tile: str) -> list[Coord]:
    return [pos for pos, ch in state.known.items() if ch == tile]


def known_char(sample: MazeSample, state: PlayerState, pos: Coord) -> str:
    if pos == state.position:
        return "@"
    ch = state.known.get(pos, "?")
    if ch == COIN and pos in state.collected_coins:
        return EMPTY
    if ch == TRAP and pos in state.triggered_traps:
        return EMPTY
    if ch == BOSS and state.boss_defeated:
        return EMPTY
    return ch


def memory_neighbors(sample: MazeSample, state: PlayerState, pos: Coord, goal: Coord | None = None):
    for action, (dr, dc) in MOVES.items():
        nxt = (pos[0] + dr, pos[1] + dc)
        ch = state.known.get(nxt)
        if nxt == goal and ch and ch != WALL:
            yield action, nxt
        elif ch and ch not in (WALL, "?"):
            yield action, nxt


def memory_shortest_path(sample: MazeSample, state: PlayerState, goal: Coord) -> list[str] | None:
    q = deque([state.position])
    parent: dict[Coord, tuple[Coord, str]] = {}
    seen = {state.position}
    while q:
        cur = q.popleft()
        if cur == goal:
            actions: list[str] = []
            while cur != state.position:
                prev, action = parent[cur]
                actions.append(action)
                cur = prev
            return list(reversed(actions))
        for action, nxt in memory_neighbors(sample, state, cur, goal):
            if nxt not in seen:
                seen.add(nxt)
                parent[nxt] = (cur, action)
                q.append(nxt)
    return None


def frontier_target(sample: MazeSample, state: PlayerState) -> Target | None:
    best: tuple[int, int, Coord] | None = None
    for pos, ch in state.known.items():
        if ch == WALL:
            continue
        unknown_adj = 0
        for dr, dc in MOVES.values():
            nxt = (pos[0] + dr, pos[1] + dc)
            if 0 <= nxt[0] < sample.rows and 0 <= nxt[1] < sample.cols and nxt not in state.known:
                unknown_adj += 1
        if unknown_adj <= 0:
            continue
        path = memory_shortest_path(sample, state, pos)
        if path is None or len(path) == 0:
            continue
        cand = (len(path), -unknown_adj, pos)
        if best is None or cand < best:
            best = cand
    if best is None:
        return None
    dist, _, pos = best
    return Target(f"frontier-{pos[0]}-{pos[1]}", "explore", pos, 0, dist, 0, 0, -dist, True)

def sample_value(sample: MazeSample, state: PlayerState, pos: Coord) -> int:
    ch = visible_char(sample, state, pos)
    if ch == COIN:
        return COIN_VALUE
    if ch == TRAP:
        return -TRAP_DAMAGE
    return 0


def aggregate_results(results: list[RunResult]) -> list[dict[str, Any]]:
    by: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        by[r.strategy].append(r)
    rows = []
    for strategy, vals in sorted(by.items()):
        n = len(vals)
        rows.append(
            {
                "strategy": strategy,
                "episodes": n,
                "success_rate": sum(x.success for x in vals) / n,
                "avg_remaining_resource": sum(x.final_resource for x in vals) / n,
                "avg_steps": sum(x.total_steps for x in vals) / n,
                "avg_resource_per_step": sum(x.final_score for x in vals) / n,
                "avg_traps": sum(x.trap_count for x in vals) / n,
                "avg_coins": sum(x.coin_count for x in vals) / n,
                "boss_success_rate": sum(x.boss_success for x in vals) / n,
                "avg_boss_rounds": sum(x.boss_rounds for x in vals) / n,
                "avg_runtime_ms": sum(x.runtime_ms for x in vals) / n,
            }
        )
    return rows


def shortest_path(grid: list[str], start: Coord, goal: Coord) -> list[str] | None:
    q = deque([start])
    parent: dict[Coord, tuple[Coord, str]] = {}
    seen = {start}
    while q:
        cur = q.popleft()
        if cur == goal:
            actions: list[str] = []
            while cur != start:
                prev, action = parent[cur]
                actions.append(action)
                cur = prev
            return list(reversed(actions))
        for action, nxt in grid_neighbors_grid(grid, cur):
            if nxt not in seen:
                seen.add(nxt)
                parent[nxt] = (cur, action)
                q.append(nxt)
    return None


def grid_neighbors(sample_grid: list[str], pos: Coord) -> list[tuple[str, Coord]]:
    return list(grid_neighbors_grid(sample_grid, pos))


def grid_neighbors_grid(grid: list[str], pos: Coord):
    rows, cols = len(grid), len(grid[0])
    for action, (dr, dc) in MOVES.items():
        nxt = (pos[0] + dr, pos[1] + dc)
        if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and grid[nxt[0]][nxt[1]] != WALL:
            yield action, nxt


def farthest_cell(grid: list[str], start: Coord) -> Coord:
    q = deque([(start, 0)])
    seen = {start}
    far = start
    far_dist = 0
    while q:
        cur, dist = q.popleft()
        if dist > far_dist:
            far, far_dist = cur, dist
        for _, nxt in grid_neighbors_grid(grid, cur):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, dist + 1))
    return far


def positions_from_actions(start: Coord, actions: list[str]) -> list[Coord]:
    pos = start
    out = [pos]
    for action in actions:
        dr, dc = MOVES[action]
        pos = (pos[0] + dr, pos[1] + dc)
        out.append(pos)
    return out


def place_many(grid: list[list[str]], cells: list[Coord], tile: str, count: int) -> None:
    added = 0
    for r, c in cells:
        if added >= count:
            return
        if grid[r][c] == EMPTY:
            grid[r][c] = tile
            added += 1


def neighbors_of_tiles(grid: list[list[str]], tile: str) -> list[Coord]:
    out: list[Coord] = []
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != tile:
                continue
            for dr, dc in MOVES.values():
                pos = (r + dr, c + dc)
                if 0 <= pos[0] < rows and 0 <= pos[1] < cols and grid[pos[0]][pos[1]] == EMPTY:
                    out.append(pos)
    return out


def find_tiles(grid: list[str], tile: str) -> list[Coord]:
    return [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == tile]


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bucket(value: int | float, cuts: list[int | float]) -> str:
    for cut in cuts:
        if value <= cut:
            return str(cut)
    return "big"


def save_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_samples(path: str | Path) -> list[MazeSample]:
    return [MazeSample.from_dict(x) for x in json.loads(Path(path).read_text(encoding="utf-8"))]

