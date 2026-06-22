from pathlib import Path


p = Path("maze_gpt_agent/agrl_core.py")
s = p.read_text(encoding="utf-8")
s = s.replace(
    '    state = PlayerState(position=sample.start, path_history=[sample.start])\n    frames = [frame(sample, state, "START", "start", None)]',
    '    state = PlayerState(position=sample.start, path_history=[sample.start])\n    observe_3x3(sample, state)\n    frames = [frame(sample, state, "START", "start", None)]',
)
s = s.replace(
    "        state = PlayerState(position=sample.start, path_history=[sample.start])\n        for _ in range(sample.rows * sample.cols):",
    "        state = PlayerState(position=sample.start, path_history=[sample.start])\n        observe_3x3(sample, state)\n        for _ in range(sample.rows * sample.cols):",
)
s = s.replace(
    "    for coin in sample.coins:\n        if coin in state.collected_coins:",
    "    for coin in known_positions(state, COIN):\n        if coin in state.collected_coins:",
)
s = s.replace(
    "        for action, nxt in grid_neighbors(sample.grid, pos):",
    "        for action, nxt in memory_neighbors(sample, state, pos, goal):",
)
s = s.replace(
    '    targets = pctsp_targets(sample, state)\n    if action == "NEAREST_GOLD":',
    '    targets = pctsp_targets(sample, state)\n    if action == "EXPLORE":\n        return frontier_target(sample, state)\n    if action == "NEAREST_GOLD":',
)
s = s.replace(
    '    if action == "GO_BOSS":\n        return Target("boss", "boss", sample.boss, 0, 0, 0, 0, 0, True)\n    if action == "GO_EXIT":\n        return Target("exit", "exit", sample.end, 0, 0, 0, 0, 0, state.boss_defeated)',
    '    if action == "GO_BOSS":\n        return Target("boss", "boss", sample.boss, 0, 0, 0, 0, 0, True) if state.known.get(sample.boss) == BOSS else frontier_target(sample, state)\n    if action == "GO_EXIT":\n        return Target("exit", "exit", sample.end, 0, 0, 0, 0, 0, state.boss_defeated) if state.known.get(sample.end) == EXIT else frontier_target(sample, state)',
)
s = s.replace(
    '    if action == "EXPLORE":\n        return targets[-1] if targets else None\n    return None',
    "    return None",
)
s = s.replace(
    "    nearest_gold = min((len(shortest_path(sample.grid, state.position, c) or []) for c in sample.coins if c not in state.collected_coins), default=99)\n    boss_dist = len(shortest_path(sample.grid, state.position, sample.boss) or [])\n    exit_dist = len(shortest_path(sample.grid, state.position, sample.end) or [])",
    "    nearest_gold = min((len(memory_shortest_path(sample, state, c) or []) for c in known_positions(state, COIN) if c not in state.collected_coins), default=99)\n    boss_dist = len(memory_shortest_path(sample, state, sample.boss) or []) if state.known.get(sample.boss) == BOSS else 99\n    exit_dist = len(memory_shortest_path(sample, state, sample.end) or []) if state.known.get(sample.end) == EXIT else 99",
)
s = s.replace(
    "    elif ch == TRAP and pos not in state.triggered_traps:\n        state.triggered_traps.add(pos)\n        state.resource -= TRAP_DAMAGE\n",
    "    elif ch == TRAP and pos not in state.triggered_traps:\n        state.triggered_traps.add(pos)\n        state.resource -= TRAP_DAMAGE\n    observe_3x3(sample, state)\n",
)
s = s.replace(
    "    if state.resource >= sample.boss_config.revive_cost:\n        boss_path = rcspp_path(sample, state, sample.boss, require_boss_resource=True)\n        if boss_path.feasible:",
    "    if state.resource >= sample.boss_config.revive_cost and state.known.get(sample.boss) == BOSS:\n        boss_path = rcspp_path(sample, state, sample.boss, require_boss_resource=True)\n        if boss_path.feasible:",
)
s = s.replace(
    '    return Target("boss", "boss", sample.boss, 0, 0, 0, 0, 0, True), "no_gold_left_go_boss", "GO_BOSS"',
    '    explore = frontier_target(sample, state)\n    if explore:\n        return explore, "explore_unknown_frontier", "EXPLORE"\n    return Target("boss", "boss", sample.boss, 0, 0, 0, 0, 0, True), "memory_exhausted_go_boss", "GO_BOSS"',
)
insert = '''

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
    best: tuple[int, Coord] | None = None
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
        if path is None:
            continue
        cand = (len(path), pos)
        if best is None or cand < best:
            best = cand
    if best is None:
        return None
    dist, pos = best
    return Target(f"frontier-{pos[0]}-{pos[1]}", "explore", pos, 0, dist, 0, 0, -dist, True)
'''
s = s.replace("\ndef sample_value(sample: MazeSample, state: PlayerState, pos: Coord) -> int:", insert + "\ndef sample_value(sample: MazeSample, state: PlayerState, pos: Coord) -> int:")
p.write_text(s, encoding="utf-8")
