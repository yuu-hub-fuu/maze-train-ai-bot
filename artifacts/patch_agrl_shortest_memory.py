from pathlib import Path


p = Path("maze_gpt_agent/agrl_core.py")
s = p.read_text(encoding="utf-8")
s = s.replace(
    '''    if strategy == "shortest":
        if not state.boss_defeated:
            return Target("boss", "boss", sample.boss, 0, 0, 0, 0, 0, True), "shortest_to_boss", "GO_BOSS"
        return Target("exit", "exit", sample.end, 0, 0, 0, 0, 0, True), "shortest_to_exit", "GO_EXIT"''',
    '''    if strategy == "shortest":
        if not state.boss_defeated:
            if state.known.get(sample.boss) != BOSS:
                explore = frontier_target(sample, state)
                return explore, "shortest_memory_explore_until_boss_seen", "EXPLORE"
            return Target("boss", "boss", sample.boss, 0, 0, 0, 0, 0, True), "shortest_to_boss", "GO_BOSS"
        if state.known.get(sample.end) != EXIT:
            explore = frontier_target(sample, state)
            return explore, "shortest_memory_explore_until_exit_seen", "EXPLORE"
        return Target("exit", "exit", sample.end, 0, 0, 0, 0, 0, True), "shortest_to_exit", "GO_EXIT"''',
)
p.write_text(s, encoding="utf-8")
