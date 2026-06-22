from pathlib import Path


p = Path("maze_gpt_agent/agrl_core.py")
s = p.read_text(encoding="utf-8")
s = s.replace(
    '''    if state.boss_defeated:
        return Target("exit", "exit", sample.end, 0, 0, 0, 0, 0, True), "boss_defeated_go_exit", "GO_EXIT"''',
    '''    if state.boss_defeated:
        if state.known.get(sample.end) != EXIT:
            explore = frontier_target(sample, state)
            if explore:
                return explore, "boss_defeated_explore_until_exit_seen", "EXPLORE"
        return Target("exit", "exit", sample.end, 0, 0, 0, 0, 0, True), "boss_defeated_go_exit", "GO_EXIT"''',
)
p.write_text(s, encoding="utf-8")
