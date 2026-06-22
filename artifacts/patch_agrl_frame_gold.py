from pathlib import Path


p = Path("maze_gpt_agent/agrl_core.py")
s = p.read_text(encoding="utf-8")
s = s.replace(
    '        "resource": state.resource,\n        "steps": state.steps,',
    '        "resource": state.resource,\n        "gold": state.resource,\n        "steps": state.steps,',
)
p.write_text(s, encoding="utf-8")
