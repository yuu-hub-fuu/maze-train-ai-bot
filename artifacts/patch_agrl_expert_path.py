from pathlib import Path


core = Path("maze_gpt_agent/agrl_core.py")
s = core.read_text(encoding="utf-8")
s = s.replace(
    '        "action": action,\n        "event": event,',
    '        "action": action,\n        "event": event,\n        "pos": list(state.position),',
)
core.write_text(s, encoding="utf-8")

gen = Path("scripts/agrl_generate_dataset.py")
s = gen.read_text(encoding="utf-8")
s = s.replace(
    '"recommended_path": [list(x) for x in expert.frames[-1].get("path", [])] if expert.frames else [],',
    '"recommended_path": [f.get("pos") for f in expert.frames if f.get("pos") is not None],',
)
gen.write_text(s, encoding="utf-8")
