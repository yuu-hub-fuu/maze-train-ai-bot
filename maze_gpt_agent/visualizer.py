from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def render_run_html(path: str | Path, title: str, frames: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"frames": frames, "summary": summary}, ensure_ascii=False)
    Path(path).write_text(HTML_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__PAYLOAD__", payload), encoding="utf-8")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f8fafc;
      --ink: #111827;
      --muted: #64748b;
      --line: #d8e0ea;
      --wall: #1f2937;
      --path: #ffffff;
      --coin: #f6c453;
      --trap: #f97373;
      --boss: #8b5cf6;
      --exit: #22c55e;
      --agent: #0ea5e9;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, "Segoe UI", Arial, sans-serif; background: var(--bg); color: var(--ink); }
    main { max-width: 1120px; margin: 0 auto; padding: 28px 20px 44px; }
    header { display: flex; justify-content: space-between; gap: 20px; align-items: end; border-bottom: 1px solid var(--line); padding-bottom: 18px; }
    h1 { margin: 0; font-size: 28px; line-height: 1.15; }
    .subtitle { color: var(--muted); margin-top: 8px; }
    .stats { display: grid; grid-template-columns: repeat(4, minmax(90px, 1fr)); gap: 10px; margin: 22px 0; }
    .stat { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
    .stat b { display: block; font-size: 20px; margin-top: 4px; }
    .stage { display: grid; grid-template-columns: minmax(320px, 520px) 1fr; gap: 22px; align-items: start; }
    .board { display: grid; gap: 2px; background: var(--line); border: 1px solid var(--line); padding: 2px; width: fit-content; max-width: 100%; }
    .cell { width: 28px; height: 28px; display: grid; place-items: center; font-size: 14px; font-weight: 700; }
    .wall { background: var(--wall); }
    .empty { background: var(--path); }
    .coin { background: var(--coin); color: #7c4a03; }
    .trap { background: var(--trap); color: #7f1d1d; }
    .boss { background: var(--boss); color: #fff; }
    .exit { background: var(--exit); color: #064e3b; }
    .start { background: #e0f2fe; color: #075985; }
    .agent { background: var(--agent); color: white; border-radius: 50%; }
    .panel { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
    .controls { display: flex; gap: 10px; align-items: center; margin-bottom: 14px; }
    button { border: 1px solid var(--line); background: #fff; border-radius: 6px; padding: 8px 12px; cursor: pointer; font-weight: 600; }
    input[type=range] { flex: 1; }
    pre { background: #f1f5f9; border-radius: 8px; padding: 12px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid var(--line); }
    @media (max-width: 820px) {
      header, .stage { display: block; }
      .stats { grid-template-columns: repeat(2, 1fr); }
      .cell { width: 22px; height: 22px; font-size: 12px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>__TITLE__</h1>
        <div class="subtitle">模型单步决策，环境安全执行；指标对齐任务书的剩余资源价值、步数和二者比值。</div>
      </div>
    </header>
    <section class="stats" id="stats"></section>
    <section class="stage">
      <div class="board" id="board"></div>
      <aside class="panel">
        <div class="controls">
          <button id="prev">上一帧</button>
          <button id="play">播放</button>
          <button id="next">下一帧</button>
          <input id="scrub" type="range" min="0" value="0" />
        </div>
        <div id="frameInfo"></div>
        <pre id="gridText"></pre>
      </aside>
    </section>
  </main>
  <script>
    const data = __PAYLOAD__;
    const frames = data.frames;
    let idx = 0;
    let timer = null;
    const board = document.getElementById('board');
    const scrub = document.getElementById('scrub');
    scrub.max = Math.max(0, frames.length - 1);
    document.getElementById('stats').innerHTML = Object.entries(data.summary)
      .filter(([k]) => ['agent','success','boss_clear','gold','steps','score','trap_count'].includes(k))
      .map(([k,v]) => `<div class="stat"><span>${k}</span><b>${typeof v === 'number' ? Number(v).toFixed(3).replace(/\.000$/,'') : v}</b></div>`)
      .join('');
    function cls(ch) {
      return ch === '#' ? 'wall' : ch === 'G' ? 'coin' : ch === 'T' ? 'trap' : ch === 'B' ? 'boss' : ch === 'E' ? 'exit' : ch === 'S' ? 'start' : ch === '@' ? 'agent' : 'empty';
    }
    function label(ch) {
      return ch === '#' || ch === '.' ? '' : ch;
    }
    function render() {
      const f = frames[idx];
      const rows = f.grid;
      board.style.gridTemplateColumns = `repeat(${rows[0].length}, 1fr)`;
      board.innerHTML = rows.flatMap(row => [...row].map(ch => `<div class="cell ${cls(ch)}">${label(ch)}</div>`)).join('');
      document.getElementById('frameInfo').innerHTML = `<table><tbody>
        <tr><th>frame</th><td>${idx + 1}/${frames.length}</td></tr>
        <tr><th>action</th><td>${f.action ?? 'START'}</td></tr>
        <tr><th>event</th><td>${f.event}</td></tr>
        <tr><th>gold</th><td>${f.gold}</td></tr>
        <tr><th>steps</th><td>${f.steps}</td></tr>
        <tr><th>score</th><td>${Number(f.score).toFixed(3)}</td></tr>
      </tbody></table>`;
      document.getElementById('gridText').textContent = rows.join('\n');
      scrub.value = idx;
    }
    document.getElementById('prev').onclick = () => { idx = Math.max(0, idx - 1); render(); };
    document.getElementById('next').onclick = () => { idx = Math.min(frames.length - 1, idx + 1); render(); };
    scrub.oninput = () => { idx = Number(scrub.value); render(); };
    document.getElementById('play').onclick = () => {
      if (timer) { clearInterval(timer); timer = null; return; }
      timer = setInterval(() => { idx = (idx + 1) % frames.length; render(); }, 350);
    };
    render();
  </script>
</body>
</html>
"""
