"""ASCII + HTML visualization of a run (frames produced by simulate/greedy)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

_TILE_CLASS = {
    "#": "wall", "@": "player", "S": "start", "E": "exit",
    "B": "boss", "G": "coin", "T": "trap", ".": "floor", " ": "floor",
}


def ascii_frame(grid: list[str]) -> str:
    return "\n".join(grid)


def render_html(title: str, frames: list[dict[str, Any]], summary: dict[str, Any], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    grids = [f["grid"] for f in frames]
    metas = [
        {k: v for k, v in f.items() if k != "grid"}
        for f in frames
    ]
    payload = {"grids": grids, "metas": metas, "summary": summary}

    rows = len(grids[0]) if grids else 0
    cols = len(grids[0][0]) if grids and grids[0] else 0

    doc = _TEMPLATE.replace("__TITLE__", html.escape(title))
    doc = doc.replace("__ROWS__", str(rows)).replace("__COLS__", str(cols))
    doc = doc.replace("__SUMMARY__", html.escape(json.dumps(summary, ensure_ascii=False, indent=2)))
    doc = doc.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    out.write_text(doc, encoding="utf-8")
    return out


_TEMPLATE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{font-family:Consolas,Menlo,monospace;background:#0f1117;color:#e6e6e6;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 8px}
 #grid{display:grid;grid-template-columns:repeat(__COLS__,22px);grid-auto-rows:22px;gap:1px;margin:12px 0}
 .cell{display:flex;align-items:center;justify-content:center;font-size:13px;border-radius:3px}
 .wall{background:#262b36}.floor{background:#1b1f29}.start{background:#2d6cdf;color:#fff}
 .exit{background:#2fae5e;color:#fff}.boss{background:#b5384d;color:#fff}
 .coin{background:#caa23a;color:#1b1f29}.trap{background:#7a3ea8;color:#fff}
 .player{background:#ff5d5d;color:#fff;font-weight:bold}
 #bar{display:flex;gap:10px;align-items:center;margin:8px 0}
 button{background:#2d6cdf;border:0;color:#fff;padding:6px 12px;border-radius:5px;cursor:pointer}
 #meta{white-space:pre;background:#161a22;padding:8px;border-radius:6px;min-height:20px}
 pre{background:#161a22;padding:10px;border-radius:6px;overflow:auto}
 input[type=range]{width:360px}
</style></head><body>
<h1>__TITLE__</h1>
<div id="bar">
 <button id="play">▶ 播放</button>
 <button id="prev">◀</button><button id="next">▶</button>
 <input id="slider" type="range" min="0" value="0">
 <span id="frameno"></span>
</div>
<div id="grid"></div>
<div id="meta"></div>
<h3>结果</h3>
<pre>__SUMMARY__</pre>
<script>
const DATA = __DATA__;
const grids = DATA.grids, metas = DATA.metas;
const gridEl = document.getElementById('grid');
const slider = document.getElementById('slider');
const frameno = document.getElementById('frameno');
const metaEl = document.getElementById('meta');
slider.max = grids.length - 1;
const cls = {'#':'wall','@':'player','S':'start','E':'exit','B':'boss','G':'coin','T':'trap','.':'floor',' ':'floor'};
function draw(i){
  const g = grids[i]; gridEl.innerHTML='';
  for(const row of g){ for(const ch of row){
    const d=document.createElement('div'); d.className='cell '+(cls[ch]||'floor');
    d.textContent = ch==='.'||ch===' '?'':ch; gridEl.appendChild(d);
  }}
  frameno.textContent = (i+1)+' / '+grids.length;
  metaEl.textContent = JSON.stringify(metas[i]);
}
let cur=0, timer=null;
function go(i){ cur=Math.max(0,Math.min(grids.length-1,i)); slider.value=cur; draw(cur); }
slider.oninput=()=>go(+slider.value);
document.getElementById('prev').onclick=()=>go(cur-1);
document.getElementById('next').onclick=()=>go(cur+1);
document.getElementById('play').onclick=function(){
  if(timer){clearInterval(timer);timer=null;this.textContent='▶ 播放';return;}
  this.textContent='⏸ 暂停';
  timer=setInterval(()=>{ if(cur>=grids.length-1){clearInterval(timer);timer=null;document.getElementById('play').textContent='▶ 播放';return;} go(cur+1); },120);
};
draw(0);
</script></body></html>"""
