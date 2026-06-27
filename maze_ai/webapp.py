"""Local one-click web app: import a maze JSON, watch the AI solve it.

Dependency-free (Python standard library only). Run:

    python -m maze_ai.webapp          # opens http://127.0.0.1:8000

Pick a maze file (official format), choose a mode, and the page animates the
real solver's run:
  - 完整探险 (task 2): DP/tree-DP optimal route, boss fight, resource/step score.
  - 3x3 贪心 (task 1): greedy real-time pickup under 3x3 vision (fog of war).
"""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .greedy3x3 import greedy_3x3_run
from .planner import plan_full_exploration
from .simulate import simulate
from .spec import MazeSpec


def solve_payload(data: dict) -> dict:
    mode = data.get("mode", "full")
    spec = MazeSpec.from_dict(data["maze"], name=data.get("name", "maze"))
    if mode == "greedy":
        gr = greedy_3x3_run(spec)
        return {
            "mode": "greedy",
            "frames": gr.frames,
            "summary": {
                "任务": "任务① 3×3 贪心实时拾取",
                "拾取资源价值": gr.picked_value,
                "步数": gr.steps,
                "每步平均拾取": round(gr.avg_per_step, 3),
                "金币": f"{gr.coins_collected}/{len(spec.coins)}",
                "陷阱": f"{gr.traps_triggered}/{len(spec.traps)}",
                "到达终点": gr.reached_exit,
            },
        }
    plan = plan_full_exploration(spec)
    run = simulate(spec, plan.moves)
    bp = plan.boss_plan
    return {
        "mode": "full",
        "frames": run.frames,
        "summary": {
            "任务": "任务② 完整迷宫探险",
            "通关": run.success,
            "剩余资源价值": run.resource,
            "步数": run.steps,
            "得分(资源/步)": round(run.score, 4),
            "金币": f"{run.coins_collected}/{len(spec.coins)}",
            "陷阱": f"{run.traps_triggered}/{len(spec.traps)}",
            "BOSS最少回合": bp.min_rounds,
            "回合上限": spec.min_rounds,
            "复活次数": bp.revives,
            "算法": plan.method,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/solve":
            self._send(404, "not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n).decode("utf-8"))
            result = solve_payload(data)
            self._send(200, json.dumps(result, ensure_ascii=False))
        except Exception as e:  # report parse/solve errors to the page
            self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))


def main(host="127.0.0.1", port=8000, open_browser=True):
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"maze_ai web app running at {url}  (Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        server.shutdown()


PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>迷宫 AI 玩家 — 本地可视化</title>
<style>
 body{font-family:Consolas,Menlo,"Microsoft YaHei",monospace;background:#0f1117;color:#e6e6e6;margin:0;padding:18px}
 h1{font-size:19px;margin:0 0 12px}
 #panel{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
 .card{background:#161a22;border:1px solid #252b36;border-radius:8px;padding:12px 14px}
 #grid{display:grid;gap:1px}
 .cell{width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-size:13px;border-radius:3px}
 .wall{background:#262b36}.floor{background:#1b1f29}.start{background:#2d6cdf;color:#fff}
 .exit{background:#2fae5e;color:#fff}.boss{background:#b5384d;color:#fff}
 .coin{background:#caa23a;color:#1b1f29}.trap{background:#7a3ea8;color:#fff}
 .player{background:#ff5d5d;color:#fff;font-weight:bold}
 .fog{filter:brightness(0.18)}
 button{background:#2d6cdf;border:0;color:#fff;padding:7px 13px;border-radius:5px;cursor:pointer;font-family:inherit}
 button:disabled{opacity:.4;cursor:default}
 label{cursor:pointer;margin-right:12px}
 #bar{display:flex;gap:10px;align-items:center;margin:10px 0;flex-wrap:wrap}
 #summary{white-space:pre;font-size:13px;line-height:1.7}
 #meta{white-space:pre;color:#9aa4b2;font-size:12px;margin-top:6px}
 input[type=range]{width:300px}
 .err{color:#ff8080}
</style></head><body>
<h1>迷宫 AI 玩家 · 本地一键导入 + 过程可视化</h1>
<div id="bar">
 <input type="file" id="file" accept=".json,application/json">
 <label><input type="radio" name="mode" value="full" checked> 完整探险(任务②)</label>
 <label><input type="radio" name="mode" value="greedy"> 3×3 贪心(任务①)</label>
 <button id="run" disabled>运行</button>
 <span id="status" class="err"></span>
</div>
<div id="panel">
 <div class="card">
   <div id="bar2">
     <button id="play" disabled>▶ 播放</button>
     <button id="prev" disabled>◀</button><button id="next" disabled>▶</button>
     <input id="slider" type="range" min="0" value="0" disabled>
     <span id="frameno"></span>
   </div>
   <div id="grid"></div>
   <div id="meta"></div>
 </div>
 <div class="card"><div id="summary">导入一个迷宫 JSON,选择模式,点击“运行”。</div></div>
</div>
<script>
const cls={'#':'wall','@':'player','S':'start','E':'exit','B':'boss','G':'coin','T':'trap','.':'floor',' ':'floor'};
let fileText=null, frames=[], metas=[], mode='full', seen=null, cur=0, timer=null;
const $=id=>document.getElementById(id);

$('file').addEventListener('change', e=>{
  const f=e.target.files[0]; if(!f) return;
  const r=new FileReader();
  r.onload=()=>{ fileText=r.result; $('run').disabled=false; $('status').textContent=''; };
  r.readAsText(f);
});
document.querySelectorAll('input[name=mode]').forEach(el=>el.addEventListener('change',()=>{mode=document.querySelector('input[name=mode]:checked').value;}));

$('run').addEventListener('click', async ()=>{
  if(!fileText) return;
  $('status').textContent='求解中…'; $('status').className='';
  let mazeObj;
  try{ mazeObj=JSON.parse(fileText); }catch(err){ $('status').textContent='JSON 解析失败: '+err; $('status').className='err'; return; }
  try{
    const resp=await fetch('/solve',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({maze:mazeObj, mode:mode})});
    const data=await resp.json();
    if(data.error){ $('status').textContent=data.error; $('status').className='err'; return; }
    frames=data.frames; metas=frames.map(f=>{const m={...f}; delete m.grid; return m;});
    mode=data.mode; buildSeen();
    setupGrid(); enable(true); go(0);
    $('summary').textContent=Object.entries(data.summary).map(([k,v])=>k+': '+v).join('\n');
    $('status').className=''; $('status').textContent='';
  }catch(err){ $('status').textContent='请求失败: '+err; $('status').className='err'; }
});

function buildSeen(){
  if(mode!=='greedy'){ seen=null; return; }
  // accumulate visible 3x3 cells across frames for fog-of-war
  seen=[]; const acc=new Set();
  for(const f of frames){ (f.vision||[]).forEach(([r,c])=>acc.add(r+','+c)); seen.push(new Set(acc)); }
}
function setupGrid(){
  const rows=frames[0].grid.length, cols=frames[0].grid[0].length;
  $('grid').style.gridTemplateColumns=`repeat(${cols},22px)`;
}
function draw(i){
  const g=frames[i].grid; const el=$('grid'); el.innerHTML='';
  const fog = seen? seen[i] : null;
  for(let r=0;r<g.length;r++){ for(let c=0;c<g[r].length;c++){
    const ch=g[r][c]; const d=document.createElement('div');
    let klass='cell '+(cls[ch]||'floor');
    if(fog && !fog.has(r+','+c) && ch!=='@') klass+=' fog';
    d.className=klass; d.textContent=(ch==='.'||ch===' ')?'':ch; el.appendChild(d);
  }}
  $('frameno').textContent=(i+1)+' / '+frames.length;
  $('meta').textContent=JSON.stringify(metas[i]);
}
function go(i){ cur=Math.max(0,Math.min(frames.length-1,i)); $('slider').value=cur; draw(cur); }
function enable(on){ ['play','prev','next','slider'].forEach(id=>$(id).disabled=!on); $('slider').max=frames.length-1; }
$('slider').oninput=()=>go(+$('slider').value);
$('prev').onclick=()=>go(cur-1);
$('next').onclick=()=>go(cur+1);
$('play').onclick=function(){
  if(timer){clearInterval(timer);timer=null;this.textContent='▶ 播放';return;}
  this.textContent='⏸ 暂停';
  timer=setInterval(()=>{ if(cur>=frames.length-1){clearInterval(timer);timer=null;$('play').textContent='▶ 播放';return;} go(cur+1); },110);
};
</script></body></html>"""


if __name__ == "__main__":
    main()
