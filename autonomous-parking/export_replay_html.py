"""Run the REAL project (ParkingEnv + real controllers) and bake the episodes into ONE html file.

    python export_replay_html.py                       # -> parking_replay.html
    python export_replay_html.py --ppo-only models/ppo_only_best --ppo-shielded models/ppo_shielded_best --n 6

Failures are included on purpose: the file shows what the controllers really do.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from environment.scenarios import Kind
from export_trajectory import build_controller, run_episode


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parking Replay (real project episodes)</title>
<style>
html,body{margin:0;height:100%;background:#0d1117;color:#e6edf3;font:14px system-ui,sans-serif;overflow:hidden}
canvas{position:absolute;inset:0;width:100%;height:100%}
#hud{position:absolute;left:14px;top:14px;background:#161b22ee;border:1px solid #30363d;border-radius:10px;padding:12px 14px;min-width:270px}
#hud h1{margin:0 0 8px;font-size:15px;color:#58a6ff}.r{display:flex;justify-content:space-between;gap:14px;color:#8b949e}.r b{color:#e6edf3;font-weight:500}
#res{margin-top:8px;font-weight:600}.ok{color:#3fb950}.bad{color:#f85149}
#bar{position:absolute;left:14px;right:14px;bottom:14px;background:#161b22ee;border:1px solid #30363d;border-radius:10px;padding:10px 14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
select,button{background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:6px 10px;font:inherit}button{cursor:pointer}
#seek{flex:1;min-width:160px}
</style></head><body>
<canvas id="c"></canvas>
<div id="hud"><h1 id="title"></h1>
<div class="r">controller<b id="h-ctrl"></b></div><div class="r">scenario<b id="h-kind"></b></div>
<div class="r">time<b id="h-t"></b></div><div class="r">speed<b id="h-v"></b></div>
<div class="r">steer action<b id="h-s"></b></div><div class="r">throttle/brake action<b id="h-a"></b></div>
<div class="r">front wheels<b id="h-d"></b></div><div class="r">fuzzy shield<b id="h-sh"></b></div>
<div id="res"></div></div>
<div id="bar"><label>controller <select id="sel-c"></select></label><label>episode <select id="sel-e"></select></label>
<button id="play">Pause</button><button id="restart">Restart</button><input id="seek" type="range" min="0" max="1" value="0">
<label>speed <select id="sel-s"><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select></label></div>
<script>
// Episodes below were produced by running the real Python project (ParkingEnv + the real controllers).
const EPISODES = /*__DATA__*/[];
const cv=document.getElementById('c'),ctx=cv.getContext('2d'),$=id=>document.getElementById(id);
let ep=null,t=0,playing=true,last=performance.now();
const ctrls=[...new Set(EPISODES.map(e=>e.controller))];
$('sel-c').innerHTML=ctrls.map(c=>`<option>${c}</option>`).join('');
function fillEps(){const c=$('sel-c').value;$('sel-e').innerHTML=EPISODES.map((e,i)=>[e,i]).filter(([e])=>e.controller===c)
 .map(([e,i])=>`<option value="${i}">${e.kind} · seed ${e.seed} · ${e.result.success?'success':e.result.collision?'collision':'timeout'}</option>`).join('');load();}
function load(){ep=EPISODES[+$('sel-e').value];t=0;playing=true;$('play').textContent='Pause';$('seek').max=ep.frames.length-1;}
$('sel-c').onchange=fillEps;$('sel-e').onchange=load;
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play'};
$('restart').onclick=()=>{t=0;playing=true;$('play').textContent='Pause'};
$('seek').oninput=e=>{t=+e.target.value;playing=false;$('play').textContent='Play'};
function size(){const d=devicePixelRatio||1;cv.width=innerWidth*d;cv.height=innerHeight*d;}addEventListener('resize',size);size();
function carPath(f,L,W){ // body centre pose
  ctx.save();ctx.translate(f.x,f.y);ctx.rotate(f.theta);
  ctx.fillStyle=f.collision?'#f85149':'#58a6ff';ctx.strokeStyle='#fff';ctx.lineWidth=.06;
  ctx.beginPath();ctx.roundRect(-L/2,-W/2,L,W,.35);ctx.fill();ctx.stroke();
  ctx.fillStyle='#0d1117aa';ctx.fillRect(-L*.22,-W*.36,L*.4,W*.72);
  const wb=2.5,rx=-L/2+(L-wb)/2+0.0,fx=rx+wb,wy=W/2-.05;// wheels
  const rear=-wb/2-0.0,front=wb/2;
  ctx.fillStyle='#111';
  for(const s of[-1,1]){ctx.fillRect(rear-.35,s*wy-.12,.7,.24);
    ctx.save();ctx.translate(front,s*wy);ctx.rotate(f.delta);ctx.fillRect(-.35,-.12,.7,.24);ctx.restore();}
  ctx.fillStyle='#ffe28a';ctx.fillRect(L/2-.15,-W/2+.1,.12,.35);ctx.fillRect(L/2-.15,W/2-.45,.12,.35);
  ctx.restore();}
function draw(){
  const d=devicePixelRatio||1,W=innerWidth,H=innerHeight,[x0,x1,y0,y1]=ep.bounds;
  // focus on the action area, not the whole 30x30 bounds
  let xs=[],ys=[];ep.frames.forEach(f=>{xs.push(f.x);ys.push(f.y)});
  ep.obstacles.forEach(o=>{xs.push(o[0],o[2]);ys.push(o[1],o[3])});xs.push(ep.slot[0]);ys.push(ep.slot[1]);
  const mx=3,vx0=Math.min(...xs)-mx,vx1=Math.max(...xs)+mx,vy0=Math.min(...ys)-mx,vy1=Math.max(...ys)+mx;
  const S=Math.min(W/(vx1-vx0),(H-150)/(vy1-vy0)),ox=W/2-S*(vx0+vx1)/2,oy=(H-70)/2+S*(vy0+vy1)/2+30;
  ctx.setTransform(d,0,0,d,0,0);ctx.fillStyle='#2c3036';ctx.fillRect(0,0,W,H);
  ctx.setTransform(d*S,0,0,-d*S,d*ox,d*oy);
  const L=ep.car.length,Wd=ep.car.width,[sx,sy,sth]=ep.slot;
  ctx.save();ctx.translate(sx,sy);ctx.rotate(sth);ctx.fillStyle='#3fb95033';ctx.fillRect(-L/2,-Wd/2,L,Wd);
  ctx.setLineDash([.25,.2]);ctx.strokeStyle='#3fb950';ctx.lineWidth=.1;ctx.strokeRect(-L/2,-Wd/2,L,Wd);ctx.restore();ctx.setLineDash([]);
  ctx.fillStyle='#6e7681';for(const o of ep.obstacles){ctx.beginPath();ctx.roundRect(o[0],o[1],o[2]-o[0],o[3]-o[1],.2);ctx.fill();}
  const f=ep.frames[Math.min(Math.floor(t),ep.frames.length-1)];
  ctx.strokeStyle='#e3b341';ctx.lineWidth=.08;ctx.beginPath();
  for(let i=0;i<=Math.floor(t)&&i<ep.frames.length;i++){const g=ep.frames[i];i?ctx.lineTo(g.x,g.y):ctx.moveTo(g.x,g.y);}ctx.stroke();
  carPath(f,L,Wd);
  $('title').textContent='Parking replay — real project episode';$('h-ctrl').textContent=ep.controller;$('h-kind').textContent=ep.kind+' (seed '+ep.seed+(ep.reverse?', reverse enabled':'')+')';
  $('h-t').textContent=(Math.floor(t)*ep.dt).toFixed(1)+' s';$('h-v').textContent=f.v.toFixed(2)+' m/s'+(f.v<-.05?' (reverse)':'');
  $('h-s').textContent=f.action[0].toFixed(2);$('h-a').textContent=f.action[1].toFixed(2);$('h-d').textContent=(f.delta*57.3).toFixed(0)+'°';
  $('h-sh').textContent=f.shield?'ACTIVE (overriding policy)':'idle';
  const r=ep.result,end=Math.floor(t)>=ep.frames.length-1;
  $('res').className=r.success?'ok':'bad';$('res').textContent=end?(r.success?`✔ parked · pos err ${r.pos_error.toFixed(2)} m · heading err ${r.heading_error_deg.toFixed(1)}°`:r.collision?'✘ collided':'✘ timed out / did not park'):'';
  $('seek').value=Math.floor(t);}
function loop(now){const dt=Math.max(0,(now-last)/1000);last=now;if(playing){t+=dt/ep.dt*(+$('sel-s').value);if(t>=ep.frames.length-1){t=ep.frames.length-1;playing=false;$('play').textContent='Play';}}draw();requestAnimationFrame(loop);}
if(EPISODES.length){fillEps();requestAnimationFrame(loop);}else document.body.innerHTML='<p style="padding:20px">No episodes embedded. Run export_replay_html.py.</p>';
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ppo-only", default="models/ppo_only_best")
    p.add_argument("--ppo-shielded", default="models/ppo_shielded_best")
    p.add_argument("--n", type=int, default=5, help="episodes per controller")
    p.add_argument("--out", default="parking_replay.html")
    a = p.parse_args()
    def pick(path):
        return path if Path(path + ".zip").exists() else path.replace("_best", "")
    po, ps = pick(a.ppo_only), pick(a.ppo_shielded)
    print("using models:", po, ps)
    plan = [("rule_based", "", Kind.PERPENDICULAR, False), ("fuzzy_only", "", Kind.PERPENDICULAR, False),
            ("ppo_only", po, Kind.PERPENDICULAR, False), ("ppo_fuzzy_bolted", po, Kind.PERPENDICULAR, False),
            ("ppo_shielded", ps, Kind.PERPENDICULAR, False), ("parallel_geometric", "", Kind.PARALLEL, True)]
    eps = []
    for name, model, kind, rev in plan:
        if model and not Path(model + ".zip").exists():
            print(f"{name:20s} SKIPPED (no model file {model}.zip - train one or copy your models/ folder in)")
            continue
        ad = build_controller(name, model)
        wins = 0
        for s in range(1000, 1000 + a.n):
            e = run_episode(ad, s, kind, name, rev)
            wins += e["result"]["success"]; eps.append(e)
        print(f"{name:20s} {wins}/{a.n} parked")
    tpl = TEMPLATE
    Path(a.out).write_text(tpl.replace("/*__DATA__*/[]", json.dumps(eps, separators=(",", ":"))))
    print("wrote", Path(a.out).resolve())

if __name__ == "__main__":
    main()
