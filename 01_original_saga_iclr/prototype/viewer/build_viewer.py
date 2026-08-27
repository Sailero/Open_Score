"""生成自包含的 HTML 回放前端: 把 outputs/*.json 回放数据内嵌进单个 HTML 文件.

双击 replay.html 即可在浏览器查看 (无需服务器).
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
VIEWER_DIR = os.path.join(ROOT, "viewer")

TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>SwarmBattle 回放器 — SAGA demo</title>
<style>
  :root { --bg:#0f1220; --panel:#181c2f; --text:#e8eaf2; --dim:#8a90a8; --accent:#5b8cff; }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--bg); color: var(--text);
         font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
         display: flex; flex-direction: column; align-items: center;
         min-height: 100vh; padding: 24px; }
  h1 { font-size: 20px; font-weight: 600; margin-bottom: 4px; }
  .sub { color: var(--dim); font-size: 13px; margin-bottom: 16px; }
  .panel { background: var(--panel); border-radius: 12px; padding: 16px;
           box-shadow: 0 4px 24px rgba(0,0,0,.4); }
  canvas { background: #0a0d1a; border-radius: 8px; display: block; }
  .controls { display: flex; gap: 10px; align-items: center; margin-top: 12px;
              flex-wrap: wrap; }
  button, select { background: #232945; color: var(--text); border: 1px solid #333a5e;
           border-radius: 6px; padding: 6px 14px; cursor: pointer; font-size: 13px; }
  button:hover { background: var(--accent); }
  input[type=range] { flex: 1; min-width: 200px; accent-color: var(--accent); }
  .stat { color: var(--dim); font-size: 13px; min-width: 150px; }
  .legend { display:flex; gap:16px; margin-top:8px; font-size:12px; color:var(--dim);}
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%;
         margin-right:4px; vertical-align:-1px;}
</style>
</head>
<body>
  <h1>SwarmBattle 回放器 <span style="color:var(--accent)">SAGA demo</span></h1>
  <div class="sub">红 = SAGA 智能体 &nbsp;|&nbsp; 蓝 = 脚本对手 (greedy chase) &nbsp;|&nbsp; 圆点大小 = 血量</div>
  <div class="panel">
    <canvas id="cv" width="640" height="640"></canvas>
    <div class="controls">
      <select id="replaySel"></select>
      <button id="playBtn">▶ 播放</button>
      <input type="range" id="seek" min="0" max="0" value="0">
      <span class="stat" id="stat"></span>
    </div>
    <div class="legend">
      <span><span class="dot" style="background:#ff5b5b"></span>红方 (受控)</span>
      <span><span class="dot" style="background:#5b8cff"></span>蓝方 (脚本)</span>
      <span><span class="dot" style="background:#3a3f5c"></span>阵亡</span>
    </div>
  </div>
<script>
const REPLAYS = __REPLAYS__;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const seek = document.getElementById('seek'), stat = document.getElementById('stat');
const sel = document.getElementById('replaySel'), playBtn = document.getElementById('playBtn');
let data = null, frame = 0, playing = false, timer = null;

Object.keys(REPLAYS).forEach(k => {
  const o = document.createElement('option'); o.value = k; o.textContent = k;
  sel.appendChild(o);
});

function load(name) {
  data = REPLAYS[name]; frame = 0;
  seek.max = data.frames.length - 1; seek.value = 0;
  draw();
}
function world2px(p, arena) {
  const s = cv.width / arena;
  return [(p[0] + arena/2) * s, cv.height - (p[1] + arena/2) * s];
}
function draw() {
  if (!data) return;
  const f = data.frames[frame], arena = f.arena;
  ctx.fillStyle = '#0a0d1a'; ctx.fillRect(0, 0, cv.width, cv.height);
  // 轨迹尾迹
  const tail = 25;
  for (let dt = tail; dt >= 1; dt--) {
    const g = data.frames[frame - dt]; if (!g) continue;
    const a = 0.25 * (1 - dt / tail);
    for (const [team, color] of [['red','255,91,91'], ['blue','91,140,255']]) {
      g[team].pos.forEach((p, i) => {
        if (!g[team].alive[i]) return;
        const [x, y] = world2px(p, arena);
        ctx.fillStyle = `rgba(${color},${a})`;
        ctx.beginPath(); ctx.arc(x, y, 2, 0, 7); ctx.fill();
      });
    }
  }
  // 当前帧单位
  for (const [team, color] of [['red','#ff5b5b'], ['blue','#5b8cff']]) {
    f[team].pos.forEach((p, i) => {
      const [x, y] = world2px(p, arena);
      const alive = f[team].alive[i];
      const hp = f[team].hp[i], h = f[team].heading[i];
      ctx.fillStyle = alive ? color : '#3a3f5c';
      const r = alive ? 4 + 4 * Math.max(hp, 0) : 3;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
      if (alive) {  // 朝向线
        ctx.strokeStyle = color; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(x, y);
        ctx.lineTo(x + Math.cos(h) * r * 2.2, y - Math.sin(h) * r * 2.2); ctx.stroke();
      }
    });
  }
  const nr = f.red.alive.filter(Boolean).length, nb = f.blue.alive.filter(Boolean).length;
  stat.textContent = `t=${f.t}  红 ${nr} : ${nb} 蓝` +
    (frame == data.frames.length-1 ? (data.result.win ? '  ✔ 红胜' : '  ✘ 红负') : '');
  seek.value = frame;
}
function tick() {
  if (!playing) return;
  frame = Math.min(frame + 1, data.frames.length - 1);
  draw();
  if (frame >= data.frames.length - 1) { playing = false; playBtn.textContent = '▶ 播放'; return; }
  timer = setTimeout(tick, 50);
}
playBtn.onclick = () => {
  playing = !playing;
  playBtn.textContent = playing ? '⏸ 暂停' : '▶ 播放';
  if (playing) { if (frame >= data.frames.length - 1) frame = 0; tick(); }
  else clearTimeout(timer);
};
seek.oninput = () => { frame = +seek.value; draw(); };
sel.onchange = () => load(sel.value);
load(Object.keys(REPLAYS)[0]);
</script>
</body>
</html>
"""


def build():
    replays = {}
    labels = {
        "replay_before.json": "训练前 6v6 (随机策略)",
        "replay_after.json": "训练后 6v6",
        "replay_after_24v24.json": "训练后 24v24 (零样本规模迁移)",
    }
    for fname, label in labels.items():
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                replays[label] = json.load(f)
    if not replays:
        print("[viewer] outputs/ 下没有回放 JSON, 先运行 train_demo.py")
        return
    html = TEMPLATE.replace("__REPLAYS__", json.dumps(replays))
    out = os.path.join(VIEWER_DIR, "replay.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[viewer] 已生成 {out} ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    build()
