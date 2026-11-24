#!/usr/bin/env python3
"""生成可拖动交互的 3D 潜空间浏览器（单文件 HTML，零依赖，数据内嵌）

输入：routeA_task_latent.npz（scripts/routeA_task_latent.py 在 6c 生成）
输出：docs/assets/latent_explorer.html（双击即可打开，不需要服务器、不需要联网）

内容：把 16 维潜变量 PCA 到 3 维，画成可旋转/缩放的点云；
     可按子任务类别 / 相位角 / 时间 / 不典型度着色；
     可选中单条 episode 看它的执行轨迹（按时间渐变）。
"""
import argparse
import base64
import json
import os

import numpy as np


def pca3(X):
    Xc = X - X.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    return Xc @ Vt[:3].T, ev[:3]


def b64(arr, dtype):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", default="/tmp/routeA_task_latent.npz")
    ap.add_argument("--out", default="docs/assets/latent_explorer.html")
    ap.add_argument("--title", default="收纳耳机与笔记本电脑 · 整任务 VAE 潜空间（PCA 3 维）")
    a = ap.parse_args()

    z = np.load(a.latent, allow_pickle=True)
    MU, Y, EP, FIDX, SCORE = z["mu"], z["y"], z["ep"], z["fidx"], z["score"]
    Z3, ev = pca3(MU)
    N = len(Z3)

    # 按 (episode, 帧号) 排序，使每条 episode 在数组里连续 —— 轨迹绘制与筛选都简单
    order = np.lexsort((FIDX, EP))
    Z3, Y, EP, FIDX, SCORE = Z3[order], Y[order], EP[order], FIDX[order], SCORE[order]

    eps = np.unique(EP)
    ep_remap = {int(e): i for i, e in enumerate(eps)}
    EPI = np.array([ep_remap[int(e)] for e in EP], np.int32)
    OFF = np.zeros(len(eps) + 1, np.int32)
    for i, c in enumerate(np.bincount(EPI, minlength=len(eps))):
        OFF[i + 1] = OFF[i] + c

    # 相位角（以 PC1-PC2 的中位点为圆心）；与 ROUTE_A_LATENT_PCA.md 的口径一致
    center = np.median(Z3[:, :2], axis=0)
    ang = np.arctan2(Z3[:, 1] - center[1], Z3[:, 0] - center[0])

    meta = dict(
        title=a.title,
        n=int(N), n_ep=int(len(eps)),
        ep_names=[f"ep{int(e):06d}" for e in eps],
        pc_var=[round(float(v) * 100, 2) for v in ev],
        center=[float(center[0]), float(center[1])],
        t_max=float(FIDX.max() / 25.0),
        cls_counts={int(c): int((Y == c).sum()) for c in np.unique(Y)},
        fps=25,
    )

    DATA = dict(
        pc=b64(Z3, np.float32),
        cls=b64(Y, np.uint8),
        ep=b64(EPI, np.int32),
        off=b64(OFF, np.int32),
        t=b64(FIDX.astype(np.float32) / 25.0, np.float32),
        score=b64(SCORE, np.float32),
        ang=b64(ang, np.float32),
    )

    html = TEMPLATE.replace("__META__", json.dumps(meta, ensure_ascii=False)) \
                   .replace("__DATA__", json.dumps(DATA))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(html)
    print(f"写出 {a.out}  ({os.path.getsize(a.out)/1024:.0f} KB)")
    print(f"点数 {N}；episode {len(eps)}；解释方差 {meta['pc_var']}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0b0d12; color:#d8dee9; font:13px/1.5 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC","Microsoft YaHei",sans-serif; overflow:hidden; }
  #wrap { display:flex; height:100vh; }
  #side { width:290px; flex:0 0 290px; background:#11141b; border-right:1px solid #202533; padding:14px; overflow-y:auto; }
  #stage { position:relative; flex:1; }
  canvas { display:block; width:100%; height:100%; cursor:grab; }
  canvas.dragging { cursor:grabbing; }
  h1 { font-size:13px; margin:0 0 4px; font-weight:600; letter-spacing:.2px; }
  .sub { color:#7c8698; font-size:11px; margin-bottom:12px; }
  .grp { margin:12px 0; padding-top:10px; border-top:1px solid #1d2230; }
  .grp:first-of-type { border-top:none; }
  .lbl { color:#8b95a8; font-size:11px; text-transform:uppercase; letter-spacing:.6px; margin-bottom:6px; }
  label.row { display:flex; align-items:center; gap:7px; padding:2px 0; cursor:pointer; }
  label.row:hover { background:#161a24; }
  .btnrow { display:grid; grid-template-columns:1fr 1fr; gap:5px; }
  .btnrow button { width:auto; padding:5px 6px; font-size:11.5px; }
  .slider { display:flex; align-items:center; gap:8px; padding:3px 0; }
  .slider span { color:#8b95a8; font-size:11.5px; white-space:nowrap; flex:0 0 46px; }
  .slider input { flex:1; min-width:0; }
  .sw { width:11px; height:11px; border-radius:2px; flex:0 0 11px; }
  .cnt { color:#5f6a7d; margin-left:auto; font-variant-numeric:tabular-nums; }
  select, input[type=range] { width:100%; }
  select { background:#1a1f2b; color:#d8dee9; border:1px solid #2a3142; border-radius:4px; padding:4px; }
  button { background:#1a1f2b; color:#cfd6e2; border:1px solid #2a3142; border-radius:4px; padding:5px 9px; cursor:pointer; width:100%; }
  button:hover { background:#222838; }
  .kv { display:flex; justify-content:space-between; font-size:11px; color:#8b95a8; padding:1px 0; }
  .kv b { color:#cfd6e2; font-weight:600; font-variant-numeric:tabular-nums; }
  #tip { position:absolute; pointer-events:none; background:#141922ee; border:1px solid #2a3142; border-radius:5px;
         padding:6px 8px; font-size:11px; line-height:1.45; display:none; white-space:nowrap; }
  #legend { position:absolute; right:12px; bottom:12px; background:#11141bdd; border:1px solid #202533;
            border-radius:6px; padding:8px 10px; font-size:10.5px; color:#8b95a8; }
  #legend .bar { height:8px; border-radius:2px; margin:3px 0 2px; }
  .hint { position:absolute; left:12px; bottom:12px; color:#4e5766; font-size:11px; }
</style>
</head>
<body>
<div id="wrap">
  <div id="side">
    <h1 id="ttl"></h1>
    <div class="sub" id="sub"></div>

    <div class="grp">
      <div class="lbl">着色方式</div>
      <select id="cmode">
        <option value="cls">子任务类别</option>
        <option value="ang">相位角（绕环角度）</option>
        <option value="t">时间</option>
        <option value="score">不典型度（越大越不像）</option>
        <option value="ep">episode</option>
      </select>
    </div>

    <div class="grp" id="clsbox">
      <div class="lbl">子任务（点击显示/隐藏）</div>
      <div id="clss"></div>
    </div>

    <div class="grp">
      <div class="lbl">轨迹</div>
      <select id="epsel"><option value="-1">（不显示轨迹）</option></select>
      <label class="row" style="margin-top:6px"><input type="checkbox" id="dim"> 淡化其它点</label>
    </div>

    <div class="grp">
      <div class="lbl">视角预设</div>
      <div class="btnrow">
        <button data-preset="iso">等轴测</button>
        <button data-preset="p12">看 PC1-PC2</button>
        <button data-preset="p23">看 PC2-PC3</button>
        <button data-preset="p13">看 PC1-PC3</button>
      </div>
      <label class="row" style="margin-top:8px"><input type="checkbox" id="spin"> 自动旋转</label>
    </div>

    <div class="grp">
      <div class="lbl">显示</div>
      <div class="slider"><span>点大小</span><input type="range" id="psize" min="1" max="6" step="0.5" value="2.5"></div>
      <div class="slider"><span>透视</span><input type="range" id="fov" min="150" max="1200" step="10" value="520"></div>
      <button id="reset" style="margin-top:8px">重置视角</button>
    </div>

    <div class="grp">
      <div class="lbl">信息</div>
      <div class="kv"><span>点数</span><b id="k_n"></b></div>
      <div class="kv"><span>episode</span><b id="k_ep"></b></div>
      <div class="kv"><span>PC1 / PC2 / PC3</span><b id="k_var"></b></div>
      <div class="kv"><span>显示中</span><b id="k_shown"></b></div>
    </div>
  </div>

  <div id="stage">
    <canvas id="cv"></canvas>
    <div id="tip"></div>
    <div id="legend"></div>
    <div class="hint">拖动旋转 · 滚轮缩放 · 右键/Shift 拖动平移 · 双击重置</div>
  </div>
</div>

<script>
const META = __META__;
const DATA = __DATA__;

function dec(b64, Type) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Type(u8.buffer);
}
const PC = dec(DATA.pc, Float32Array);
const CLS = dec(DATA.cls, Uint8Array);
const EPI = dec(DATA.ep, Int32Array);
const OFF = dec(DATA.off, Int32Array);
const TT = dec(DATA.t, Float32Array);
const SC = dec(DATA.score, Float32Array);
const ANG = dec(DATA.ang, Float32Array);
const N = META.n;

const PAL = ['#4c9be8','#f0932b','#2ecc71','#e74c3c','#9b59b6','#1abc9c','#e67e22','#95a5a6',
             '#3498db','#e84393','#00b894','#fdcb6e','#6c5ce7','#d63031','#00cec9','#b2bec3'];
const clsList = Object.keys(META.cls_counts).map(Number).sort((a,b)=>a-b);
const clsColor = {}; clsList.forEach((c,i)=> clsColor[c] = PAL[i % PAL.length]);
const clsVisible = {}; clsList.forEach(c => clsVisible[c] = true);

// ---------- 相机 ----------
let yaw = -0.6, pitch = 0.5, zoom = 1, panX = 0, panY = 0, fov = 520;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
let W = 0, H = 0, DPR = 1;

// 统一缩放到 [-1,1] 量级
let ext = 0;
for (let i = 0; i < 3 * N; i++) ext = Math.max(ext, Math.abs(PC[i]));
ext = ext || 1;

const rot = new Float32Array(3 * N), zrot = new Float32Array(N);
const sx = new Float32Array(N), sy = new Float32Array(N), ssz = new Float32Array(N);
let orderBuf = new Int32Array(N);

function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  const r = cv.parentElement.getBoundingClientRect();
  cv.width = Math.round(r.width * DPR); cv.height = Math.round(r.height * DPR);
  W = cv.width; H = cv.height;
  draw();
}
window.addEventListener('resize', resize);

// ---------- 颜色映射 ----------
function ramp(v, stops) {  // stops: [[pos,[r,g,b]],...]
  v = Math.max(0, Math.min(1, v));
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i], b = stops[i+1];
    if (v >= a[0] && v <= b[0]) {
      const t = (v - a[0]) / Math.max(1e-9, b[0] - a[0]);
      const c = [0,1,2].map(j => Math.round(a[1][j] + t * (b[1][j] - a[1][j])));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  const c = stops[stops.length-1][1];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
const CMAP_T = [[0,[40,20,90]],[0.25,[30,120,180]],[0.5,[40,190,120]],[0.75,[250,200,60]],[1,[250,240,180]]];
const CMAP_S = [[0,[20,70,130]],[0.5,[250,200,60]],[1,[200,30,30]]];

let cmode = 'cls';
function colorOf(i) {
  switch (cmode) {
    case 'cls':   return clsColor[CLS[i]] || '#888';
    case 'ang':   { const a = (ANG[i] + Math.PI) / (2*Math.PI); return ramp(a, CMAP_T); }
    case 't':     return ramp(TT[i] / (META.t_max || 1), CMAP_T);
    case 'score': {
      const lo = window.__slo, hi = window.__shi;
      return ramp((SC[i] - lo) / Math.max(1e-6, hi - lo), CMAP_S);
    }
    case 'ep':    return PAL[(EPI[i] % PAL.length)];
  }
  return '#888';
}
{
  const p = Array.from(SC).sort((a,b)=>a-b);
  window.__slo = p[Math.floor(0.02*p.length)]; window.__shi = p[Math.floor(0.98*p.length)];
}

// ---------- 主绘制 ----------
function draw() {
  const cy = Math.cos(yaw), sy_ = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const scale0 = Math.min(W, H) * 0.42 * zoom / ext;
  const cx = W / 2 + panX * DPR, cyy = H / 2 + panY * DPR;

  for (let i = 0; i < N; i++) {
    const x = PC[3*i], y = PC[3*i+1], z = PC[3*i+2];
    const x1 = cy * x + sy_ * z;
    const z1 = -sy_ * x + cy * z;
    const y2 = cp * y - sp * z1;
    const z2 = sp * y + cp * z1;
    zrot[i] = z2;
    const f = fov / (fov + z2 * ext * 1.2);
    sx[i] = cx + x1 * scale0 * f;
    sy[i] = cyy - y2 * scale0 * f;
    ssz[i] = f;
  }
  orderBuf = new Int32Array(N);
  for (let i = 0; i < N; i++) orderBuf[i] = i;
  const ord = Array.prototype.slice.call(orderBuf).sort((a, b) => zrot[a] - zrot[b]);

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0b0d12'; ctx.fillRect(0, 0, W, H);

  // 坐标轴
  const o = project(0, 0, 0, cy, sy_, cp, sp, scale0, cx, cyy);
  drawAxis(o, project(ext, 0, 0, cy, sy_, cp, sp, scale0, cx, cyy), '#c0392b', `PC1 ${META.pc_var[0]}%`);
  drawAxis(o, project(0, ext, 0, cy, sy_, cp, sp, scale0, cx, cyy), '#27ae60', `PC2 ${META.pc_var[1]}%`);
  drawAxis(o, project(0, 0, ext, cy, sy_, cp, sp, scale0, cx, cyy), '#2980b9', `PC3 ${META.pc_var[2]}%`);

  // 点云
  const base = parseFloat(document.getElementById('psize').value);
  const dimOn = document.getElementById('dim').checked;
  const selEp = parseInt(document.getElementById('epsel').value, 10);
  let shown = 0;
  for (let k = 0; k < N; k++) {
    const i = ord[k];
    if (cmode === 'cls' && !clsVisible[CLS[i]]) continue;
    if (dimOn && selEp >= 0 && EPI[i] !== selEp) continue;
    const s = base * ssz[i] * DPR;
    ctx.fillStyle = colorOf(i);
    ctx.globalAlpha = 0.55;
    ctx.fillRect(sx[i] - s/2, sy[i] - s/2, s, s);
    shown++;
  }
  ctx.globalAlpha = 1;

  // 轨迹
  if (selEp >= 0) {
    const a0 = OFF[selEp], a1 = OFF[selEp + 1];
    ctx.lineWidth = 1.8 * DPR; ctx.lineCap = 'round';
    for (let i = a0; i + 1 < a1; i++) {
      const t = (TT[i] - TT[a0]) / Math.max(1e-6, TT[a1-1] - TT[a0]);
      ctx.strokeStyle = ramp(t, CMAP_T);
      ctx.globalAlpha = 0.95;
      ctx.beginPath(); ctx.moveTo(sx[i], sy[i]); ctx.lineTo(sx[i+1], sy[i+1]); ctx.stroke();
    }
    // 起终点
    const s0 = Math.max(2.6, base) * DPR * 1.6;
    ctx.globalAlpha = 1;
    ctx.beginPath(); ctx.arc(sx[a0], sy[a0], s0, 0, 7); ctx.fillStyle = '#2ecc71'; ctx.fill();
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1.2*DPR; ctx.stroke();
    ctx.beginPath(); ctx.arc(sx[a1-1], sy[a1-1], s0, 0, 7); ctx.fillStyle = '#e74c3c'; ctx.fill(); ctx.stroke();
  }
  document.getElementById('k_shown').textContent = shown.toLocaleString();
}

function project(x, y, z, cy, sy_, cp, sp, scale0, cx, cyy) {
  const x1 = cy * x + sy_ * z;
  const z1 = -sy_ * x + cy * z;
  const y2 = cp * y - sp * z1;
  const z2 = sp * y + cp * z1;
  const f = fov / (fov + z2 * ext * 1.2);
  return [cx + x1 * scale0 * f, cyy - y2 * scale0 * f];
}
function drawAxis(o, p, col, label) {
  ctx.strokeStyle = col; ctx.globalAlpha = 0.5; ctx.lineWidth = 1.2 * DPR;
  ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
  ctx.globalAlpha = 0.85; ctx.fillStyle = col; ctx.font = `${11*DPR}px sans-serif`;
  ctx.fillText(label, p[0] + 4*DPR, p[1] - 4*DPR);
  ctx.globalAlpha = 1;
}

// ---------- 交互 ----------
let drag = null;
cv.addEventListener('mousedown', e => {
  drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
  cv.classList.add('dragging'); e.preventDefault();
});
window.addEventListener('mousemove', e => {
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan) { panX += dx; panY += dy; }
    else { yaw += dx * 0.008; pitch = Math.max(-1.55, Math.min(1.55, pitch + dy * 0.008)); }
    draw();
  } else hover(e);
});
window.addEventListener('mouseup', () => { drag = null; cv.classList.remove('dragging'); });
cv.addEventListener('contextmenu', e => e.preventDefault());
cv.addEventListener('wheel', e => {
  e.preventDefault();
  zoom *= Math.exp(-e.deltaY * 0.0012);
  zoom = Math.max(0.25, Math.min(8, zoom));
  draw();
}, { passive: false });
cv.addEventListener('dblclick', () => { resetView(); });

const PRESETS = {
  iso: [-0.6, 0.5],      // 等轴测
  p12: [0, 0],           // 沿 PC3 看 → 屏幕上是 PC1-PC2
  p23: [-Math.PI/2, 0],  // 沿 PC1 看 → PC3-PC2
  p13: [0, 1.45],        // 沿 PC2 看 → PC1-PC3
};
function resetView() {
  [yaw, pitch] = PRESETS.iso; zoom = 1; panX = 0; panY = 0; draw();
}
document.getElementById('reset').onclick = resetView;
document.querySelectorAll('[data-preset]').forEach(b => {
  b.onclick = () => { [yaw, pitch] = PRESETS[b.dataset.preset]; draw(); };
});
let spinReq = null;
document.getElementById('spin').onchange = e => {
  if (e.target.checked) {
    const step = () => { yaw += 0.006; draw(); spinReq = requestAnimationFrame(step); };
    spinReq = requestAnimationFrame(step);
  } else if (spinReq) { cancelAnimationFrame(spinReq); spinReq = null; }
};

function hover(e) {
  const r = cv.getBoundingClientRect();
  const mx = (e.clientX - r.left) * DPR, my = (e.clientY - r.top) * DPR;
  let best = -1, bd = (9 * DPR) ** 2;
  for (let i = 0; i < N; i++) {
    if (cmode === 'cls' && !clsVisible[CLS[i]]) continue;
    const d = (sx[i]-mx)**2 + (sy[i]-my)**2;
    if (d < bd) { bd = d; best = i; }
  }
  const tip = document.getElementById('tip');
  if (best < 0) { tip.style.display = 'none'; return; }
  tip.style.display = 'block';
  tip.style.left = (e.clientX - r.left + 14) + 'px';
  tip.style.top = (e.clientY - r.top + 14) + 'px';
  tip.innerHTML = `ep${META.ep_names[EPI[best]]}　t=${TT[best].toFixed(2)}s<br>`
    + `类 c${CLS[best]}　相位 ${(ANG[best]*180/Math.PI).toFixed(0)}°<br>`
    + `不典型度 ${SC[best].toFixed(3)}`;
}

// ---------- 侧栏 ----------
document.getElementById('ttl').textContent = '潜空间浏览器';
document.getElementById('sub').textContent = META.title;
document.getElementById('k_n').textContent = META.n.toLocaleString();
document.getElementById('k_ep').textContent = META.n_ep;
document.getElementById('k_var').textContent = META.pc_var.join(' / ') + ' %';

const clsBox = document.getElementById('clss');
clsList.forEach(c => {
  const row = document.createElement('label');
  row.className = 'row';
  row.innerHTML = `<input type="checkbox" checked data-c="${c}">
    <span class="sw" style="background:${clsColor[c]}"></span>
    <span>c${c}</span><span class="cnt">${(META.cls_counts[c]||0).toLocaleString()}</span>`;
  row.querySelector('input').onchange = ev => { clsVisible[c] = ev.target.checked; draw(); };
  clsBox.appendChild(row);
});

const epSel = document.getElementById('epsel');
META.ep_names.forEach((name, i) => {
  const o = document.createElement('option');
  o.value = i; o.textContent = `${name}  (${OFF[i+1] - OFF[i]} 帧)`;
  epSel.appendChild(o);
});
epSel.onchange = () => draw();
document.getElementById('dim').onchange = () => draw();
document.getElementById('psize').oninput = () => draw();
document.getElementById('fov').oninput = e => { fov = parseFloat(e.target.value); draw(); };
document.getElementById('cmode').onchange = e => {
  cmode = e.target.value;
  document.getElementById('clsbox').style.display = (cmode === 'cls') ? '' : 'none';
  updateLegend();
  draw();
};

function updateLegend() {
  const el = document.getElementById('legend');
  if (cmode === 'cls') {
    el.innerHTML = clsList.map(c => `<div><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${clsColor[c]};margin-right:5px"></span>c${c}</div>`).join('');
  } else if (cmode === 'ep') {
    el.textContent = '每个 episode 一种颜色（循环取色）';
  } else {
    const name = {ang:'相位角 −π → π', t:'时间 0 → ' + META.t_max.toFixed(1) + 's', score:'不典型度 低 → 高'}[cmode];
    const grad = Array.from({length: 40}, (_, i) => ramp(i/39, cmode === 'score' ? CMAP_S : CMAP_T)).join(',');
    el.innerHTML = `${name}<div class="bar" style="width:150px;background:linear-gradient(90deg,${grad})"></div>`;
  }
}
updateLegend();
resize();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
