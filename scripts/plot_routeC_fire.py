#!/usr/bin/env python3
"""路线 C（冻结 VLA 表征 + VAE）预实验的可视化：测试轨迹 + 接管检出点

输入（从 ws:/home/syt/e2/ 取回）：
  /tmp/eval.npz                评测 8 局的 2048 维表征 E / EP / T（1,214 帧）
  /tmp/demo.npz                演示 20 局的表征（2,035 帧，用于拟合 PCA 基）
  /tmp/eval_score_frames.npz   逐帧 score / p
  /tmp/eval_score.json         逐集汇总与官方得分

输出：
  docs/assets/routeC_score_timeline.png   8 局的 score/p 曲线 + 触发点 + 规范阶段背景
  docs/assets/routeC_latent_traj.png      路线 C 表征空间（PCA 3）的轨迹 + 触发点
  docs/assets/routeC_metrics.json         图里用到的数值

触发规则（报告 §4.4）：滑窗 5 个采样点、窗口内 ≥3 个 p<0.05 → 判定脱轨。
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

# 中文字体（否则标题全变方框）
for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    try:
        font_manager.fontManager.addfont(_f)
    except Exception:
        pass
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 报告 §4.5 的规范阶段时间线（累计结束百分比）
PHASES = [("pick headphones", 12.8), ("hang headphones", 22.8), ("close laptop", 44.8),
          ("pick laptop", 67.0), ("hand over", 83.0), ("into stand", 100.0)]
PHASE_COLORS = ["#3b6ea5", "#4f9d69", "#8a6fbf", "#c98a3a", "#b05a5a", "#7a8794"]
ALPHA = 0.05
WIN, NEED = 5, 3


def fire_point(p):
    """滑窗 5、窗口内 >=3 个 p<alpha → 返回触发窗口起点（采样点序号），否则 -1。"""
    p = np.asarray(p)
    for i in range(0, len(p) - WIN + 1):
        if int((p[i:i + WIN] < ALPHA).sum()) >= NEED:
            return i
    return -1


def main():
    ev = np.load("/tmp/eval.npz", allow_pickle=True)
    de = np.load("/tmp/demo.npz", allow_pickle=True)
    fr = np.load("/tmp/eval_score_frames.npz", allow_pickle=True)
    meta = json.load(open("/tmp/eval_score.json"))

    E, EP, T = ev["E"], ev["EP"], ev["T"]
    S, P = fr["score"], fr["p"]
    assert len(E) == len(S) == len(EP)
    off_meta = {e["ep"]: e for e in meta["episodes"]}
    eps = sorted(set(EP.tolist()))
    print(f"评测 {len(E)} 帧 / {len(eps)} 局；演示 {len(de['E'])} 帧")

    # ---- PCA：只在演示表征上拟合，再投影评测 ----
    mu = de["E"].mean(0, keepdims=True)
    _, sv, Vt = np.linalg.svd(de["E"] - mu, full_matrices=False)
    ev_ratio = (sv ** 2) / (sv ** 2).sum()
    basis = Vt[:3].T
    Z = (E - mu) @ basis
    print(f"路线 C 表征 PCA（演示集拟合）：PC1-3 解释方差 "
          f"{ev_ratio[0]*100:.1f}/{ev_ratio[1]*100:.1f}/{ev_ratio[2]*100:.1f}%")

    # ---- 逐集算触发点 ----
    fire = {}
    for e in eps:
        m = EP == e
        idx = np.where(m)[0]
        order = idx[np.argsort(T[idx])]
        fp = fire_point(P[order])
        fire[e] = dict(first_fire=fp, n=len(order),
                       frac=(fp / max(1, len(order) - 1) * 100) if fp >= 0 else None,
                       t=(T[order][fp] / 25.0 if fp >= 0 else None))
    for e in eps:
        f = fire[e]
        print(f"  ep{e:02d} n={f['n']:3d} "
              f"{'触发@%d (%.1f%%, %.1fs)' % (f['first_fire'], f['frac'], f['t']) if f['first_fire'] >= 0 else '未触发'}")

    # ================= 图 1：score / p 曲线 + 触发点 =================
    ncol, nrow = 2, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.1 * nrow))
    for ax, e in zip(axes.ravel(), eps):
        m = EP == e
        idx = np.where(m)[0]
        order = idx[np.argsort(T[idx])]
        t = T[order]; s = S[order]; p = P[order]
        n = len(order)
        prog = (t - t.min()) / max(1e-6, t.max() - t.min()) * 100

        # 规范阶段背景
        x0 = 0.0
        for (name, x1), col in zip(PHASES, PHASE_COLORS):
            ax.axvspan(x0, min(x1, 100), color=col, alpha=0.07, lw=0)
            if e == eps[0] and x1 <= 100:
                ax.text((x0 + x1) / 2, 0.97, name.split()[0], transform=ax.get_xaxis_transform(),
                        fontsize=6, ha="center", va="top", color=col, rotation=90)
            x0 = x1

        ax.plot(prog, s, "-", color="#c0392b", lw=1.4, label="score (重建 MSE+KL)")
        ax.set_ylabel("score", color="#c0392b", fontsize=8)
        ax.tick_params(axis="y", labelcolor="#c0392b", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylim(min(s.min() - 0.02, 0.3), max(s.max() + 0.02, 0.7))

        ax2 = ax.twinx()
        ax2.plot(prog, p, "-", color="#2c6fbb", lw=1.0, alpha=0.85, label="conformal p")
        ax2.axhline(ALPHA, color="#2c6fbb", ls=":", lw=0.8, alpha=0.6)
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_ylabel("p", color="#2c6fbb", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#2c6fbb", labelsize=7)

        f = fire[e]
        if f["first_fire"] >= 0:
            x = (f["first_fire"] / max(1, n - 1)) * 100
            span = (WIN / max(1, n - 1)) * 100
            ax.axvspan(x, x + span, color="#e67e22", alpha=0.35, lw=0)
            ax.axvline(x, color="#d35400", lw=1.6)
            ax.annotate(f"接管检出\n{f['frac']:.0f}% / {f['t']:.1f}s",
                        xy=(x, ax.get_ylim()[1]), xytext=(4, -10), textcoords="offset points",
                        fontsize=7, color="#d35400", va="top")

        off = off_meta.get(int(e), {}).get("official_score", float("nan"))
        ax.set_title(f"layout {e} · 官方 {off:.1f} · 被标 "
                     f"{off_meta.get(int(e), {}).get('flag_rate', 0)*100:.0f}% 帧 · "
                     f"p 中位 {off_meta.get(int(e), {}).get('p_median', 0):.2f}",
                     fontsize=8.5, loc="left")
        ax.set_xlabel("轨迹进度 %", fontsize=7)
        ax.grid(alpha=0.18)
        ax.set_xlim(0, 100)

    fig.suptitle("路线 C · 8 局评测的分数曲线与接管检出点\n"
                 "背景色带 = 演示集的规范阶段时间线；橙色竖线 = 滑窗规则的首次触发（5 点窗内 ≥3 点 p<0.05）",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out1 = "docs/assets/routeC_score_timeline.png"
    fig.savefig(out1, dpi=150)
    print("写出", out1)

    # ================= 图 2：每局一个小面板的轨迹 + 触发点 =================
    ncol, nrow = 4, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow))
    for ax, e in zip(axes.ravel(), eps):
        m = EP == e
        idx = np.where(m)[0]
        order = idx[np.argsort(T[idx])]
        x, y, tt = Z[order, 0], Z[order, 1], T[order]
        ax.plot(x, y, "-", color="0.72", lw=0.9, zorder=1)
        sc = ax.scatter(x, y, c=tt / 25.0, cmap="viridis", s=16, zorder=2)
        ax.scatter(x[0], y[0], marker="o", s=55, color="#2ecc71", edgecolor="k", lw=0.6, zorder=4)
        ax.scatter(x[-1], y[-1], marker="s", s=55, color="#c0392b", edgecolor="k", lw=0.6, zorder=4)
        f = fire[e]
        off = off_meta.get(int(e), {}).get("official_score", float("nan"))
        if f["first_fire"] >= 0:
            k = f["first_fire"]
            ax.scatter(x[k], y[k], marker="*", s=320, color="#f39c12",
                       edgecolor="#7a3e00", lw=0.8, zorder=5)
            ax.annotate(f"接管 {f['frac']:.0f}%", (x[k], y[k]), xytext=(6, 5),
                        textcoords="offset points", fontsize=7, color="#7a3e00", zorder=6)
            ax.plot(x[:k + 1], y[:k + 1], "-", color="#f39c12", lw=2.0, alpha=0.9, zorder=3)
        ax.set_title(f"layout {e} · 官方 {off:.1f} · "
                     f"{'触发 @%.0f%%' % f['frac'] if f['first_fire'] >= 0 else '未触发'}",
                     fontsize=8.5, loc="left")
        ax.set_xlabel(f"PC1 ({ev_ratio[0]*100:.1f}%)", fontsize=7)
        ax.set_ylabel(f"PC2 ({ev_ratio[1]*100:.1f}%)", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.18)

    fig.suptitle("路线 C 表征空间（演示集拟合的 PCA）· 每局轨迹与接管检出点\n"
                 "颜色 = 时间（深→浅）；绿圆 = 起点、红方 = 终点；橙星 = 首次触发，橙色加粗段为触发前已走路径",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out2 = "docs/assets/routeC_latent_traj.png"
    fig.savefig(out2, dpi=150)
    print("写出", out2)

    json.dump(dict(pc_var=[round(float(v) * 100, 2) for v in ev_ratio[:3]],
                   n_eval=int(len(E)), n_demo=int(len(de["E"])),
                   fire={int(e): dict(first_fire=int(fire[e]["first_fire"]),
                                      progress_pct=(round(fire[e]["frac"], 1) if fire[e]["frac"] is not None else None),
                                      t_s=(round(float(fire[e]["t"]), 2) if fire[e]["t"] is not None else None),
                                      n=int(fire[e]["n"]))
                         for e in eps},
                   official={int(e): off_meta.get(int(e), {}).get("official_score") for e in eps},
                   rule=dict(window=WIN, need=NEED, alpha=ALPHA)),
              open("docs/assets/routeC_metrics.json", "w"), indent=2, ensure_ascii=False)
    print("写出 docs/assets/routeC_metrics.json")


if __name__ == "__main__":
    main()
