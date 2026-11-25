#!/usr/bin/env python3
"""路线 C α 核验图：阈值扫描、逐局 p 热条、触发进度散点。

输入：/tmp/eval25_score_frames.npz（EP/T/score/p） + /tmp/routeC_alpha_metrics.json
输出：docs/assets/routeC_alpha_sweep.png, routeC_alpha_strips.png, routeC_alpha_timing.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    try:
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

A0 = 0.12


def main():
    m = json.load(open("/tmp/routeC_alpha_metrics.json"))
    d = np.load("/tmp/eval25_score_frames.npz")
    EP, T, P = d["EP"], d["T"], d["p"]
    eps = sorted(set(EP.tolist()))
    off_meta = {r["ep"]: r for r in m["episodes"]}

    # ---- 图 1：阈值扫描 ----
    sw = m["sweep"]
    al = [s["alpha"] for s in sw]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(al, [s["tp"] / s["n_fail"] * 100 for s in sw], "o-", color="#c0392b", label="失败轨迹检出率")
    ax.plot(al, [s["fp"] / max(1, s["n_succ"]) * 100 for s in sw], "s--", color="#2c7fb8", label="成功轨迹误报率")
    ax.axvline(A0, color="#e0a33a", lw=1.2, ls=":")
    ax.annotate(f"α={A0}", (A0, 40), color="#e0a33a", fontsize=10, ha="left")
    ax.set_xlabel("阈值 α（p < α 视为不像演示）")
    ax.set_ylabel("百分比 %")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    ax.set_title(f"路线 C 判据：α 扫描（{m['n_fail']} 失败 / {m['n_succ']} 成功，滑窗 5 帧 ≥3 帧）",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig("docs/assets/routeC_alpha_sweep.png", dpi=140)
    print("写出 docs/assets/routeC_alpha_sweep.png")

    # ---- 图 2：逐局 p 热条 ----
    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    for i, e in enumerate(eps):
        idx = np.where(EP == e)[0]
        o = idx[np.argsort(T[idx])]
        p = P[o]
        prog = (T[o] - T[o].min()) / max(1e-6, T[o].max() - T[o].min())
        for j in range(len(p)):
            c = plt.cm.RdYlBu(np.clip((p[j] - 0.0) / 0.5, 0, 1))
            ax.add_patch(plt.Rectangle((prog[j], i - 0.38), max(0.012, 1.0 / len(p)), 0.76,
                                       color=c, lw=0))
        r = off_meta[e]
        f = r[f"fire_{A0:.2f}"]
        if f >= 0:
            ax.plot([prog[f]], [i], marker="*", ms=11, color="#ff5a1f", zorder=5)
        tag = "成功" if r["official"] >= 0.999 else "失败"
        ax.text(-0.03, i, f"ep{e:2d} {tag} 官{r['official']:.1f}", ha="right", va="center", fontsize=8)
    ax.set_xlim(-0.22, 1.0)
    ax.set_ylim(-0.8, len(eps) - 0.2)
    ax.set_yticks([])
    ax.set_xlabel("轨迹进度（归一化）")
    ax.set_title(f"逐局 p 值热条（蓝=像演示 p→1，红=p→0）；★=α={A0} 时的首次触发点", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap="RdYlBu", norm=plt.Normalize(0, 0.5))
    cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("p 值", fontsize=9)
    plt.tight_layout()
    plt.savefig("docs/assets/routeC_alpha_strips.png", dpi=140)
    print("写出 docs/assets/routeC_alpha_strips.png")

    # ---- 图 3：触发进度 vs 官方得分 ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r in m["episodes"]:
        f, prog = r[f"fire_{A0:.2f}"], r[f"prog_{A0:.2f}"]
        ins = "in-sample" if r["ep"] in (2, 8, 11, 12, 13, 17, 19, 24) else "样本外"
        if f >= 0:
            ax.scatter(r["official"], prog, s=46,
                       color="#c0392b" if ins == "in-sample" else "#e67e22",
                       marker="o" if ins == "in-sample" else "^", zorder=3)
            ax.annotate(f"ep{r['ep']}", (r["official"], prog), fontsize=7,
                        xytext=(4, 3), textcoords="offset points")
        else:
            ax.scatter(r["official"], -6, s=42, facecolors="none",
                       edgecolors="#7c8698", marker="x")
            ax.annotate(f"ep{r['ep']}（未检出）", (r["official"], -6), fontsize=7,
                        xytext=(4, -9), textcoords="offset points", color="#7c8698")
    ax.axhline(0, color="#39404e", lw=0.8)
    ax.set_xlabel("官方部分得分（1.0 = 完成）")
    ax.set_ylabel(f"α={A0} 首次触发处的进度 %")
    ax.set_ylim(-10, 105)
    ax.grid(alpha=0.25)
    ax.set_title("触发时刻落在任务进度的哪里", fontsize=10)
    plt.tight_layout()
    plt.savefig("docs/assets/routeC_alpha_timing.png", dpi=140)
    print("写出 docs/assets/routeC_alpha_timing.png")


if __name__ == "__main__":
    main()
