#!/usr/bin/env python3
"""画 25 条真实评测轨迹的两级 OOD 判定图（本地跑，需要 matplotlib）。

输入：
  <prefix>_scores.json   routeA_eval25_ood.py score 的输出
  <prefix>_idcheck.json  routeA_eval25_ood.py idcheck 的输出（演示留出集对照）
  <prefix>_result.json   评测 _result.json（官方 success / 部分得分）

输出：
  <out>_timeline.png  25 条轨迹的原始子任务 NLL 随时间变化（按官方得分着色）
  <out>_summary.png   逐集 NLL vs 官方得分、评测/演示分布对比、判定占比
  <out>_metrics.json  图上引用的全部数字
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体（本机有 Noto CJK；找不到就退回英文标签也不会报错）
for _f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK JP",
           "Droid Sans Fallback", "AR PL UMing CN"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.family"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

SCORE_COLOR = {1.0: "#1a7f37", 0.8: "#4c9a2a", 0.2: "#d98f00", 0.0: "#c0392b"}


def load(p):
    with open(p) as f:
        return json.load(f)


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="score/idcheck 输出的公共前缀")
    ap.add_argument("--out", default="docs/assets/eval25_ood")
    a = ap.parse_args()

    rec = load(a.prefix + "_scores.json")
    official = load(a.prefix + "_result.json")["details"]
    try:
        idc = load(a.prefix + "_idcheck.json")
    except FileNotFoundError:
        idc = {}

    eps = rec["episodes"]
    scores = np.array([official[str(e["ep"])]["score"] for e in eps])
    raw_sub = np.array([e["raw_sub_mean"] for e in eps])
    raw_task = np.array([e["raw_task_mean"] for e in eps])

    metrics = dict(
        n=len(eps), n_success=int(sum(e["success"] for e in eps)),
        rate_subtask=float(np.mean([e["rate_subtask"] for e in eps])),
        rate_task=float(np.mean([e["rate_task"] for e in eps])),
        rate_cap=float(np.mean([e["rate_cap"] for e in eps])),
        corr_raw_sub_score=dict(pearson=float(np.corrcoef(raw_sub, scores)[0, 1]),
                                spearman=spearman(raw_sub, scores)),
        corr_raw_task_score=dict(pearson=float(np.corrcoef(raw_task, scores)[0, 1]),
                                 spearman=spearman(raw_task, scores)),
        eval_raw_sub_p50=float(np.median(np.concatenate(
            [np.asarray(pf["raw_sub"]) for pf in rec["per_frame"]]))),
        id_check=idc,
    )

    # ---- 图 1：逐集 NLL 曲线 ----
    n = len(eps)
    ncol, nrow = 5, int(np.ceil(n / 5))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 1.5 * nrow), squeeze=False)
    orders = np.argsort(-scores)
    for ax_i, idx in enumerate(orders):
        e, pf = eps[idx], rec["per_frame"][idx]
        ax = axes[ax_i // ncol][ax_i % ncol]
        rs = np.asarray(pf["raw_sub"])
        f = np.asarray(pf["f"])
        col = SCORE_COLOR.get(float(scores[idx]), "#666")
        ax.plot(f, rs, color=col, lw=1.1)
        if idc:
            ax.axhline(idc["raw_sub_quantiles"]["p50"], color="#999", lw=0.7, ls=":")
            ax.axhline(idc["raw_sub_quantiles"]["p90"], color="#999", lw=0.7, ls="--")
        ax.set_title(f"ep{e['ep']:03d} score {scores[idx]:.1f}  mean {rs.mean():.2f}",
                     fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0.8, 2.4)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("25 条评测轨迹：最近子任务 VAE 的原始 NLL（虚线=演示留出中位/90 分位）", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out + "_timeline.png", dpi=140)
    plt.close(fig)

    # ---- 图 2：汇总 ----
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    # (a) 逐集均值 vs 官方得分
    for s in sorted(set(scores.tolist())):
        m = scores == s
        ax[0].scatter(np.full(m.sum(), s) + np.linspace(-0.03, 0.03, m.sum()), raw_sub[m],
                      color=SCORE_COLOR.get(s, "#666"), s=42, alpha=0.85,
                      label=f"score {s:.1f}（{m.sum()} 集）")
    ax[0].set_xlabel("官方部分得分")
    ax[0].set_ylabel("整集平均 raw_sub（越小越像演示）")
    ax[0].set_title(f"逐集相关性：Spearman {metrics['corr_raw_sub_score']['spearman']:+.2f}")
    ax[0].legend(fontsize=7)
    # (b) 分布对比：评测 vs 演示留出
    ev = np.concatenate([np.asarray(pf["raw_sub"]) for pf in rec["per_frame"]])
    ax[1].hist(ev, bins=40, density=True, alpha=0.55, color="#2b7bba", label="评测轨迹")
    if idc:
        q = idc["raw_sub_quantiles"]
        ax[1].axvline(q["p50"], color="#1f4e79", lw=1.2, ls="--",
                      label=f"演示留出中位 {q['p50']:.2f}")
        ax[1].axvline(q["p90"], color="#1f4e79", lw=1.0, ls=":")
    ax[1].set_xlim(0.8, 2.6)
    ax[1].set_xlabel("raw_sub（帧级）")
    ax[1].set_title("评测 vs 演示的帧级分布")
    ax[1].legend(fontsize=7)
    # (c) 判定占比
    parts = [("子任务内", metrics["rate_subtask"]),
             ("仅整任务内", metrics["rate_task"]),
             ("超出→CAP", metrics["rate_cap"])]
    ax[2].bar([p[0] for p in parts], [p[1] for p in parts],
              color=["#2b7bba", "#e8a33d", "#c0392b"])
    for i, (_, v) in enumerate(parts):
        ax[2].text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    ax[2].set_ylim(0, 1.1)
    ax[2].set_title("冻结口径下的两路判定（25 集 × 全部帧）")
    fig.tight_layout()
    fig.savefig(a.out + "_summary.png", dpi=140)
    plt.close(fig)

    with open(a.out + "_metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
