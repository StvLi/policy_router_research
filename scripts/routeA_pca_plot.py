#!/usr/bin/env python3
"""路线 A：把"整任务 VAE"的潜变量做 PCA（3 主成分），并按子任务类别着色可视化

输入：routeA_task_latent.npz（由 scripts/routeA_task_latent.py 在 6c 生成）
输出：docs/assets/ 下四张图 + 终端指标

图：
  1. pca_classes.png     三个 2D 投影（PC1-2 / PC1-3 / PC2-3），按子任务着色 + 类质心
  2. pca_3d.png          3D 散点，按子任务着色
  3. pca_trajectory.png  若干条 episode 在 PC1-PC2 上的执行轨迹（按时间渐变、按类别打点）
  4. pca_time.png        PC1 与集内归一化时间的关系（看"潜空间主轴是不是进度轴"）

指标：
  - 三个主成分的解释方差比
  - 3 维 vs 16 维 上的「最近质心」与「softmax 回归」分类准确率（子任务 5 类）
  - 每条 episode 内 Spearman(时间, PC1) 的中位数（轨迹是否单调推进）
"""
import argparse
import json
import os

import numpy as np


def pca3(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    return Xc @ Vt[:3].T, ev[:3], Vt[:3]


def softmax_reg(X, y, k, iters=400, lr=0.5, seed=0):
    """极简多项 logistic 回归（numpy 梯度下降），返回训练好的权重。"""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(0, 0.01, (d, k)); b = np.zeros(k)
    onehot = np.eye(k)[y]
    for _ in range(iters):
        logits = X @ W + b
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits); p /= p.sum(1, keepdims=True)
        g = (p - onehot) / n
        W -= lr * (X.T @ g + 1e-4 * W)
        b -= lr * g.sum(0)
    return W, b


def acc(W, b, X, y):
    return float(((X @ W + b).argmax(1) == y).mean())


def nearest_centroid_acc(Xtr, ytr, Xte, yte, k):
    C = np.stack([Xtr[ytr == c].mean(0) if (ytr == c).any() else np.zeros(Xtr.shape[1]) for c in range(k)])
    d = ((Xte[:, None, :] - C[None]) ** 2).sum(-1)
    return float((d.argmin(1) == yte).mean())


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", default="/tmp/routeA_task_latent.npz")
    ap.add_argument("--outdir", default="docs/assets")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    os.makedirs(a.outdir, exist_ok=True)
    z = np.load(a.latent, allow_pickle=True)
    MU, Y, EP, FIDX = z["mu"], z["y"], z["ep"], z["fidx"]
    caps = list(z["caps"]) if "caps" in z.files else []
    print(f"潜变量 {MU.shape}；episode {len(np.unique(EP))}；类别 {sorted(set(Y.tolist()))}")

    # ---- PCA ----
    Z3, ev, _ = pca3(MU)
    print(f"解释方差比 PC1/2/3 = {ev[0]*100:.1f}% / {ev[1]*100:.1f}% / {ev[2]*100:.1f}%"
          f"（合计 {ev.sum()*100:.1f}%）")

    # ---- 按 episode 切分，评估 3 维 vs 16 维的子任务可分性 ----
    rng = np.random.default_rng(0)
    eps = np.unique(EP); rng.shuffle(eps)
    n_ho = max(1, int(round(len(eps) * a.holdout_frac)))
    ho = set(eps[:n_ho].tolist())
    te = np.array([e in ho for e in EP]); tr = ~te
    k = int(z["k"]) if "k" in z.files else int(Y.max() + 1)
    res = {}
    for name, X in (("PC1-3", Z3), ("full-16", MU)):
        W, b = softmax_reg(X[tr], Y[tr], k)
        res[name] = dict(
            softmax_train=acc(W, b, X[tr], Y[tr]),
            softmax_test=acc(W, b, X[te], Y[te]),
            centroid=nearest_centroid_acc(X[tr], Y[tr], X[te], Y[te], k))
    print(f"\n子任务 {k} 类可分性（留出 {n_ho} 集 / {te.sum()} 帧；随机 {100/k:.1f}%）")
    for n, d in res.items():
        print(f"  {n:8} softmax 测试 {d['softmax_test']*100:5.1f}%   "
              f"最近质心 {d['centroid']*100:5.1f}%   （训练 {d['softmax_train']*100:5.1f}%）")

    # ---- 每条 episode 内：PCA 主轴是否随进度单调推进 ----
    cors = []
    for e in np.unique(EP):
        m = EP == e
        if m.sum() < 20:
            continue
        t = FIDX[m].astype(float)
        cors.append(spearman(t, Z3[m, 0]))
    cors = np.array(cors)
    print(f"\n集内 Spearman(时间, PC1)：中位 {np.median(cors):+.3f}  "
          f"（|ρ|>0.5 的集 {np.mean(np.abs(cors)>0.5)*100:.0f}%，"
          f"正相关 {np.mean(cors>0.5)*100:.0f}%，负相关 {np.mean(cors<-0.5)*100:.0f}%）")

    # ---- 相位分析：轨迹若成闭环，则"角度"才是进度轴 ----
    center = np.median(Z3[:, :2], axis=0)
    ang = np.arctan2(Z3[:, 1] - center[1], Z3[:, 0] - center[0])
    radius = np.hypot(Z3[:, 0] - center[0], Z3[:, 1] - center[1])
    phase_cors, turns = [], []
    for e in np.unique(EP):
        idx = np.where(EP == e)[0]
        if len(idx) < 20:
            continue
        idx = idx[np.argsort(FIDX[idx])]
        th = np.unwrap(ang[idx])
        phase_cors.append(spearman(FIDX[idx].astype(float), th))
        turns.append(abs(th[-1] - th[0]) / (2 * np.pi))
    phase_cors = np.array(phase_cors); turns = np.array(turns)
    print(f"\n相位分析（以 PC1-PC2 的中位点为圆心）：")
    print(f"  集内 Spearman(时间, 展开后的角度)：中位 {np.median(phase_cors):+.3f}"
          f"（|ρ|>0.7 的集 {np.mean(np.abs(phase_cors)>0.7)*100:.0f}%）")
    print(f"  每集平均绕圈数：中位 {np.median(turns):.2f} 圈 ")
    # 用 (角度, 半径) 两个特征做分类，对比 3 个主成分
    PH = np.stack([np.cos(ang) * radius, np.sin(ang) * radius], 1)
    W, b = softmax_reg(PH[tr], Y[tr], k)
    print(f"  用圆心坐标(等价 2 维)做 softmax：测试 {acc(W, b, PH[te], Y[te])*100:.1f}%")

    # ---- 配色 ----
    ks = sorted(set(Y.tolist()))
    cmap = plt.get_cmap("tab10")
    colors = {c: cmap(i % 10) for i, c in enumerate(ks)}
    label_of = {}
    for c in ks:
        idx = np.where(np.array(z["cap_labels"]) == c)[0] if "cap_labels" in z.files else []
        label_of[c] = (caps[idx[0]][:26] if len(idx) else f"class {c}")

    # ---- 图 1：三个 2D 投影 ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (i, j) in zip(axes, pairs):
        for c in ks:
            m = Y == c
            ax.scatter(Z3[m, i], Z3[m, j], s=6, alpha=0.35, color=colors[c],
                       label=f"c{c} {label_of[c]}" if (i, j) == pairs[0] else None)
            ax.scatter(*np.array([np.median(Z3[m, i]), np.median(Z3[m, j])]),
                       marker="X", s=180, color=colors[c], edgecolor="k", linewidth=1.0, zorder=5)
        ax.set_xlabel(f"PC{i+1} ({ev[i]*100:.1f}%)")
        ax.set_ylabel(f"PC{j+1} ({ev[j]*100:.1f}%)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, markerscale=3, loc="best")
    fig.suptitle("Task-level VAE latent space (PCA, 3 components) — colored by subtask class\n"
                 "X = class median", fontsize=11)
    fig.tight_layout()
    p1 = os.path.join(a.outdir, "pca_classes.png"); fig.savefig(p1, dpi=150); plt.close(fig)
    print(f"写出 {p1}")

    # ---- 图 2：3D ----
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    for c in ks:
        m = Y == c
        ax.scatter(Z3[m, 0], Z3[m, 1], Z3[m, 2], s=5, alpha=0.3, color=colors[c], label=f"c{c}")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({ev[2]*100:.1f}%)")
    ax.legend(fontsize=8, markerscale=3)
    ax.set_title("Task-level VAE latent, 3D PCA")
    fig.tight_layout()
    p2 = os.path.join(a.outdir, "pca_3d.png"); fig.savefig(p2, dpi=150); plt.close(fig)
    print(f"写出 {p2}")

    # ---- 图 3：若干条的执行轨迹 ----
    show = eps[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, e in zip(axes.ravel(), show):
        m = EP == e
        if m.sum() < 5:
            ax.set_visible(False); continue
        order = np.argsort(FIDX[m])
        xs, ys, cs, ts = Z3[m, 0][order], Z3[m, 1][order], Y[m][order], FIDX[m][order]
        ax.plot(xs, ys, "-", color="0.75", lw=0.8, zorder=1)
        sc_ = ax.scatter(xs, ys, c=ts / 25.0, cmap="viridis", s=18, zorder=2)
        ax.scatter(xs, ys, s=60, facecolors="none", edgecolors=[colors[c] for c in cs],
                   linewidths=1.3, zorder=3)
        ax.scatter(xs[0], ys[0], marker="o", s=110, color="lime", edgecolor="k", zorder=4, label="start")
        ax.scatter(xs[-1], ys[-1], marker="s", s=110, color="red", edgecolor="k", zorder=4, label="end")
        ax.set_title(f"ep{e:06d}  ({m.sum()} frames)", fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(alpha=0.25)
        cb = plt.colorbar(sc_, ax=ax); cb.set_label("time (s)", fontsize=8)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Execution trajectory in PC1-PC2\n"
                 "line/dot color = time; ring color = subtask class", fontsize=11)
    fig.tight_layout()
    p3 = os.path.join(a.outdir, "pca_trajectory.png"); fig.savefig(p3, dpi=150); plt.close(fig)
    print(f"写出 {p3}")

    # ---- 图 4：PC1 vs 归一化时间 ----
    fig, ax = plt.subplots(figsize=(7, 5))
    tn, pc = [], []
    for e in np.unique(EP):
        m = EP == e
        if m.sum() < 10:
            continue
        t = FIDX[m].astype(float); t = (t - t.min()) / max(t.max() - t.min(), 1)
        tn.append(t); pc.append(Z3[m, 0])
    tn = np.concatenate(tn); pc = np.concatenate(pc)
    hb = ax.hexbin(tn, pc, gridsize=45, cmap="magma", mincnt=1)
    plt.colorbar(hb, ax=ax, label="frames")
    ax.set_xlabel("normalized time within episode")
    ax.set_ylabel("PC1")
    ax.set_title(f"PC1 vs progress (median within-episode Spearman {np.median(cors):+.2f})")
    fig.tight_layout()
    p4 = os.path.join(a.outdir, "pca_time.png"); fig.savefig(p4, dpi=150); plt.close(fig)
    print(f"写出 {p4}")

    # ---- 图 5：角度随时间的推进（相位图）----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for e in eps[:6]:
        idx = np.where(EP == e)[0]
        if len(idx) < 20:
            continue
        idx = idx[np.argsort(FIDX[idx])]
        ax.plot(FIDX[idx] / 25.0, np.unwrap(ang[idx]) / (2 * np.pi), lw=1.2,
                label=f"ep{e:06d}")
    ax.set_xlabel("time (s)"); ax.set_ylabel("unwrapped angle / 2π  (turns)")
    ax.set_title(f"Phase advance along the loop (median Spearman {np.median(phase_cors):+.2f})")
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    ax = axes[1]
    sc_ = ax.scatter(ang, radius, c=Y, cmap="tab10", s=5, alpha=0.4)
    ax.set_xlabel("angle (rad)"); ax.set_ylabel("radius in PC1-PC2")
    ax.set_title("Classes as arcs of the loop")
    fig.tight_layout()
    p5 = os.path.join(a.outdir, "pca_phase.png"); fig.savefig(p5, dpi=150); plt.close(fig)
    print(f"写出 {p5}")

    json.dump(dict(explained_variance=ev.tolist(), k=k, separability=res,
                   phase=dict(spearman_median=float(np.median(phase_cors)),
                              frac_abs_gt07=float(np.mean(np.abs(phase_cors) > 0.7)),
                              turns_median=float(np.median(turns))),
                   spearman_pc1_time=dict(median=float(np.median(cors)),
                                          frac_abs_gt05=float(np.mean(np.abs(cors) > 0.5)),
                                          frac_pos=float(np.mean(cors > 0.5)),
                                          frac_neg=float(np.mean(cors < -0.5)))),
              open(os.path.join(a.outdir, "pca_metrics.json"), "w"), indent=2)
    print(f"写出 {os.path.join(a.outdir, 'pca_metrics.json')}")


if __name__ == "__main__":
    main()
