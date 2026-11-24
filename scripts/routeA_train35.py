#!/usr/bin/env python3
"""对 35 个任务逐一：句向量聚类 → 每类训 VAE → 在留出 episode 上测"分段有效性"

与路线 A 完全同一套判据（3 相机当前帧 + t/t-12/t-50 关节状态 + 自重构 + 池化 3x4 + 共形 p），
只是把它从 1 个任务推广到 35 个任务，并按 worker 分片以吃满 6 张卡。

每个任务输出：
  - 聚类结果（k、各类句数/帧数）
  - 每个类的 VAE 是否训练成功（样本阈值）
  - 留出 episode 上的路由准确率 / 平衡准确率（= 分段有效性）
汇总写到 --out 指定的 jsonl（每任务一行），便于后续合并。

用法（6c）：
  CUDA_VISIBLE_DEVICES=0 python routeA_train35.py --worker 0 --workers 6
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, "/tmp")
import pilot_store_laptop as A


def task_episodes(dataset):
    import pyarrow.parquet as pq
    out = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(dataset, "meta", "episodes", "chunk-*", "file-*.parquet"))):
        for r in pq.read_table(p, columns=["episode_index", "tasks"]).to_pylist():
            if not r["tasks"]:
                continue
            head = r["tasks"][0].split("Current subtask:")[0].strip()
            if head.lower().startswith("task:"):
                head = head[5:].strip()
            out[head].append(int(r["episode_index"]))
    return {k: sorted(v) for k, v in out.items()}


def decode_all(clips, ep):
    return A.decode_views(clips, ep, 1, keep_all=True)


def run_task(task, eps, a, dev, cap2vec, torch):
    import torch.nn.functional as F
    segs_of_ep = {}
    for ep in eps:
        s = A.load_segments(a.labels, ep)
        if s:
            segs_of_ep[ep] = s
    if len(segs_of_ep) < 4:
        return dict(task=task, status="skip", reason=f"only {len(segs_of_ep)} labelled episodes")

    caps = sorted({x["caption"].strip().lower() for v in segs_of_ep.values() for x in v})
    if len(caps) < a.k:
        return dict(task=task, status="skip", reason=f"only {len(caps)} distinct captions")

    # --- 第 1 步：句向量（预算查表）+ 聚类 ---
    missing = [c for c in caps if c not in cap2vec]
    if missing:
        return dict(task=task, status="skip",
                    reason=f"{len(missing)}/{len(caps)} captions 不在预算表里，例如: {missing[0][:60]}")
    E = np.stack([cap2vec[c] for c in caps]).astype(np.float32)
    lab, _ = A.spherical_kmeans(E, a.k)
    cap2clu = {c: int(l) for c, l in zip(caps, lab)}

    # --- 取帧 ---
    Xi, EP, FIDX, CI = [], [], [], []
    ci = {c: i for i, c in enumerate(caps)}
    st_map = A.load_all_states(a.dataset, list(segs_of_ep))   # 一次读全（原来在循环里反复全量扫）
    for ep in segs_of_ep:
        st = st_map.get(ep)
        if st is None:
            continue
        views = decode_all(a.clips, ep)
        if views is None:
            continue
        T = min((len(views) - 1) // a.stride + 1, (len(st) - 1) // a.stride + 1)
        lb, amb = A.frame_labels(segs_of_ep[ep], len(st))
        for i in range(T):
            f = i * a.stride
            if f >= len(views) or f >= len(st) or amb[f]:
                continue
            j = lb[f]
            if j < 0:
                continue
            Xi.append(views[f][None]); EP.append(ep); FIDX.append(f)
            CI.append(ci[segs_of_ep[ep][j]["caption"].strip().lower()])
    if len(Xi) < 300:
        return dict(task=task, status="skip", reason=f"only {len(Xi)} frames")

    Xi = np.stack(Xi)
    Xt = Xi                     # 自重构（--pred-k 0）：重建目标就是输入本身
    EP = np.asarray(EP); FIDX = np.asarray(FIDX); CI = np.asarray(CI)
    Y = np.array([cap2clu[caps[i]] for i in CI], np.int64)
    Xs = A.state_windows(st_map, EP, FIDX)
    mu_s = Xs.reshape(-1, A.NS).mean(0); sd_s = Xs.reshape(-1, A.NS).std(0) + 1e-6
    Xs = ((Xs - mu_s) / sd_s).astype(np.float32); Xst = Xs.reshape(len(Xs), -1)

    # --- 划分 ---
    eps_sorted = sorted(set(EP.tolist())); random.Random(0).shuffle(eps_sorted)
    n_val = max(2, int(round(len(eps_sorted) * 0.25)))
    val = set(eps_sorted[:n_val]); tr = np.array([e not in val for e in EP])

    # --- 每类 VAE + 共形校准 ---
    models, cal, sizes, skipped = {}, {}, {}, []
    for c in range(a.k):
        m = tr & (Y == c)
        sizes[c] = int(m.sum())
        if m.sum() < a.min_frames:
            skipped.append(c); continue
        vae, _ = A.train_vae(Xi[m], Xt[m], Xs[m], Xst[m], a.iters, a.beta, dev, seed=c)
        models[c] = vae
        idx = np.where(m)[0]; cut = int(len(idx) * 0.7)
        cal[c] = A.nll_of(vae, Xi[idx[:cut]], Xt[idx[:cut]], Xs[idx[:cut]], Xst[idx[:cut]], dev)
    if len(models) < 2:
        return dict(task=task, status="skip", reason=f"only {len(models)} trainable classes")

    # --- 路由（分段有效性）---
    mask = ~tr
    idx = np.where(mask)[0]
    P = np.zeros((len(idx), a.k))
    for c, vae in models.items():
        P[:, c] = A.conformal_p(A.nll_of(vae, Xi[idx], Xt[idx], Xs[idx], Xst[idx], dev), cal[c])
    pred = P.argmax(1); y = Y[idx]
    acc = float((pred == y).mean())
    have = np.isin(y, list(models))
    rec = [float((pred[y == c] == c).mean()) for c in sorted(models) if (y == c).any()]
    return dict(task=task, status="ok", k=a.k,
                n_ep=len(segs_of_ep), n_frames=int(len(Y)), n_val_frames=int(len(idx)),
                sizes={int(k): int(v) for k, v in sizes.items()},
                skipped=[int(x) for x in skipped], n_models=len(models),
                acc=acc, balanced=float(np.mean(rec)),
                acc_avail=float((pred[have] == y[have]).mean()) if have.any() else None,
                cover=float(have.mean()), recall={int(c): float(r) for c, r in zip(sorted(models), rec)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=A.DEFAULT_DATASET)
    ap.add_argument("--labels", default=A.DEFAULT_LABELS)
    ap.add_argument("--clips", default=A.DEFAULT_CLIPS)
    ap.add_argument("--emb-cache", default="/tmp/allcap_emb.npz",
                    help="预算好的 caption->句向量表（vllm 环境生成）")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--episodes-per-task", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--min-frames", type=int, default=200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    a.out = a.out or f"/tmp/routeA35_w{a.worker}.jsonl"

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A.set_hist("vlm3"); A.set_img_hist("single"); A.set_pred_k(0); A.STATE_W = 4.0
    bundle = np.load(a.emb_cache, allow_pickle=True)
    cap2vec = {c: v for c, v in zip(list(bundle["caps"]), bundle["emb"])}
    print(f"[w{a.worker}] 句向量表 {len(cap2vec)} 条", flush=True)

    tasks = task_episodes(a.dataset)
    names = sorted(tasks)
    mine = [t for i, t in enumerate(names) if i % a.workers == a.worker]
    print(f"[w{a.worker}] 共 {len(names)} 任务，本片 {len(mine)} 个；GPU={dev}", flush=True)

    with open(a.out, "w") as f:
        for n, t in enumerate(mine, 1):
            eps = tasks[t][:a.episodes_per_task]
            try:
                r = run_task(t, eps, a, dev, cap2vec, torch)
            except Exception as e:
                r = dict(task=t, status="error", reason=f"{type(e).__name__}: {e}")
            r["worker"] = a.worker
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(f"[w{a.worker}] {n}/{len(mine)} {t[:52]:54} {r.get('status')} "
                  f"acc={r.get('acc')} bal={r.get('balanced')}", flush=True)


if __name__ == "__main__":
    main()
