#!/usr/bin/env python3
"""路线 A：用**整条任务**的帧训一个 VAE，导出每帧的潜变量（供本地做 PCA 可视化）

与路线 A 的单任务分流区别：这里不分类，只用全部帧训练**一个** task-level VAE，
拿到 z 之后再看"子任务类别是否在潜空间里形成可分的区域/轨迹"。

复用 route A 的冻结配置（3 相机当前帧 + t/t-12/t-50 状态，自重构，池化 3x4，隐维 16）
—— 通过 import /tmp/pilot_store_laptop.py 保证与冻结版本逐字节同源。

用法（6c，labeler 环境）：
  CUDA_VISIBLE_DEVICES=4 $L routeA_task_latent.py
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/tmp")
import pilot_store_laptop as A          # 路线 A 的模块（冻结配置）

CACHE = "/tmp/pilot_frames_single_s5_k0_nall.npz"
SPEC = "/tmp/pilot_spec.json"
EMB = "/tmp/pilot_emb.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--spec", default=SPEC)
    ap.add_argument("--emb", default=EMB)
    ap.add_argument("--dataset", default=A.DEFAULT_DATASET)
    ap.add_argument("--k", type=int, default=5, help="子任务类别数（只用于着色与评估，不参与训练）")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--state-weight", type=float, default=4.0)
    ap.add_argument("--out", default="/tmp/routeA_task_latent.npz")
    a = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A.set_hist("vlm3"); A.set_img_hist("single"); A.set_pred_k(0)
    A.STATE_W = a.state_weight
    A.check_prep_img()

    spec = json.load(open(a.spec))
    z = np.load(a.cache, allow_pickle=True)
    Xi, Xt, EP, FIDX, CI = z["Xi"], z["Xt"], z["EP"], z["FIDX"], z["CI"]
    print(f"帧 {len(CI)}；episode {len(np.unique(EP))}")

    # 子任务类别（仅用于着色/评估）
    bundle = np.load(a.emb, allow_pickle=True)
    caps = list(bundle["caps"]); E = bundle["emb"]
    cap_lab, _ = A.spherical_kmeans(E, min(a.k, len(caps)))
    cap2clu = {c: int(l) for c, l in zip(caps, cap_lab)}
    import collections
    Y = np.array([cap2clu[caps[i]] for i in CI], np.int64)
    print("类别分布:", dict(sorted(collections.Counter(Y.tolist()).items())))

    # 状态窗口 + 目标（与冻结配置一致：k=0，目标=输入）
    st = A.load_all_states(a.dataset, np.unique(EP))
    Xs = A.state_windows(st, EP, FIDX)
    mu_s = Xs.reshape(-1, A.NS).mean(0); sd_s = Xs.reshape(-1, A.NS).std(0) + 1e-6
    Xs = ((Xs - mu_s) / sd_s).astype(np.float32)
    Xst = Xs.reshape(len(Xs), -1)

    # 整任务一个 VAE：全部帧都参与训练（不再按类别分）
    print(f"\n训练**整任务** VAE（{len(Xs)} 帧，{a.iters} 步，不分类别）...")
    vae, npar = A.train_vae(Xi, Xt, Xs, Xst, a.iters, a.beta, dev, seed=0, log_every=a.iters)
    print(f"参数量 {npar}")

    # 编码：取均值 μ（确定性，不用采样）
    mus = []
    with torch.no_grad():
        for i in range(0, len(Xi), 512):
            sl = slice(i, i + 512)
            img = A.prep_img(Xi[sl], dev)
            s = torch.from_numpy(Xs[sl]).float().to(dev)
            mu, lv, zz, ri, rs = vae(img, s, len(Xi[sl]))
            mus.append(mu.cpu().numpy())
    MU = np.concatenate(mus).astype(np.float32)
    print(f"潜变量 {MU.shape}")

    # 逐帧非一致性分数（可选：用于看"典型/不典型"）
    sc = A.nll_of(vae, Xi, Xt, Xs, Xst, dev, a.beta)

    np.savez(a.out, mu=MU, y=Y, ep=EP, fidx=FIDX, score=sc,
             caps=np.array(caps, dtype=object),
             cap_labels=np.array(cap_lab, dtype=np.int64),
             params=npar, iters=a.iters, k=a.k)
    print(f"写出 {a.out}")


if __name__ == "__main__":
    main()
