#!/usr/bin/env python3
"""一次性生成「训练 + 评测」两套潜空间坐标（**同一个 VAE、同一套 PCA 基**）

为什么必须合成一个脚本：
  之前 routeA_task_latent.py 和 routeA_project_eval.py 各自训练了一次 task-level VAE。
  即使配置与种子相同，GPU 非确定性会让两次权重有微小差异 → 两套潜空间、两套 PCA 基
  → 训练云与评测云落在不同坐标系里，看起来"轴没对齐"。这个脚本保证：
    只训练一次 VAE → 只在训练帧上拟合一次 PCA 基 → 两边用同一基投影。

评测侧的限制（不因本脚本而消失）：
  评测只存了 head 视频，没有左右腕、没有关节状态。
  这里用「head 画面最相似的演示帧」补齐左右腕与状态窗口，属**近似**；
  写出的 npz 里带 `approx=1` 标记，前端据此显示警告。

用法（6c，labeler 环境）：
  CUDA_VISIBLE_DEVICES=4 $L routeA_build_latents.py --videos /tmp/eval_clips \
      --out /tmp/routeA_latents.npz
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/tmp")
import pilot_store_laptop as A

CACHE = "/tmp/pilot_frames_single_s5_k0_nall.npz"
SPEC = "/tmp/pilot_spec.json"
EMB = "/tmp/pilot_emb.npz"


def load_eval_clips(root):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*", "store_laptop_and_headphones"))):
        cfg = os.path.basename(os.path.dirname(d))
        for p in sorted(glob.glob(os.path.join(d, "*.mp4"))):
            name = os.path.basename(p)
            out.append(dict(path=p, cfg=cfg, ok=("_success" in name),
                            layout=name.split("_")[1] if name.startswith("episode_") else "?"))
    return out


def decode_head(path, stride):
    import av
    import cv2
    frames = []
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i % stride:
                continue
            frames.append(cv2.resize(f.to_ndarray(format="rgb24"), (A.IMG_W, A.IMG_H),
                                     interpolation=cv2.INTER_AREA))
    return np.stack(frames) if frames else None


def descriptor(imgs):
    """head 画面的粗描述子（4x5 网格 RGB 均值，60 维，L2 归一化），用于最近邻插补。"""
    import cv2
    out = np.zeros((len(imgs), 4 * 5 * 3), np.float32)
    for i, im in enumerate(imgs):
        out[i] = (cv2.resize(im, (5, 4), interpolation=cv2.INTER_AREA)
                  .astype(np.float32) / 255.0).reshape(-1)
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--spec", default=SPEC)
    ap.add_argument("--emb", default=EMB)
    ap.add_argument("--dataset", default=A.DEFAULT_DATASET)
    ap.add_argument("--videos", default="/tmp/eval_clips")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--state-weight", type=float, default=4.0)
    ap.add_argument("--out", default="/tmp/routeA_latents.npz")
    a = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A.set_hist("vlm3"); A.set_img_hist("single"); A.set_pred_k(0)
    A.STATE_W = a.state_weight
    A.check_prep_img()

    # ---------------- 训练侧 ----------------
    z = np.load(a.cache, allow_pickle=True)
    Xi, Xt, EP, FIDX, CI = z["Xi"], z["Xt"], z["EP"], z["FIDX"], z["CI"]
    spec = json.load(open(a.spec))
    bundle = np.load(a.emb, allow_pickle=True)
    caps = list(bundle["caps"]); E = bundle["emb"]
    cap_lab, _ = A.spherical_kmeans(E, min(a.k, len(caps)))
    cap2clu = {c: int(l) for c, l in zip(caps, cap_lab)}
    Y = np.array([cap2clu[caps[i]] for i in CI], np.int64)

    sm = A.load_all_states(a.dataset, np.unique(EP))
    Xs = A.state_windows(sm, EP, FIDX)
    mu_s = Xs.reshape(-1, A.NS).mean(0); sd_s = Xs.reshape(-1, A.NS).std(0) + 1e-6
    Xs = ((Xs - mu_s) / sd_s).astype(np.float32)
    Xst = Xs.reshape(len(Xs), -1)
    print(f"训练帧 {len(Xi)}；类别分布 {dict(sorted(collections.Counter(Y.tolist()).items()))}")

    print(f"\n训练 task-level VAE（{a.iters} 步）...")
    vae, npar = A.train_vae(Xi, Xt, Xs, Xst, a.iters, a.beta, dev, seed=0, log_every=a.iters)
    print(f"参数量 {npar}")

    def encode(imgs, states):
        mus = []
        with torch.no_grad():
            for i in range(0, len(imgs), 512):
                sl = slice(i, i + 512)
                img = A.prep_img(imgs[sl], dev)
                s = torch.from_numpy(states[sl]).float().to(dev)
                mu, _, _, _, _ = vae(img, s, len(imgs[sl]))
                mus.append(mu.cpu().numpy())
        return np.concatenate(mus).astype(np.float32)

    MU_TR = encode(Xi, Xs)
    score_tr = A.nll_of(vae, Xi, Xt, Xs, Xst, dev, a.beta)

    # ---- 唯一一次 PCA 拟合（只在训练帧上）----
    mean = MU_TR.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(MU_TR - mean, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    basis = Vt[:3].T
    Z_TR = (MU_TR - mean) @ basis
    print(f"PCA 基（训练帧拟合）：解释方差 {ev[0]*100:.1f}/{ev[1]*100:.1f}/{ev[2]*100:.1f}%")

    # ---------------- 评测侧（同一 VAE、同一基）----------------
    DTR = descriptor(Xi[:, 0, 0])
    clips = load_eval_clips(a.videos)
    print(f"\n评测片段 {len(clips)} 个")
    EZ, ET, EEP, EOK, ECFG, ELAY = [], [], [], [], [], []
    for n, c in enumerate(clips, 1):
        head = decode_head(c["path"], a.stride)
        if head is None or len(head) < 2:
            continue
        f_idx = np.arange(len(head)) * a.stride
        nn = (descriptor(head) @ DTR.T).argmax(1)        # 最近的演示帧
        imgs = np.stack([np.stack([head[i], Xi[nn[i], 0, 1], Xi[nn[i], 0, 2]], axis=0)
                         for i in range(len(head))], axis=0).astype(np.uint8)[:, None]
        MU = encode(imgs, Xs[nn])
        EZ.append((MU - mean) @ basis)                    # ← 与训练用同一个 mean/basis
        ET.append(f_idx / A.FPS)
        EEP += [n - 1] * len(head)
        EOK += [c["ok"]] * len(head)
        ECFG += [0 if c["cfg"] == "no_vlm" else 1] * len(head)
        ELAY += [c["layout"]] * len(head)
        if n % 10 == 0 or n == len(clips):
            print(f"  投影 {n}/{len(clips)}，累计 {sum(len(x) for x in EZ)} 帧")

    EZ = np.concatenate(EZ); ET = np.concatenate(ET)
    EEP = np.asarray(EEP, np.int32); EOK = np.asarray(EOK, bool); ECFG = np.asarray(ECFG, np.uint8)
    ELAY = np.array(ELAY, dtype=object)
    rollouts = []
    for r in range(int(EEP.max()) + 1):
        m = EEP == r
        if not m.any():
            continue
        rollouts.append(dict(cfg=("no_vlm" if ECFG[m][0] == 0 else "with_vlm"),
                             ok=bool(EOK[m][0]), layout=str(ELAY[m][0]), n=int(m.sum())))
    # 统一 SANITY：两边都用同一基，训练点均值应接近 0
    print(f"\n训练点 PC 均值 {np.abs(Z_TR.mean(0)).round(3)}（应接近 0，说明基一致）")
    print(f"评测帧 {len(EZ)}；成功 {int(EOK.sum())} / 失败 {int((~EOK).sum())}")

    np.savez(a.out,
             # 训练
             tr_z3=Z_TR.astype(np.float32), tr_y=Y, tr_ep=EP, tr_t=(FIDX / A.FPS).astype(np.float32),
             tr_score=score_tr.astype(np.float32), pc_var=ev[:3],
             caps=np.array(caps, dtype=object), cap_labels=np.array(cap_lab, np.int64),
             npar=npar, k=a.k,
             tr_phi=np.array([0.0]),  # 占位，前端自己按 PC1-PC2 算相位
             # 评测
             ev_z3=EZ.astype(np.float32), ev_t=ET.astype(np.float32),
             ev_ep=EEP, ev_ok=EOK, ev_cfg=ECFG, ev_layout=ELAY,
             ev_rollouts=np.array([json.dumps(r, ensure_ascii=False) for r in rollouts], dtype=object),
             approx=np.array(1.0))
    print(f"写出 {a.out}")


if __name__ == "__main__":
    main()
