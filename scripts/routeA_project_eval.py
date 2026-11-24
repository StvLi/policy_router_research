#!/usr/bin/env python3
"""把 VLM-VLA 联合评测的 rollout 投影到路线 A 的任务级潜空间（供交互页的"评测"栏使用）

背景与限制（必须随结果一起引用）：
  评测产物只有 **head 相机视频 + success/fail 文件名**，没有保存关节状态，也没有左右腕视角。
  所以这里做**近似投影**：
    1. 用真实 head 画面；
    2. 左右腕画面与 t/t-12/t-50 关节状态，用「head 画面最相似的演示帧」补齐（最近邻插补）；
    3. 过同一个 task-level VAE，再用**只在演示数据上拟合**的 PCA 基投影。
  这个近似会让评测点向"看起来像的那批演示帧"收缩，因此**不能**用它判断"评测偏离了演示分布"。
  用途仅限：看评测轨迹在演示流形上大致走哪条路径，以及成功/失败是否落在不同区域。

用法（6c，labeler 环境）：
  CUDA_VISIBLE_DEVICES=4 $L routeA_project_eval.py --videos /tmp/eval_clips
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/tmp")
import pilot_store_laptop as A          # 路线 A 的冻结配置

CACHE = "/tmp/pilot_frames_single_s5_k0_nall.npz"


def load_eval_clips(root, stride):
    """扫描 no_vlm / with_vlm 两个配置下的 mp4，返回 [(path, config, success)]。"""
    out = []
    for cfg_dir in sorted(glob.glob(os.path.join(root, "*", "store_laptop_and_headphones"))):
        cfg = os.path.basename(os.path.dirname(cfg_dir))
        for p in sorted(glob.glob(os.path.join(cfg_dir, "*.mp4"))):
            name = os.path.basename(p)
            ok = "_success" in name
            layout = name.split("_")[1] if name.startswith("episode_") else "?"
            out.append((p, cfg, ok, layout))
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
    """粗描述子：每视角 4x5 网格的 RGB 均值 → 60 维，L2 归一化。用于最近邻插补。"""
    import cv2
    N = len(imgs)
    d = np.zeros((N, 4 * 5 * 3), np.float32)
    for i, im in enumerate(imgs):
        small = cv2.resize(im, (5, 4), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        d[i] = small.reshape(-1)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--dataset", default=A.DEFAULT_DATASET)
    ap.add_argument("--videos", default="/tmp/eval_clips")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--state-weight", type=float, default=4.0)
    ap.add_argument("--out", default="/tmp/routeA_eval_latent.npz")
    a = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A.set_hist("vlm3"); A.set_img_hist("single"); A.set_pred_k(0)
    A.STATE_W = a.state_weight
    A.check_prep_img()

    # ---- 演示帧（训练数据）----
    z = np.load(a.cache, allow_pickle=True)
    Xi, Xt, EP, FIDX, CI = z["Xi"], z["Xt"], z["EP"], z["FIDX"], z["CI"]
    st = A.load_all_states(a.dataset, np.unique(EP))
    Xtr = A.state_windows(st, EP, FIDX)
    mu_s = Xtr.reshape(-1, A.NS).mean(0); sd_s = Xtr.reshape(-1, A.NS).std(0) + 1e-6
    Xtr = ((Xtr - mu_s) / sd_s).astype(np.float32)
    Xtr_t = Xtr.reshape(len(Xtr), -1)
    print(f"演示帧 {len(Xi)}")

    # ---- 训练 task-level VAE（与看板一致：整任务、不分类）----
    print(f"训练 task-level VAE（{a.iters} 步）...")
    vae, npar = A.train_vae(Xi, Xt, Xtr, Xtr_t, a.iters, a.beta, dev, seed=0, log_every=a.iters)
    print(f"参数量 {npar}")

    def encode(imgs, states, states_t=None):
        mus = []
        with torch.no_grad():
            for i in range(0, len(imgs), 512):
                sl = slice(i, i + 512)
                img = A.prep_img(imgs[sl], dev)
                s = torch.from_numpy(states[sl]).float().to(dev)
                mu, _, _, _, _ = vae(img, s, len(imgs[sl]))
                mus.append(mu.cpu().numpy())
        return np.concatenate(mus).astype(np.float32)

    MU_TR = encode(Xi, Xtr)
    # PCA 基只由演示数据决定
    mean = MU_TR.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(MU_TR - mean, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    basis = Vt[:3].T
    print(f"演示集解释方差 PC1-3 = {ev[0]*100:.1f}/{ev[1]*100:.1f}/{ev[2]*100:.1f}%")

    # ---- 演示集粗描述子（用于最近邻插补）----
    DTR = descriptor(Xi[:, 0, 0])       # 演示帧的 head 视角（Xi: (N, NT, 视角, H, W, C)）
    print(f"演示描述子 {DTR.shape}")

    # ---- 评测 rollout ----
    clips = load_eval_clips(a.videos, a.stride)
    print(f"评测片段 {len(clips)} 个")
    Z, TT, CFG, OK, LAYOUT, EPI = [], [], [], [], [], []
    for n, (p, cfg, ok, layout) in enumerate(clips, 1):
        head = decode_head(p, a.stride)
        if head is None:
            continue
        T = len(head)
        f_idx = np.arange(T) * a.stride
        # 最近邻：为每个评测帧找最相似的演示帧，借用它的左右腕画面与状态窗口
        de = descriptor(head)
        sim = de @ DTR.T
        nn = sim.argmax(1)
        # 真实 head + 最近邻演示帧的左右腕；再补一个 NT 轴 → (T, 1, 3视角, H, W, 3)
        imgs = np.stack([np.stack([head[i], Xi[nn[i], 0, 1], Xi[nn[i], 0, 2]], axis=0)
                         for i in range(T)], axis=0).astype(np.uint8)
        imgs = imgs[:, None]
        states = Xtr[nn]
        MU = encode(imgs, states)
        Z.append((MU - mean) @ basis)
        TT.append(f_idx / A.FPS)
        CFG += [cfg] * T
        OK += [ok] * T
        LAYOUT += [layout] * T
        EPI += [n - 1] * T
        if n % 10 == 0 or n == len(clips):
            print(f"  投影 {n}/{len(clips)} 个片段，累计 {sum(len(x) for x in Z)} 帧")

    if not Z:
        sys.exit("没有可用的评测片段")
    Z = np.concatenate(Z); TT = np.concatenate(TT)
    np.savez(a.out, z3=Z.astype(np.float32), t=TT.astype(np.float32),
             cfg=np.array(CFG, dtype=object), ok=np.array(OK, bool),
             layout=np.array(LAYOUT, dtype=object), epi=np.array(EPI, np.int32),
             pc_var=ev[:3], train_mean=mean.astype(np.float32),
             train_mu=MU_TR.astype(np.float32))
    print(f"写出 {a.out}  评测帧 {len(Z)}；成功 {int(np.sum(OK))} / 失败 {int(len(OK)-np.sum(OK))}")


if __name__ == "__main__":
    main()
