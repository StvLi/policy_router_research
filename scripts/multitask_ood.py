#!/usr/bin/env python3
"""L1：逐任务 VAE + 共形 p，检验"每个任务一个 VAE"能不能把本任务认成正常、把别的任务认成异常。

对每个任务 t：
  1. 用它自己的 100 集 demo 训练一个 VAE（与交接包同结构、同迭代数）；
  2. 按集 75/25 切分：拟合集（参考集）与校准集；
  3. 统计量 = 局部密度归一化 kNN（与线上方案 v2 一致）；
  4. p = 分裂共形（校准集）；
  5. 评测：
       ID  = 该任务**留出集**的帧（应该被判正常）
       OOD = 其它任务的帧，**经本任务 VAE 编码**后算同一个统计量

指标：AUROC(ID↔OOD)、ID 上 α 处误报率、OOD 上 α 处检出率、以及两者的间隔。
所有结果按任务落盘 JSON，可直接汇总。

用法（6 卡机，抽取完成后）：
  python3 multitask_ood.py --latent /root/rtc/latent --out /root/rtc/results
  python3 multitask_ood.py --tasks 5 --dims 32          # 先跑 5 个任务看看
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_routec_handover_js import lof_p, knn_dist, lof_stat  # noqa: E402
from routeC_vae_failure import build_vae, conformal_p, vae_nll  # noqa: E402


def train_vae(E, EP, iters=6000, latent=32, seed=0, device="cpu"):
    import torch
    import torch.nn.functional as F
    mu, sd = E.mean(0), E.std(0) + 1e-6
    En = (E - mu) / sd
    torch.manual_seed(seed)
    vae = build_vae(E.shape[1], latent).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    X = torch.from_numpy(En).float().to(device)
    bs = min(64, len(X))
    for _ in range(iters):
        idx = torch.randint(0, len(X), (bs,), device=device)
        x = X[idx]
        mu_, lv, dec, _ = vae(x)
        rec = dec(mu_)
        mse = F.mse_loss(rec, x)
        kl = -0.5 * torch.sum(1 + lv - mu_.pow(2) - lv.exp(), dim=1).mean() / mu_.shape[1]
        loss = mse + kl
        opt.zero_grad(); loss.backward(); opt.step()
    vae.eval()
    return vae, mu, sd


def encode(vae, E, mu, sd):
    import torch
    with torch.no_grad():
        z = vae.mu(vae.enc(torch.from_numpy(((E - mu) / sd).astype(np.float32))))
    return z.numpy().astype(np.float32)


def auroc(neg, pos):
    """neg=ID 分数（小=正常），pos=OOD 分数；返回 AUROC（越大=分得越开）。"""
    a = np.asarray(neg, np.float64)
    b = np.asarray(pos, np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    # 秩方法，处理并列
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 并列取平均秩
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r_pos = ranks[len(a):].sum()
    return float((r_pos - len(b) * (len(b) + 1) / 2) / (len(a) * len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", default="/root/rtc/latent")
    ap.add_argument("--out", default="/root/rtc/results")
    ap.add_argument("--dims", type=int, default=32)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.12)
    ap.add_argument("--tasks", type=int, default=0, help="只跑前 N 个任务（0=全部）")
    ap.add_argument("--ood-frames", type=int, default=3000,
                    help="每个任务从其它任务抽多少帧当 OOD")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    dev = a.device if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    dirs = sorted(glob.glob(os.path.join(a.latent, "*", "demo.npz")))
    if a.tasks:
        dirs = dirs[:a.tasks]
    print(f"{len(dirs)} 个任务，VAE d={a.dims}，device={dev}", flush=True)

    # 先把所有任务的 E 读进来（OOD 要用原始 2048 维喂给每个任务的 VAE）
    data = {}
    for p in dirs:
        d = np.load(p)
        data[os.path.basename(os.path.dirname(p))] = (d["E"], d["EP"], d["T"])
    names = list(data)

    for i, name in enumerate(names):
        out_json = os.path.join(a.out, f"{name}.json")
        if os.path.exists(out_json):
            continue
        E, EP, T = data[name]
        eps = np.unique(EP)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(eps))
        n_cal = max(1, int(round(len(eps) * 0.25)))
        cal_eps = set(eps[perm[:n_cal]].tolist())
        is_cal = np.isin(EP, list(cal_eps))

        vae, mu, sd = train_vae(E, EP, a.iters, a.dims, a.seed, dev)
        Z = encode(vae, E, mu, sd)

        # 参考集 = 拟合集（非校准集）
        Zf, EPf = Z[~is_cal], EP[~is_cal]
        Zc, EPc = Z[is_cal], EP[is_cal]
        dk_ref = knn_dist(Zf, EPf, Zf, EPf)
        s_cal = lof_stat(Zc, EPc, Zf, EPf, dk_ref)
        p_cal = (1.0 + (s_cal[None, :] >= s_cal[:, None]).sum(1)) / (len(s_cal) + 1.0)

        # ID 测试 = 校准集本身（留一集算法已排除同集），OOD = 其它任务抽样
        p_id = p_cal
        others = [n for n in names if n != name]
        rng2 = np.random.default_rng(1)
        ood_E = []
        for o in rng2.choice(others, size=min(3, len(others)), replace=False):
            Eo = data[o][0]
            k = min(a.ood_frames // 3, len(Eo))
            ood_E.append(Eo[rng2.choice(len(Eo), size=k, replace=False)])
        E_ood = np.concatenate(ood_E)
        Z_ood = encode(vae, E_ood, mu, sd)
        s_ood = lof_stat(Z_ood, np.full(len(Z_ood), -1), Zf, EPf, dk_ref)
        p_ood = (1.0 + (s_cal[None, :] >= s_ood[:, None]).sum(1)) / (len(s_cal) + 1.0)

        rec = dict(task=name, n_frames=int(len(E)), n_episodes=int(len(eps)),
                   latent=a.dims, iters=a.iters, seed=a.seed, alpha=a.alpha,
                   auroc=auroc(s_cal, s_ood),
                   fpr=float((p_id < a.alpha).mean()),
                   tpr=float((p_ood < a.alpha).mean()),
                   med_p_id=float(np.median(p_id)), med_p_ood=float(np.median(p_ood)))
        json.dump(rec, open(out_json, "w"), ensure_ascii=False, indent=1)
        print(f"[{i+1}/{len(names)}] {name}: AUROC {rec['auroc']:.3f}  "
              f"FPR {rec['fpr']*100:.1f}%  TPR {rec['tpr']*100:.1f}%", flush=True)

    print("完成", flush=True)


if __name__ == "__main__":
    main()
