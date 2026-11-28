#!/usr/bin/env python3
"""把 VLA rollout 的表征投影进**该任务自己的 VAE 潜空间**，算逐帧共形 p，供页面右栏使用。

对每个任务：
  1. 用与 worker 完全相同的设置重训 VAE（seed 0 / 6k iters，确定性）；
  2. demo 帧 → Z → PCA 基（供左栏 3D）；
  3. rollout 帧 → 同一个 VAE 编码 → LOF 统计量（参考集 = demo 拟合集）→ 共形 p（校准集 = demo 的 25%）；
  4. 写 results/page/<task>/eval.npz：pc3(投影到 demo 的 PCA 基) / ep(布局号) / t / p / ok / score

用法（工作站，可循环等新 rollout）：
  python3 project_rollouts.py           # 处理所有已就绪任务
  python3 project_rollouts.py --watch   # 每 5 分钟轮询，直到全部完成
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_routec_handover_js import knn_dist, lof_stat  # noqa: E402
from routeC_vae_failure import build_vae  # noqa: E402


def train_vae(E, latent=32, iters=6000, seed=0, dev="cuda"):
    import torch
    import torch.nn.functional as F
    mu, sd = E.mean(0), E.std(0) + 1e-6
    En = ((E - mu) / sd).astype(np.float32)
    torch.manual_seed(seed)
    vae = build_vae(E.shape[1], latent).to(dev)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    X = torch.from_numpy(En).to(dev)
    bs = min(64, len(X))
    for _ in range(iters):
        idx = torch.randint(0, len(X), (bs,), device=dev)
        x = X[idx]
        mu_, lv, dec, _ = vae(x)
        loss = F.mse_loss(dec(mu_), x) + \
            (-0.5 * torch.sum(1 + lv - mu_.pow(2) - lv.exp(), dim=1).mean() / mu_.shape[1])
        opt.zero_grad(); loss.backward(); opt.step()
    vae.eval()
    return vae, mu, sd


def encode(vae, E, mu, sd):
    import torch
    with torch.no_grad():
        z = vae.mu(vae.enc(torch.from_numpy(((E - mu) / sd).astype(np.float32)).cuda()))
    return z.cpu().numpy().astype(np.float32)


def result_json_for(task, eval_root):
    """该任务最新一次 run 的 _result.json → {layout_id: (ok, score)}。"""
    best = None
    for d in glob.glob(os.path.join(eval_root, task, "*", "*", "*", "*")):
        p = os.path.join(d, "_result.json")
        if os.path.isfile(p) and len(glob.glob(os.path.join(d, "episode_*_cam_head_*.mp4"))) >= 20:
            if best is None or os.path.getmtime(p) > best[0]:
                best = (os.path.getmtime(p), p)
    if not best:
        return {}
    det = json.load(open(best[1])).get("details", {})
    return {int(v["layout_id"]): (bool(v["success"]), float(v.get("score", 0.0)))
            for v in det.values()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", default="/home/admin/e2rc/latent")
    ap.add_argument("--rollouts", default="/home/admin/e2rc/rollouts")
    ap.add_argument("--out", default="/home/admin/e2rc/results/page")
    ap.add_argument("--eval-root", default="/home/syt/rdojo_ws/eval_result/RoboDojo")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    while True:
        ready = []
        for rp in sorted(glob.glob(os.path.join(a.rollouts, "*.npz"))):
            task = os.path.basename(rp)[:-4]
            if not os.path.exists(os.path.join(a.latent, task, "demo.npz")):
                continue
            if os.path.exists(os.path.join(a.out, task, "eval.npz")):
                continue
            ready.append((task, rp))
        if ready:
            print(f"待投影 {len(ready)} 个任务", flush=True)
        for task, rp in ready:
            t0 = time.time()
            try:
                d = np.load(os.path.join(a.latent, task, "demo.npz"))
                E, EP, T = d["E"], d["EP"], d["T"]
                eps = np.unique(EP)
                rng = np.random.default_rng(0)
                perm = rng.permutation(len(eps))
                n_cal = max(1, int(round(len(eps) * 0.25)))
                cal_eps = set(eps[perm[:n_cal]].tolist())
                is_cal = np.isin(EP, list(cal_eps))

                vae, mu, sd = train_vae(E, a.latent_dim, a.iters)
                Z = encode(vae, E, mu, sd)

                # demo 的 PCA 基（左栏 3D，rollout 也投影到同一基）
                Zm = Z.mean(0, keepdims=True)
                _, sv, Vt = np.linalg.svd((Z - Zm).astype(np.float64), full_matrices=False)
                basis = Vt[:3].T
                pc3_demo = (Z - Zm) @ basis

                Zf, EPf = Z[~is_cal], EP[~is_cal]
                Zc = Z[is_cal]
                dk = knn_dist(Zf, EPf, Zf, EPf)
                s_cal = lof_stat(Zc, EP[is_cal], Zf, EPf, dk)

                r = np.load(rp)
                Er, EPr, Tr = r["E"], r["EP"], r["T"]
                Zr = encode(vae, Er, mu, sd)
                s_r = lof_stat(Zr, EPr, Zf, EPf, dk)
                p_r = (1.0 + (s_cal[None, :] >= s_r[:, None]).sum(1)) / (len(s_cal) + 1.0)
                pc3_r = (Zr - Zm) @ basis

                jmap = result_json_for(task, a.eval_root)
                ok = np.array([1 if jmap.get(int(e), (False, 0))[0] else 0 for e in EPr], np.uint8)
                sc = np.array([jmap.get(int(e), (False, 0.0))[1] for e in EPr], np.float32)

                od = os.path.join(a.out, task)
                os.makedirs(od, exist_ok=True)
                np.savez_compressed(os.path.join(od, "eval.npz"),
                                    pc3=pc3_r.astype(np.float32), ep=EPr, t=Tr,
                                    p=p_r.astype(np.float32), ok=ok, score=sc,
                                    demo_pc3=pc3_demo.astype(np.float32))
                print(f"[✓] {task}: rollout {len(Er)} 帧 / {len(set(EPr.tolist()))} 局 · "
                      f"成功 {int(ok.sum())} 帧 · p 中位 {np.median(p_r):.3f} · {time.time()-t0:.0f}s",
                      flush=True)
            except Exception as ex:
                print(f"[!] {task}: {type(ex).__name__}: {ex}", flush=True)
        if not a.watch:
            break
        done = len(glob.glob(os.path.join(a.out, "*", "eval.npz")))
        print(f"... 已投影 {done} 个任务，5 分钟后继续", flush=True)
        time.sleep(300)


if __name__ == "__main__":
    main()
