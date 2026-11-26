#!/usr/bin/env python3
"""导出路线 C 不同隐空间维度下的 latent 状态，供可视化。

对每个 latent ∈ {32,16,8,4}：
  1. 在演示集（本任务 100 集 / 10590 帧）上训练 VAE（隐藏层 2048→512→256→latent）
  2. 用 encoder 把三份数据编码成 latent：
       demo —— 本任务演示（ID）
       ood  —— 另一个任务（Arrange the numbers…）
       eval —— 本任务 8 局真实评测（含成功 1 局）
  3. 拟合 PCA（在 demo 上拟合，投影全部），给出 3 维坐标方便直接画
  4. 存 encoder 权重 + 每份数据的 npz（Z / PCA / 元信息）

输出（默认 /home/syt/e2/export）：
  latent{N}_vae.pt          encoder/decoder 全部权重 + 标准化 mu/sd + 配置
  latent{N}_demo.npz        Z(N,d) PC3(N,3) EP T CAP(阶段标签)
  latent{N}_ood.npz         Z PC3 EP T
  latent{N}_eval.npz        Z PC3 EP T SUCCESS SCORE PROGRESS
  export_index.json         汇总索引（哪档 latent、多少帧、AUROC 之类）
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routeC_latent_sweep import train_one  # noqa: E402


def phase_of(caption):
    c = caption.lower()
    if "headphone" in c:
        return "hang_headphones" if ("hang" in c or "stand" in c or "hook" in c or "rack" in c) else "pick_headphones"
    if "laptop" in c and ("close" in c or "lid" in c):
        return "close_laptop"
    if "laptop" in c and ("pick" in c or "lift" in c or "grasp" in c):
        return "pick_laptop"
    if "hand over" in c or "handover" in c or "transfer" in c:
        return "handover"
    if "place" in c or "put" in c:
        return "place_into_stand"
    return "other"


def demo_phase_labels(spec_path, EP, T, fps=25):
    """按 spec 的分段（秒）给每个采样帧打阶段标签。"""
    if not spec_path or not os.path.exists(spec_path):
        return np.array([""] * len(EP), dtype=object)
    spec = json.load(open(spec_path))
    segs = {int(k): v for k, v in spec["segs"].items()}
    out = []
    for ep, t in zip(EP.tolist(), T.tolist()):
        ss = segs.get(ep)
        if not ss:
            out.append("unknown")
            continue
        ts = t / fps
        lab = "boundary"
        for s in ss:
            if s["start_s"] <= ts < s["end_s"]:
                lab = phase_of(s["caption"])
                break
        else:
            lab = phase_of(ss[-1]["caption"]) if ts >= ss[-1]["start_s"] else "before_start"
        out.append(lab)
    return np.array(out, dtype=object)


def encode(model, E):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    En = (E - model["mu"]) / model["sd"]
    with torch.no_grad():
        z = model["vae"].enc(torch.from_numpy(En.astype(np.float32)).to(dev))
        return model["vae"].mu(z).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="/home/syt/e2/demo100.npz")
    ap.add_argument("--ood", default="/home/syt/e2/ood.npz")
    ap.add_argument("--eval", default="/home/syt/e2/eval.npz")
    ap.add_argument("--official", default="/home/syt/e2/eval_result.json")
    ap.add_argument("--spec", default="/home/syt/e2/spec_task_store_laptop.json")
    ap.add_argument("--latents", default="32,16,8,4")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/syt/e2/export")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    demo = np.load(a.demo); ood = np.load(a.ood); ev = np.load(a.eval)
    off = json.load(open(a.official))["details"]
    index = []

    for latent in [int(x) for x in a.latents.split(",")]:
        print(f"===== latent {latent} =====", flush=True)
        m = train_one(demo, latent, a.iters, 1.0, seed=a.seed)

        Zd = encode(m, demo["E"]); Zo = encode(m, ood["E"]); Ze = encode(m, ev["E"])
        # PCA 在 demo 上拟合，统一投影（OOD/eval 的偏移才有意义）
        Xc = Zd - Zd.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:3]
        var = (S ** 2) / max((S ** 2).sum(), 1e-12)

        def proj(Z):
            return ((Z - Zd.mean(0)) @ comps.T).astype(np.float32)

        torch.save(dict(state=m["vae"].state_dict(), mu=m["mu"], sd=m["sd"],
                        latent=latent, d_in=int(demo["E"].shape[1]), seed=a.seed,
                        pca_mean=Zd.mean(0).astype(np.float32),
                        pca_components=comps.astype(np.float32),
                        pca_explained=var[:3].astype(np.float32),
                        cal=m["cal"]),
                   os.path.join(a.out, f"latent{latent}_vae.pt"))

        np.savez_compressed(
            os.path.join(a.out, f"latent{latent}_demo.npz"),
            Z=Zd.astype(np.float32), PC3=proj(Zd), EP=demo["EP"], T=demo["T"],
            CAP=demo_phase_labels(a.spec, demo["EP"], demo["T"]).astype("<U24"))
        np.savez_compressed(
            os.path.join(a.out, f"latent{latent}_ood.npz"),
            Z=Zo.astype(np.float32), PC3=proj(Zo), EP=ood["EP"], T=ood["T"])
        succ = np.array([bool(off[str(int(e))]["success"]) for e in ev["EP"]])
        score = np.array([float(off[str(int(e))]["score"]) for e in ev["EP"]])
        # 进度：每个 episode 内归一化（该集自己的采样点序号 / 总点数）
        prog = np.zeros(len(ev["EP"]), np.float32)
        for e in set(ev["EP"].tolist()):
            idx = np.where(ev["EP"] == e)[0]
            prog[idx] = np.arange(len(idx)) / max(len(idx) - 1, 1)
        np.savez_compressed(
            os.path.join(a.out, f"latent{latent}_eval.npz"),
            Z=Ze.astype(np.float32), PC3=proj(Ze), EP=ev["EP"], T=ev["T"],
            SUCCESS=succ, SCORE=score, PROGRESS=prog.astype(np.float32))

        index.append(dict(latent=latent, seed=a.seed,
                          n_demo=int(len(Zd)), n_ood=int(len(Zo)), n_eval=int(len(Ze)),
                          pca_explained=[float(x) for x in var[:3]],
                          latent_std=float(Zd.std()),
                          files=dict(vae=f"latent{latent}_vae.pt",
                                     demo=f"latent{latent}_demo.npz",
                                     ood=f"latent{latent}_ood.npz",
                                     eval=f"latent{latent}_eval.npz")))
        print(f"  PCA 解释方差 {var[0]:.1%}/{var[1]:.1%}/{var[2]:.1%}；"
              f"latent 标准差 {Zd.std():.3f}", flush=True)

    json.dump(index, open(os.path.join(a.out, "export_index.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n写出 {a.out}/export_index.json")
    for r in index:
        print(f"  latent {r['latent']:2d}: demo {r['n_demo']} / ood {r['n_ood']} / eval {r['n_eval']} 帧，"
              f"PCA {r['pca_explained'][0]:.0%}+{r['pca_explained'][1]:.0%}+{r['pca_explained'][2]:.0%}")


if __name__ == "__main__":
    main()
