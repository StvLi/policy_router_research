#!/usr/bin/env python3
"""路线 C · 隐空间维度扫描：latent ∈ {32, 16, 8, 4}，比较 OOD 鉴别与失败检出。

动机：若约 1 万条样本在 32 维隐空间上就能拉出近似正态、且 p≈0.12 可有效鉴别 OOD，
说明有效自由度可能远低于 32——VAE 的隐空间维度值得当超参扫。

三组数据（都用冻结 VLA 的 2048 维表征）：
  ID   ：本任务演示集（按集 75/25 划分，训练 / 留出校准+测试）
  OOD  ：另一个任务（不同任务，同机器人）
  EVAL ：本任务 8 局真实评测（含成功 1 局、彻底失败 4 局）

每个 latent 报四个数：
  1. AUROC(ID 留出 vs OOD)                    —— OOD 鉴别力
  2. 逐集分数 vs 官方部分得分的 Spearman       —— 失败检出（排序）
  3. 中断规则触发情况（滑窗 1 秒 ≥3 点 p<0.05）—— 能否打断
  4. 校准集上 p 的分布离正态有多远（偏度/峰度） —— 隐空间的"正态性"

用法：
  python3 routeC_latent_sweep.py --demo demo100.npz --ood ood.npz --eval eval.npz \
      --official eval_result.json --out sweep.json --latents 32,16,8,4
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routeC_vae_failure import build_vae, vae_nll  # noqa: E402


def auroc_id_vs_ood(id_score, ood_score):
    """返回 P(ID 分数 < OOD 分数)：1.0 = 完美分离（ID 分数更低）。"""
    pos = np.asarray(id_score); neg = np.asarray(ood_score)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # 并列取平均秩
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[:len(pos)].sum()
    u = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))   # P(ID 排名更高)
    return float(1.0 - u)                                                  # P(ID 分数更低)


def moments(x):
    x = np.asarray(x, float)
    m, s = x.mean(), x.std() + 1e-12
    z = (x - m) / s
    return dict(mean=float(m), std=float(s), skew=float((z ** 3).mean()),
                kurtosis=float((z ** 4).mean() - 3.0))


def train_one(demo, latent, iters, beta, seed=0):
    import torch
    import torch.nn.functional as F
    E, EP = demo["E"].astype(np.float32), demo["EP"]
    mu, sd = E.mean(0), E.std(0) + 1e-6
    En = (E - mu) / sd
    eps = sorted(set(EP.tolist()))
    rng = np.random.default_rng(seed); rng.shuffle(eps)
    n_val = max(1, int(round(len(eps) * 0.25)))
    val = set(eps[:n_val])
    tr = np.array([e not in val for e in EP])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    vae = build_vae(E.shape[1], latent).to(dev)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    X = torch.from_numpy(En[tr]).float().to(dev)
    bs = min(128, len(X))
    for it in range(1, iters + 1):
        idx = torch.randint(0, len(X), (bs,), device=dev)
        x = X[idx]
        mu_, lv, dec, _ = vae(x)
        rec = dec(mu_)
        mse = F.mse_loss(rec, x)
        kl = -0.5 * torch.sum(1 + lv - mu_.pow(2) - lv.exp(), dim=1).mean() / mu_.shape[1]
        loss = mse + beta * kl
        opt.zero_grad(); loss.backward(); opt.step()
    vae.eval()
    with torch.no_grad():
        cal = vae_nll(vae, torch.from_numpy(En[~tr]).float().to(dev)).cpu().numpy()
        lat = vae.enc(torch.from_numpy(En).float().to(dev))
        lat_mu = vae.mu(lat).cpu().numpy()
    return dict(vae=vae, mu=mu, sd=sd, cal=cal, tr=tr, val=val,
                n_train=int(tr.sum()), n_cal=int((~tr).sum()),
                latent_moments=moments(lat_mu.reshape(-1)))


def score_with(model, E):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    En = (E - model["mu"]) / model["sd"]
    with torch.no_grad():
        sc = vae_nll(model["vae"], torch.from_numpy(En.astype(np.float32)).float().to(dev)).cpu().numpy()
    return sc


def pvals(score, cal):
    return (1.0 + (cal[None, :] >= score[:, None]).sum(1)) / (len(cal) + 1.0)


def interrupt(ep_arr, p, alpha=0.05, win=5, need=3):
    """返回每集首次触发（窗口起点，采样点序号）与轨迹占比。"""
    out = {}
    for ep in sorted(set(ep_arr.tolist())):
        idx = np.where(ep_arr == ep)[0]
        flag = p[idx] < alpha
        n = len(flag)
        fire = -1
        for i in range(n - win + 1):
            if flag[i:i + win].sum() >= need:
                fire = i
                break
        out[int(ep)] = dict(n=int(n), fire=int(fire),
                            fire_frac=(fire / max(n - 1, 1)) if fire >= 0 else None,
                            flag_rate=float(flag.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="/home/syt/e2/demo100.npz")
    ap.add_argument("--ood", default="/home/syt/e2/ood.npz")
    ap.add_argument("--eval", default="/home/syt/e2/eval.npz")
    ap.add_argument("--official", default="/home/syt/e2/eval_result.json")
    ap.add_argument("--latents", default="32,16,8,4")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/syt/e2/sweep.json")
    a = ap.parse_args()

    demo = np.load(a.demo)
    ood = np.load(a.ood) if a.ood and os.path.exists(a.ood) else None
    ev = np.load(a.eval)
    off = json.load(open(a.official))["details"]
    print(f"ID {demo['E'].shape} / OOD {ood['E'].shape if ood is not None else '—'} / "
          f"EVAL {ev['E'].shape}")

    results = []
    for latent in [int(x) for x in a.latents.split(",")]:
        print(f"\n===== latent = {latent} =====", flush=True)
        m = train_one(demo, latent, a.iters, a.beta, seed=a.seed)
        rec = dict(latent=latent, seed=a.seed, n_train=m["n_train"], n_cal=m["n_cal"],
                   latent_moments=m["latent_moments"])

        # ID 留出（校准集就是测试集：路线 A 的口径，标注在文档里）
        id_score = score_with(m, demo["E"][~m["tr"]])
        rec["id_score_median"] = float(np.median(id_score))

        if ood is not None:
            ood_score = score_with(m, ood["E"])
            rec["ood_score_median"] = float(np.median(ood_score))
            rec["auroc_id_vs_ood"] = auroc_id_vs_ood(id_score, ood_score)

        # 评测集：失败检出 + 中断
        escore = score_with(m, ev["E"])
        ep, cal = ev["EP"], m["cal"]
        p = pvals(escore, cal)
        rec["eval_flag_rate"] = float((p < a.alpha).mean())
        rec["interrupt"] = interrupt(ep, p, a.alpha)
        rows = []
        for e in sorted(set(ep.tolist())):
            mm = ep == e
            rows.append(dict(ep=int(e), n=int(mm.sum()), score_mean=float(escore[mm].mean()),
                             p_median=float(np.median(p[mm])),
                             flag_rate=float((p[mm] < a.alpha).mean()),
                             official=float(off[str(e)]["score"])))
        rec["episodes"] = rows
        xs = np.array([r["score_mean"] for r in rows]); ys = np.array([r["official"] for r in rows])
        rec["spearman_vs_official"] = float(np.corrcoef(np.argsort(np.argsort(xs)),
                                                        np.argsort(np.argsort(ys)))[0, 1])
        # 中断：彻底失败的集(官方 0)是否触发；好集(>=0.8)是否误触发
        bad = [r["ep"] for r in rows if r["official"] == 0.0]
        good = [r["ep"] for r in rows if r["official"] >= 0.8]
        rec["interrupt_bad"] = f"{sum(1 for e in bad if rec['interrupt'][e]['fire'] >= 0)}/{len(bad)}"
        rec["interrupt_good_false"] = f"{sum(1 for e in good if rec['interrupt'][e]['fire'] >= 0)}/{len(good)}"
        results.append(rec)

        print(f"  AUROC(ID vs OOD) = {rec.get('auroc_id_vs_ood')}")
        print(f"  Spearman(分数, 官方得分) = {rec['spearman_vs_official']:+.3f}")
        print(f"  彻底失败触发 {rec['interrupt_bad']}，良好误触发 {rec['interrupt_good_false']}")
        print(f"  隐空间激活 偏度 {m['latent_moments']['skew']:+.2f} 峰度 {m['latent_moments']['kurtosis']:+.2f}")

    json.dump(results, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n写出 {a.out}")
    print(f"\n{'latent':>6} {'AUROC':>7} {'Spearman':>9} {'bad触发':>8} {'good误触':>8} {'峰度':>6}")
    for r in results:
        print(f"{r['latent']:6d} {r.get('auroc_id_vs_ood', float('nan')):7.3f} "
              f"{r['spearman_vs_official']:+9.3f} {r['interrupt_bad']:>8} "
              f"{r['interrupt_good_false']:>8} {r['latent_moments']['kurtosis']:+6.2f}")


if __name__ == "__main__":
    main()
