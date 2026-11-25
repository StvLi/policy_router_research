#!/usr/bin/env python3
"""路线 C · E2：在冻结 VLA 表征上训 VAE，检验能否检出失败并触发打断。

三个 stage（都在 ws 上跑，需要 openpi venv + 检查点）：

  extract  抽 embedding：演示集（ep0031xx，正常执行）或评测集（含失败）
  train    在演示 embedding 上训 VAE（输入 embedding → 复原 embedding），留出集做共形校准
  score    给目标 embedding 打分：逐帧 p 值、逐集均值、与官方部分得分的秩相关

判定口径（沿用路线 A 的两级拒识，但只关心"任务内脱轨 → 打断拉 CAP"）：
  p < 0.05 视为"该帧不像正常执行"。逐集统计被标帧比例；比例高 = 该集应被中断。

用法：
  python3 routeC_vae_failure.py extract --source demo --episodes 3100-3119 --out /tmp/e2/demo.npz
  python3 routeC_vae_failure.py extract --source eval --episodes 2,8,11,12,13,17,19,24 --out /tmp/e2/eval.npz
  python3 routeC_vae_failure.py train --demo /tmp/e2/demo.npz --out /tmp/e2/vae.pt
  python3 routeC_vae_failure.py score --model /tmp/e2/vae.pt --emb /tmp/e2/eval.npz --out /tmp/e2/eval_score.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

OPENPI_SRC = "/srv/vla_shared/robodojo/RoboDojo/XPolicyLab/policy/Pi_05/openpi/src"
PROMPT = ("Hang the headphones on the headphone stand, close the laptop, "
          "then place it into the vertical laptop stand.")


# ---------------------------------------------------------------- 抽特征
def load_policy(args):
    sys.path.insert(0, OPENPI_SRC)
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import normalize as _normalize
    from openpi.training import config as _config
    cfg = _config.get_config(args.train_config)
    ns = _normalize.load(os.path.join(args.ckpt, "assets", args.repo_id))
    return _policy_config.create_trained_policy(cfg, args.ckpt, norm_stats=ns)


def make_extractor(policy, args):
    import copy
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    mdl = policy._model

    def hidden_batch(triples):
        """triples: [(head_bgr, left_bgr, right_bgr), ...] → (B, 2048) 均值池化 hidden states。"""
        prepared = []
        for head, left, right in triples:
            images = {k: np.ascontiguousarray(np.transpose(v, (2, 0, 1)))
                      for k, v in (("cam_high", head), ("cam_left_wrist", left),
                                   ("cam_right_wrist", right))}
            obs = dict(state=np.zeros(14, np.float32), images=images, prompt=PROMPT)
            prepared.append(policy._input_transform(copy.deepcopy(obs)))
        stacked = jax.tree.map(lambda *xs: jnp.asarray(np.stack([np.asarray(x) for x in xs])),
                               *prepared)
        observation = _model.Observation.from_dict(stacked)
        toks, mask, ar = mdl.embed_prefix(observation)
        out, _ = mdl.PaliGemma.llm([toks, None], mask=make_attn_mask(mask, ar),
                                   positions=jnp.cumsum(mask, axis=1) - 1)
        hid = np.asarray(out[0])[:, :768]        # 三路图像 token（各 256）
        return hid.mean(1)                        # (B, 2048) 全图像 token 均值

    return hidden_batch


def iter_demo(args):
    """演示集：ep0031xx/{main,left_wrist,right_wrist}.mp4"""
    for name in args.episodes.split(","):
        name = name.strip()
        if "-" in name:
            lo, hi = name.split("-")
            eps = list(range(int(lo), int(hi) + 1))
        else:
            eps = [int(name)]
        for ep in eps:
            d = os.path.join(args.demo_clips, f"ep{ep:06d}")
            if not os.path.isdir(d):
                print(f"  [skip] 缺 {d}")
                continue
            yield ep, (os.path.join(d, "main.mp4"), os.path.join(d, "left_wrist.mp4"),
                       os.path.join(d, "right_wrist.mp4"))


def iter_eval(args, eval_dir):
    """评测集：目录下 episode_XXXXXXX_cam_{head,left_wrist,right_wrist}_*.mp4"""
    want = None
    if args.episodes:
        want = set()
        for name in args.episodes.split(","):
            name = name.strip()
            if "-" in name:
                lo, hi = name.split("-")
                want |= set(range(int(lo), int(hi) + 1))
            else:
                want.add(int(name))
    for f in sorted(glob.glob(os.path.join(eval_dir, "episode_*_cam_head_*.mp4"))):
        ep = int(os.path.basename(f).split("_")[1])
        if want is not None and ep not in want:
            continue
        def find(cam):
            hits = glob.glob(os.path.join(eval_dir, f"episode_{ep:07d}_cam_{cam}_*.mp4"))
            return hits[0] if hits else None
        l, r = find("left_wrist"), find("right_wrist")
        if l is None or r is None:
            continue
        yield ep, (f, l, r)


def stage_extract(a):
    import cv2
    policy = load_policy(a)
    hidden_batch = make_extractor(policy, a)
    if a.source == "demo":
        it = iter_demo(a)
    else:
        it = iter_eval(a, a.eval_dir)

    FE, EP, T = [], [], []
    for ep, (fh, fl, fr) in it:
        caps = [cv2.VideoCapture(p) for p in (fh, fl, fr)]
        ns = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
        n = min(ns)
        triples, ts = [], []
        for t in range(0, n, a.stride):
            frames = []
            ok_all = True
            for c in caps:
                c.set(cv2.CAP_PROP_POS_FRAMES, t)
                ok, fr_ = c.read()
                if not ok:
                    ok_all = False
                    break
                frames.append(cv2.cvtColor(fr_, cv2.COLOR_BGR2RGB))
            if not ok_all:
                continue
            triples.append(tuple(frames))
            ts.append(t)
        for c in caps:
            c.release()
        if not triples:
            print(f"  ep{ep}: 0 帧")
            continue
        for i in range(0, len(triples), a.batch):
            FE.append(hidden_batch(triples[i:i + a.batch]))
        EP += [ep] * len(triples)
        T += ts
        print(f"  ep{ep}: {len(triples)} 帧", flush=True)

    if not FE:
        raise SystemExit("没有抽到帧")
    E = np.concatenate(FE).astype(np.float32)
    np.savez_compressed(a.out, E=E, EP=np.asarray(EP), T=np.asarray(T))
    print(f"写出 {a.out}：{E.shape}")


# ---------------------------------------------------------------- VAE
def build_vae(d_in, d_lat, hidden=(512, 256)):
    import torch.nn as nn
    h1, h2 = hidden

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d_in, h1), nn.ReLU(),
                                     nn.Linear(h1, h2), nn.ReLU())
            self.mu = nn.Linear(h2, d_lat)
            self.lv = nn.Linear(h2, d_lat)
            self.dec = nn.Sequential(nn.Linear(d_lat, h2), nn.ReLU(),
                                     nn.Linear(h2, h1), nn.ReLU(),
                                     nn.Linear(h1, d_in))

        def forward(self, x):
            h = self.enc(x)
            return self.mu(h), self.lv(h), self.dec, h

    return VAE()


def vae_nll(vae, x, beta=1.0):
    """逐样本分数 = 重建 MSE + beta*KL（与路线 A 的口径一致）。"""
    import torch
    import torch.nn.functional as F
    mu, lv, dec, h = vae(x)
    h2 = h
    # 用 dec 的输入重建：这里直接调用 dec(mu)（VAE 均值路径，保证确定性）
    rec = dec(mu)
    mse = F.mse_loss(rec, x, reduction="none").mean(1)
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1) / mu.shape[1]
    return mse + beta * kl


def stage_train(a):
    import torch
    z = np.load(a.demo)
    E, EP = z["E"].astype(np.float32), z["EP"]
    print(f"演示 embedding {E.shape}，{len(set(EP.tolist()))} 集")

    mu = E.mean(0); sd = E.std(0) + 1e-6
    En = (E - mu) / sd
    eps = sorted(set(EP.tolist()))
    rng = np.random.default_rng(0); rng.shuffle(eps)
    n_val = max(1, int(round(len(eps) * 0.25)))
    val = set(eps[:n_val])
    tr = np.array([e not in val for e in EP])
    print(f"划分：训练 {len(eps)-n_val} 集 / 留出 {n_val} 集；帧 {tr.sum()} / {(~tr).sum()}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    vae = build_vae(E.shape[1], a.latent).to(dev)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    X = torch.from_numpy(En[tr]).float().to(dev)
    import torch.nn.functional as F
    bs = min(a.batch, len(X))
    for it in range(1, a.iters + 1):
        idx = torch.randint(0, len(X), (bs,), device=dev)
        x = X[idx]
        mu_, lv, dec, _ = vae(x)
        rec = dec(mu_)
        mse = F.mse_loss(rec, x)
        kl = -0.5 * torch.sum(1 + lv - mu_.pow(2) - lv.exp(), dim=1).mean() / mu_.shape[1]
        loss = mse + a.beta * kl
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"  it {it:5d}  mse {mse.item():.4f}  kl {kl.item():.4f}", flush=True)

    vae.eval()
    with torch.no_grad():
        valx = torch.from_numpy(En[~tr]).float().to(dev)
        cal = vae_nll(vae, valx).cpu().numpy()
    torch.save(dict(state=vae.state_dict(), mu=mu, sd=sd, latent=a.latent,
                    d_in=int(E.shape[1]), cal=cal, beta=a.beta), a.out)
    print(f"模型写出 {a.out}；校准帧 {len(cal)}，分数中位 {np.median(cal):.4f}")


def conformal_p(score, cal):
    return (1.0 + (cal[None, :] >= score[:, None]).sum(1)) / (len(cal) + 1.0)


def stage_score(a):
    import torch
    z = torch.load(a.model, map_location="cpu", weights_only=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae = build_vae(z["d_in"], z["latent"]).to(dev)
    vae.load_state_dict(z["state"]); vae.eval()
    cal = z["cal"]

    d = np.load(a.emb)
    E, EP, T = d["E"].astype(np.float32), d["EP"], d["T"]
    En = (E - z["mu"]) / z["sd"]
    with torch.no_grad():
        sc = vae_nll(vae, torch.from_numpy(En).float().to(dev)).cpu().numpy()
    p = conformal_p(sc, cal)

    rec = dict(n=int(len(E)), cal_median=float(np.median(cal)),
               score_median=float(np.median(sc)), p_median=float(np.median(p)),
               flag_rate=float((p < a.alpha).mean()), alpha=a.alpha, episodes=[])
    for ep in sorted(set(EP.tolist())):
        m = EP == ep
        rec["episodes"].append(dict(
            ep=int(ep), n=int(m.sum()),
            score_mean=float(sc[m].mean()), score_median=float(np.median(sc[m])),
            p_median=float(np.median(p[m])), flag_rate=float((p[m] < a.alpha).mean()),
            score_p90=float(np.percentile(sc[m], 90)),
        ))
    # 若有官方得分，做秩相关
    if a.official:
        off = json.load(open(a.official))["details"]
        for e in rec["episodes"]:
            e["official_score"] = float(off[str(e["ep"])]["score"])
        xs = np.array([e["score_mean"] for e in rec["episodes"]])
        ys = np.array([e["official_score"] for e in rec["episodes"]])
        rx = np.argsort(np.argsort(xs)); ry = np.argsort(np.argsort(ys))
        rec["spearman_vs_official"] = float(np.corrcoef(rx, ry)[0, 1])
        rec["pearson_vs_official"] = float(np.corrcoef(xs, ys)[0, 1])
    # 逐帧落盘（供中断规则分析）
    np.savez_compressed(a.out.replace(".json", "_frames.npz"),
                        EP=EP, T=T, score=sc, p=p)
    json.dump(rec, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "episodes"}, ensure_ascii=False, indent=1))
    print(f"{'ep':>4} {'n':>4} {'score_mean':>10} {'p_med':>6} {'flag%':>6} {'official':>8}")
    for e in rec["episodes"]:
        print(f"{e['ep']:4d} {e['n']:4d} {e['score_mean']:10.4f} {e['p_median']:6.3f} "
              f"{e['flag_rate']*100:5.1f}% {e.get('official_score', float('nan')):8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "train", "score"])
    ap.add_argument("--source", default="demo", choices=["demo", "eval"])
    ap.add_argument("--episodes", default="")
    ap.add_argument("--demo-clips", default="/home/syt/demo_clips")
    ap.add_argument("--eval-dir", default="")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ckpt", default="/home/syt/tmp_ws/floraU_eval/30000")
    ap.add_argument("--train-config", default="pi05_base_aloha_full_sim_arx-x5_seed_0")
    ap.add_argument("--repo-id", default="arx_x5_sim")
    ap.add_argument("--out", default="/tmp/e2/out")
    ap.add_argument("--demo", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--emb", default="")
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--official", default="")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    {"extract": stage_extract, "train": stage_train, "score": stage_score}[a.stage](a)


if __name__ == "__main__":
    main()
