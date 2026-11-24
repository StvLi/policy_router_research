#!/usr/bin/env python3
"""路线 B：只按"原子技能"划分（跨任务通用），不按 task × subtask 划分

与路线 A 的区别（路线 A 冻结在 archive/route-A-vae3tile-state-k0/）：
  A：每个任务内部聚成 5 类 → 每类的 VAE 只见过该任务
  B：**全部 35 个任务的句子放在一起聚成 N=32 类** → 每一类是跨任务的"原子技能"
     （抓取/放置/递交/挂…，且区分左右手），每类的 VAE 见过所有任务

评估的关键是**跨任务**：VAE 只在 28 个任务上训练，然后在 7 个**没见过**的任务上测。

四段执行（两个 Python 环境互补：labeler 有 pyarrow/av/torch，vllm 有 transformers）：
  --stage prep     labeler：枚举 episode、抽段标注、拿任务名 → spec json
  --stage embed    vllm   ：12k 条句子 → 句向量 npz
  --stage cluster  labeler：句向量 → k=32 球面 k-means → skill index（先看分布再决定是否取帧）
  --stage run      labeler：取帧 → 每类训一个 VAE → 评估（同任务 / 跨任务）

用法：
  L=/hdfs/users/peize/vlm_vla/env/labeler/bin/python
  V=/hdfs/users/peize/vlm_vla/env/vllm/bin/python
  $L routeB_atomic_skills.py --stage prep
  CUDA_VISIBLE_DEVICES=4 $V routeB_atomic_skills.py --stage embed
  $L routeB_atomic_skills.py --stage cluster --k 32
  CUDA_VISIBLE_DEVICES=4 $L routeB_atomic_skills.py --stage run --k 32
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

DEFAULT_DATASET = "/hdfs/users/peize/vlm_vla/datasets/RoboDojo35FloraUnified-v1"
DEFAULT_LABELS = "/hdfs/users/peize/vlm_vla/flora/labels/merged_v1"
DEFAULT_CLIPS = "/hdfs/users/peize/vlm_vla/flora/clips"
DEFAULT_ENCODER = "/hdfs/models/huggingface/BAAI/bge-small-en-v1.5"
FPS = 25
VIEWS = 3
IMG_H, IMG_W = 48, 64
REC_H, REC_W = 24, 32
NS = 14
SH = 3
HIST_OFFSETS = [50, 12, 0]        # t-2.0s / t-0.48s / t
POOL_H, POOL_W = 3, 4


# ================================================================= 数据定位
def all_task_episodes(dataset):
    """episode_index -> 任务名（取 tasks[0] 的 'Task: ...' 前缀）"""
    import pyarrow.parquet as pq
    out = {}
    for p in sorted(glob.glob(os.path.join(dataset, "meta", "episodes", "chunk-*", "file-*.parquet"))):
        for r in pq.read_table(p, columns=["episode_index", "tasks", "length"]).to_pylist():
            ts = r["tasks"]
            if not ts:
                continue
            head = ts[0].split("Current subtask:")[0].strip()
            if head.lower().startswith("task:"):
                head = head[5:].strip()
            out[int(r["episode_index"])] = (head, int(r["length"]))
    return out


def load_segments(labels_dir, ep):
    p = os.path.join(labels_dir, f"ep{ep:06d}.json")
    if not os.path.exists(p):
        return []
    d = json.load(open(p))
    return [dict(start_s=float(s["start_s"]), end_s=float(s["end_s"]),
                 caption=s["caption"].strip().lower())
            for s in d.get("segments", []) if s.get("caption", "").strip()]


def frame_labels(segs, n_frames):
    lab = np.full(n_frames, -1, np.int32)
    cover = np.zeros(n_frames, np.int16)
    for i, s in enumerate(segs):
        a = max(0, min(int(round(s["start_s"] * FPS)), n_frames))
        b = max(0, min(int(round(s["end_s"] * FPS)), n_frames))
        if b > a:
            lab[a:b] = i
            cover[a:b] += 1
    kn = np.where(lab >= 0)[0]
    if len(kn) == 0:
        return lab, np.ones(n_frames, bool)
    unk = np.where(lab < 0)[0]
    if len(unk):
        idx = np.clip(np.searchsorted(kn, unk), 0, len(kn) - 1)
        left = kn[np.clip(idx - 1, 0, len(kn) - 1)]
        right = kn[idx]
        pick = np.where(np.abs(unk - left) <= np.abs(right - unk), left, right)
        lab[unk] = lab[pick]
    return lab, cover > 1


# ================================================================= 句向量与聚类
def embed(captions, model_dir, device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    mdl = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(captions), 128):
            b = captions[i:i + 128]
            enc = tok(b, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    X = np.concatenate(out).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X


def spherical_kmeans(X, k, iters=200, seed=0, restarts=3):
    """多次重启取簇内余弦之和最大的一次（N=32 时更容易陷局部解）。"""
    best = None
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        C = X[rng.choice(len(X), k, replace=False)].copy()
        lab = np.zeros(len(X), np.int64)
        for _ in range(iters):
            new = (X @ C.T).argmax(1)
            if np.array_equal(new, lab):
                break
            lab = new
            for c in range(k):
                m = lab == c
                if m.any():
                    v = X[m].mean(0)
                    C[c] = v / (np.linalg.norm(v) + 1e-12)
        score = float((X * C[lab]).sum())
        if best is None or score > best[0]:
            best = (score, lab.copy(), C.copy())
    return best[1], best[2], best[0]


# ================================================================= 取帧
def decode_all_views(clips_dir, ep):
    import av
    import cv2
    frames = []
    for v in ("main", "left_wrist", "right_wrist"):
        p = os.path.join(clips_dir, f"ep{ep:06d}", f"{v}.mp4")
        if not os.path.exists(p):
            return None
        out = []
        try:
            with av.open(p) as c:
                for f in c.decode(video=0):
                    out.append(cv2.resize(f.to_ndarray(format="rgb24"), (IMG_W, IMG_H),
                                          interpolation=cv2.INTER_AREA))
        except Exception as e:
            print(f"  [warn] {type(e).__name__} {p}")
            return None
        frames.append(np.stack(out) if out else None)
    if any(f is None for f in frames):
        return None
    n = min(len(f) for f in frames)
    return np.stack([f[:n] for f in frames], axis=1)      # (T, 3视角, H, W, 3)


def load_all_states(dataset, eps):
    import pyarrow.parquet as pq
    want = set(int(e) for e in eps)
    out = {}
    for p in sorted(glob.glob(os.path.join(dataset, "data", "chunk-*", "*.parquet"))):
        tb = pq.read_table(p, columns=["episode_index", "frame_index", "observation.state"])
        epi = np.asarray(tb["episode_index"].to_pylist())
        if not want.intersection(epi.tolist()):
            continue
        m = np.isin(epi, list(want))
        fi = np.asarray(tb["frame_index"].to_pylist())[m]
        epsel = epi[m]
        rows = [x for x, keep in zip(tb["observation.state"].to_pylist(), m) if keep]
        st = np.stack([np.asarray(x, np.float32) for x in rows])
        for e in np.unique(epsel):
            sel = epsel == e
            o = np.argsort(fi[sel])
            out[int(e)] = st[sel][o]
    return out


def state_windows(sm, EP, FIDX):
    Xs = np.zeros((len(EP), SH, NS), np.float32)
    for i, (e, f) in enumerate(zip(EP, FIDX)):
        st = sm.get(int(e))
        if st is None:
            continue
        for j, off in enumerate(HIST_OFFSETS):
            Xs[i, j] = st[min(max(0, int(f) - off), len(st) - 1)]
    return Xs


# ================================================================= 模型
def build_vae(channels, latent, device):
    import torch
    import torch.nn as nn
    c1, c2 = channels
    TILES = VIEWS

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, c1, 3, 2, 1)                  # 48x64 -> 24x32
            self.c2 = nn.Conv2d(c1, c2, 3, 2, 1)                 # -> 12x16
            self.pool = nn.AdaptiveAvgPool2d((POOL_H, POOL_W))
            self.vproj = nn.Linear(c2 * POOL_H * POOL_W, 24)
            self.sproj = nn.Linear(NS * SH, 32)
            self.head = nn.Linear(24 * TILES + 32, 2 * latent)
            self.v = nn.Linear(latent, TILES * c2 * POOL_H * POOL_W)
            self.up = nn.Sequential(
                nn.ConvTranspose2d(c2, c1, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(c1, 8, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(8, 3, 4, 2, 1))
            self.sdec = nn.Linear(latent, NS * SH)

        def forward(self, img, st, B):
            x = torch.relu(self.c1(img))
            x = torch.relu(self.c2(x))
            x = self.pool(x).reshape(B * TILES, -1)
            x = self.vproj(x).reshape(B, TILES * 24)
            s = torch.relu(self.sproj(st.reshape(B, -1)))
            mu, lv = self.head(torch.cat([x, s], 1)).chunk(2, 1)
            lv = lv.clamp(-6, 4)
            z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
            v = self.v(z).reshape(B * TILES, c2, POOL_H, POOL_W)
            return mu, lv, z, torch.sigmoid(self.up(v)), self.sdec(z)

    return VAE().to(device)


def prep_img(x, device):
    import torch
    t = torch.from_numpy(np.ascontiguousarray(x)).float().to(device) / 255.0
    B = t.shape[0]
    return t.permute(0, 1, 4, 2, 3).reshape(B * VIEWS, 3, IMG_H, IMG_W)


def check_prep(device):
    probe = np.zeros((1, 3, IMG_H, IMG_W, 3), np.uint8)
    probe[0, 0, 0, 0] = [10, 200, 30]
    probe[0, 2, 0, 0] = [7, 8, 9]
    o = (prep_img(probe, device) * 255).cpu().numpy()
    assert abs(o[0, 0, 0, 0] - 10) < 1 and abs(o[0, 1, 0, 0] - 200) < 1 \
        and abs(o[2, 0, 0, 0] - 7) < 1, "prep_img 通道顺序错误"
    print("prep_img 自检通过")


def train_one(Xi, Xs, channels, latent, iters, beta, device, seed, state_w=4.0):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    vae = build_vae(channels, latent, device)
    opt = torch.optim.Adam(vae.parameters(), lr=2e-3)
    n = len(Xi)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    for it in range(1, iters + 1):
        rng.shuffle(idx)
        b = idx[:64]
        img = prep_img(Xi[b], device)
        st = torch.from_numpy(Xs[b]).float().to(device)
        B = len(b)
        mu, lv, z, ri, rs = vae(img, st, B)
        tgt = F.adaptive_avg_pool2d(img, (REC_H, REC_W))
        rec_i = F.binary_cross_entropy(ri, tgt)
        rec_s = F.mse_loss(rs, st.reshape(B, -1))
        kl = (-0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), 1)).mean() / (NS * SH)
        loss = rec_i + state_w * rec_s + beta * kl
        opt.zero_grad(); loss.backward(); opt.step()
        if it == iters:
            print(f"    it {it} img {rec_i.item():.4f} st {rec_s.item():.4f} kl {kl.item():.4f}")
    return vae, sum(p.numel() for p in vae.parameters())


def nll(vae, Xi, Xs, device, beta=1.0, state_w=4.0, bs=512):
    """逐样本非一致性分数：重建项 + KL 项（越大越不像）。"""
    import torch
    import torch.nn.functional as F
    out = []
    with torch.no_grad():
        for i in range(0, len(Xi), bs):
            sl = slice(i, i + bs)
            img = prep_img(Xi[sl], device)
            st = torch.from_numpy(Xs[sl]).float().to(device)
            B = len(Xi[sl])
            mu, lv, z, ri, rs = vae(img, st, B)
            tgt = F.adaptive_avg_pool2d(img, (REC_H, REC_W))
            pi = F.binary_cross_entropy(ri, tgt, reduction="none").flatten(1).mean(1).reshape(B, VIEWS).mean(1)
            ps = F.mse_loss(rs, st.reshape(B, -1), reduction="none").mean(1)
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), 1) / (NS * SH)
            out.append((pi + state_w * ps + beta * kl).cpu().numpy())
    return np.concatenate(out)


def conformal_p(score, cal):
    return (1.0 + (cal[None, :] >= score[:, None]).sum(1)) / (len(cal) + 1.0)


# ================================================================= 各阶段
def stage_prep(a):
    ep2task = all_task_episodes(a.dataset)
    by_task = collections.defaultdict(list)
    for ep, (task, _) in sorted(ep2task.items()):
        by_task[task].append(ep)
    print(f"任务数 {len(by_task)}；episode 总数 {len(ep2task)}")
    picked = {}
    for task, eps in sorted(by_task.items()):
        eps = sorted(eps)
        picked[task] = eps[:a.episodes_per_task]
    total = sum(len(v) for v in picked.values())
    print(f"每任务取前 {a.episodes_per_task} 集 → 共 {total} 集")

    spec = {}
    caps = set()
    for task, eps in picked.items():
        for ep in eps:
            segs = load_segments(a.labels, ep)
            if segs:
                spec[str(ep)] = dict(task=task, segments=segs)
                caps.update(s["caption"] for s in segs)
    caps = sorted(caps)
    print(f"有标注 episode {len(spec)}；不同句子 {len(caps)}")
    json.dump(dict(tasks=sorted(picked), segs=spec, caps=caps),
              open(a.spec, "w"), ensure_ascii=False)
    print(f"写出 {a.spec}")


def stage_embed(a):
    import torch
    spec = json.load(open(a.spec))
    caps = spec["caps"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E = embed(caps, a.encoder, dev)
    np.savez(a.emb, caps=np.array(caps, dtype=object), emb=E)
    print(f"句向量 {E.shape} → {a.emb}")


def stage_cluster(a):
    spec = json.load(open(a.spec))
    z = np.load(a.emb, allow_pickle=True)
    caps = list(z["caps"]); E = z["emb"]
    lab, C, score = spherical_kmeans(E, a.k, seed=0, restarts=a.restarts)
    sizes = collections.Counter(lab.tolist())
    # 按句子在数据中的出现次数加权（同一句话可能覆盖很多帧）
    cap_count = collections.Counter()
    for ep, d in spec["segs"].items():
        for s in d["segments"]:
            cap_count[s["caption"]] += 1
    w = collections.Counter()
    for c, l in zip(caps, lab):
        w[int(l)] += cap_count[c]
    print(f"k={a.k}  簇内余弦之和 {score:.1f}")
    print(f"{'簇':>3} {'句子数':>6} {'段数':>7}  代表句")
    for c in sorted(sizes, key=lambda x: -w[x]):
        ex = [caps[i] for i in range(len(caps)) if lab[i] == c][:2]
        print(f"{c:>3} {sizes[c]:>6} {w[c]:>7}  {ex[0][:66]}")
    json.dump(dict(k=a.k, caps=caps, labels=lab.tolist(),
                   sizes={int(k): int(v) for k, v in sizes.items()},
                   seg_counts={int(k): int(v) for k, v in w.items()}),
              open(a.skill, "w"), ensure_ascii=False)
    print(f"写出 {a.skill}")


def stage_run(a):
    import random as _random
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    check_prep(dev)
    spec = json.load(open(a.spec))
    sk = json.load(open(a.skill))
    caps = sk["caps"]; labels = sk["labels"]
    cap2skill = {c: int(l) for c, l in zip(caps, labels)}
    print(f"原子技能数 k={sk['k']}；任务数 {len(spec['tasks'])}")

    # ---- 任务切分：留出 a.holdout_tasks 个任务完全不参与训练 ----
    tasks = sorted(spec["tasks"])
    rt = _random.Random(0); rt.shuffle(tasks)
    holdout = set(tasks[:a.holdout_tasks])
    print(f"训练任务 {len(tasks)-len(holdout)} 个；留出任务 {len(holdout)} 个："
          f"{[t[:40] for t in sorted(holdout)]}")

    eps_train, eps_test = [], []
    for ep_s, d in spec["segs"].items():
        (eps_test if d["task"] in holdout else eps_train).append(int(ep_s))
    print(f"episode：训练任务 {len(eps_train)} 集 / 留出任务 {len(eps_test)} 集")

    cap_index = {c: i for i, c in enumerate(spec["caps"])}
    cache = a.cache
    if os.path.exists(cache) and "CI" in np.load(cache, allow_pickle=True):
        z = np.load(cache, allow_pickle=True)
        Xi, Xs, CI, EP = z["Xi"], z["Xs"], z["CI"], z["EP"]
        print(f"复用取帧缓存 {cache}：{len(CI)} 帧")
    else:
        Xi, Xs, CI, EP = _collect_frames(a, spec, cap_index, sorted(eps_train + eps_test))
        np.savez(cache, Xi=Xi, Xs=Xs, CI=CI, EP=EP)
        print(f"取帧缓存写出 {cache}")
    CI = np.asarray(CI); EP = np.asarray(EP)
    # 技能标签按**当前**聚类重算（缓存与 k 解耦，这是关键）
    SK = np.array([cap2skill[spec["caps"][i]] for i in CI], np.int64)
    print(f"按当前 k={a.k} 重算技能标签：{np.unique(SK).size} 类")
    mu = Xs.reshape(-1, NS).mean(0); sd = Xs.reshape(-1, NS).std(0) + 1e-6
    Xs = ((Xs - mu) / sd).astype(np.float32)

    tr_ep = np.isin(EP, eps_train)
    print(f"帧数：训练任务 {tr_ep.sum()} / 留出任务 {(~tr_ep).sum()}")
    print(f"技能分布（训练任务内）: {dict(sorted(collections.Counter(SK[tr_ep].tolist()).items()))}")

    # ---- 每类一个 VAE（只用训练任务的帧）----
    channels = tuple(int(x) for x in a.channels.split(","))
    models, params, skipped = {}, {}, []
    for c in sorted(set(SK[tr_ep].tolist())):
        m = tr_ep & (SK == c)
        if m.sum() < a.min_frames:
            skipped.append((c, int(m.sum()))); continue
        print(f"\n[技能 {c}] {m.sum()} 帧，通道 {channels}，隐维 {a.latent} ...")
        vae, np_ = train_one(Xi[m], Xs[m], channels, a.latent, a.iters, a.beta, dev, seed=c)
        models[c] = vae; params[c] = np_
    print(f"\n训练完成 {len(models)} 个 VAE；参数 {list(params.values())[:1]}；跳过 {skipped}")

    # ---- 校准（训练任务内、按集再分 80/20）----
    eps_tr_sorted = sorted(set(EP[tr_ep].tolist())); _random.Random(1).shuffle(eps_tr_sorted)
    n_ho = max(1, int(round(len(eps_tr_sorted) * 0.2)))
    ho_eps = set(eps_tr_sorted[:n_ho])
    cal_mask = tr_ep & np.isin(EP, eps_tr_sorted[n_ho:])
    eval_ind = tr_ep & np.isin(EP, list(ho_eps))
    eval_cross = ~tr_ep
    print(f"校准帧 {cal_mask.sum()}；同任务评估帧 {eval_ind.sum()}；跨任务评估帧 {eval_cross.sum()}")

    cal = {}
    for c, vae in models.items():
        mc = cal_mask & (SK == c)
        if mc.sum() >= 20:
            cal[c] = nll(vae, Xi[mc], Xs[mc], dev, a.beta)
    print(f"可用校准集 {len(cal)} 个技能")

    def evaluate(mask, name):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        y = SK[idx]
        have = np.isin(y, list(models))          # 该技能的模型是否存在
        P = np.zeros((len(idx), a.k))
        for c, vae in models.items():
            P[:, c] = conformal_p(nll(vae, Xi[idx], Xs[idx], dev, a.beta), cal[c])
        pred = P.argmax(1)
        acc = float((pred == y).mean())
        acc_av = float((pred[have] == y[have]).mean()) if have.any() else float("nan")
        rec = [float((pred[y == c] == c).mean()) for c in sorted(models) if (y == c).any()]
        print(f"\n--- {name}（{len(idx)} 帧，随机 {100/a.k:.1f}%）---")
        print(f"  技能可用性：{have.sum()}/{len(idx)} = {have.mean()*100:.1f}%（其余技能的模型在训练侧不存在）")
        print(f"  准确率（全部帧，含不可答） {acc*100:.1f}%")
        print(f"  准确率（仅可答帧）        {acc_av*100:.1f}%")
        print(f"  有模型类的平衡准确率 {np.mean(rec)*100:.1f}%")
        print(f"  各类召回 {[f'{r*100:.0f}%' for r in rec]}")
        return dict(n=int(len(idx)), n_avail=int(have.sum()), acc=acc, acc_available=acc_av,
                    balanced=float(np.mean(rec)), recall=rec)

    res_same = evaluate(eval_ind, "同任务（同任务的留出 episode）")
    res_cross = evaluate(eval_cross, "跨任务（完全没见过的任务）")
    json.dump(dict(k=a.k, channels=list(channels), latent=a.latent, iters=a.iters,
                   n_models=len(models), params=list(params.values())[:1], skipped=skipped,
                   holdout_tasks=sorted(holdout), same_task=res_same, cross_task=res_cross),
              open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n写出 {a.out}")


def _collect_frames(a, spec, cap_index, eps):
    """存句子索引 CI（而不是技能 id），这样同一份取帧缓存可被任意 k 复用。"""
    Xi, Xs, CI, EP = [], [], [], []
    sm = load_all_states(a.dataset, eps)
    for n_done, ep in enumerate(sorted(eps), 1):
        d = spec["segs"].get(str(ep))
        if d is None or ep not in sm:
            continue
        segs = d["segments"]
        views = decode_all_views(a.clips, ep)
        if views is None:
            continue
        st = sm[ep]
        T = min((len(views) - 1) // a.stride + 1, (len(st) - 1) // a.stride + 1)
        lab, amb = frame_labels(segs, len(st))
        for i in range(T):
            f = i * a.stride
            if f >= len(views) or f >= len(st) or amb[f]:
                continue
            j = lab[f]
            if j < 0:
                continue
            tiles = views[f]                       # 当前帧的 3 视角（与路线 A 的图像输入同构）
            win = np.stack([st[min(max(0, f - o), len(st) - 1)] for o in HIST_OFFSETS])
            Xi.append(tiles); Xs.append(win)
            CI.append(cap_index[segs[j]["caption"]]); EP.append(ep)
        if n_done % 20 == 0 or n_done == len(eps):
            print(f"  取帧 {n_done}/{len(eps)} 集，累计 {len(Xi)} 帧")
    if not Xi:
        sys.exit("没有取到任何帧")
    return (np.stack(Xi), np.stack(Xs).astype(np.float32),
            np.asarray(CI, np.int64), np.asarray(EP, np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--clips", default=DEFAULT_CLIPS)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--stage", default="run", choices=["prep", "embed", "cluster", "run"])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--episodes-per-task", type=int, default=6)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--holdout-tasks", type=int, default=7)
    ap.add_argument("--channels", default="16,32", help="两个卷积层的通道数；更大模型用 32,64")
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--min-frames", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--spec", default="/tmp/routeB_spec.json")
    ap.add_argument("--emb", default="/tmp/routeB_emb.npz")
    ap.add_argument("--skill", default="/tmp/routeB_skills.json")
    ap.add_argument("--cache", default="/tmp/routeB_frames.npz")
    ap.add_argument("--out", default="/tmp/routeB_result.json")
    a = ap.parse_args()
    # 缓存名带上取帧参数，避免改了集数/步长却复用旧缓存
    a.spec = a.spec.replace(".json", f"_n{a.episodes_per_task}.json")
    a.emb = a.emb.replace(".npz", f"_n{a.episodes_per_task}.npz")
    a.skill = a.skill.replace(".json", f"_n{a.episodes_per_task}_k{a.k}.json")
    a.cache = a.cache.replace(".npz", f"_n{a.episodes_per_task}_s{a.stride}.npz")
    {"prep": stage_prep, "embed": stage_embed,
     "cluster": stage_cluster, "run": stage_run}[a.stage](a)


if __name__ == "__main__":
    main()
