#!/usr/bin/env python3
"""单任务试点：第 1/2/3 步技术路线可用性验证

⚠️ 本文件是**会继续演进的线上版本**。
   若要看"路线 A（3 格图像 + 状态 vlm3 + 自重构 VAE）冻结时到底是什么"，
   以 archive/route-A-vae3tile-state-k0/code/routeA_pilot_store_laptop.py 为准。
   默认参数已固定为路线 A 的配置（不带参数运行 = 复现路线 A）。


目标任务：Hang the headphones on the headphone stand, close the laptop,
          then place it into the vertical laptop stand.

三步：
  1. 子任务句子 -> 句向量（bge-small-en-v1.5）-> k-means 得 N 类（= docs 的"子任务"）
  2. 每一类训练一个极小的变分自编码器（VAE），输入 = 三视角图像 + 关节状态历史；
     **编码器做激进池化**（640x480 -> 48x64 -> 16x24 -> 12x16 -> 自适应均值池化 3x4），
     参数量约 2 万，用来避免 5.9k 帧/类 的过拟合
  3. 用留出集算"共形 p 值"，检验能否据此判断"当前属于哪一类"，并给出风险-覆盖率

用法：
  P=/hdfs/users/peize/vlm_vla/env/vllm/bin/python
  $P pilot_store_laptop.py --smoke            # 每集 20 帧、5 集，验证管道
  $P pilot_store_laptop.py                    # 全量该任务
  $P pilot_store_laptop.py --no-image         # 只用电机的消融
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np

DEFAULT_DATASET = "/hdfs/users/peize/vlm_vla/datasets/RoboDojo35FloraUnified-v1"
DEFAULT_LABELS = "/hdfs/users/peize/vlm_vla/flora/labels/merged_v1"
DEFAULT_CLIPS = "/hdfs/users/peize/vlm_vla/flora/clips"
DEFAULT_ENCODER = "/hdfs/models/huggingface/BAAI/bge-small-en-v1.5"
FPS = 25
VIEWS = ("main", "left_wrist", "right_wrist")
TASK_KEY = "headphone"          # 用子串定位任务（大小写不敏感）
IMG_H, IMG_W = 48, 64           # 编码器输入
REC_H, REC_W = 24, 32           # 重建目标（相对原图 20 倍下采样）
POOL_H, POOL_W = 3, 4           # 激进池化后的空间尺寸
NS = 14                         # 关节状态维度
# 历史窗口：显式的"往回多少帧"，与采样间隔无关（这样不同 stride 的样本仍然可比）
#   vlm3    = [50, 12, 0]  与 SUBTASK_VLM_SPEC 冻结的 t-50 / t-12 / t 对齐
#   window4 = [15, 10, 5, 0]  连续 0.6 s 四帧（本试点原先的做法）
#   single  = [0]           只看当前帧
HIST_OFFSETS = [15, 10, 5, 0]
SH = len(HIST_OFFSETS)

# 图像的时间点。single = 只看当前帧（3 格）；vlm3 = t / t-12 / t-50（3 相机 × 3 时刻 = 9 格，
# 与 SUBTASK_VLM_SPEC 冻结的九宫格同构）。行=时刻（旧→新），列=相机。
IMG_OFFSETS = [50, 12, 0]
NT = len(IMG_OFFSETS)
TILES = NT * 3

# 预测步长：0 = 重构输入本身（自重构）；>0 = 预测 t+k 的观测（越界取末帧）
PRED_K = 0
ST_OUT = NS * len(HIST_OFFSETS)          # 解码器的状态输出维度；PRED_K==0 时改为 NS*SH
CLAMPED = [0, 0]     # [被截到末帧的样本数, 总样本数]


# ----------------------------------------------------------------- 数据定位
def resolve_task(dataset, key):
    import pyarrow.parquet as pq
    rows = pq.read_table(os.path.join(dataset, "meta", "tasks.parquet")).to_pylist()
    total = {}
    for r in rows:
        s = list(r.values())[-1]
        if not isinstance(s, str) or "Current subtask:" not in s:
            continue
        head = s.split("Current subtask:")[0].strip()
        total[head] = total.get(head, 0) + 1
    hits = [k for k in total if key.lower() in k.lower()]
    if len(hits) != 1:
        sys.exit(f"任务定位失败，命中 {len(hits)} 个: {hits}")
    name = hits[0]
    eps_path = sorted(glob.glob(os.path.join(dataset, "meta", "episodes", "chunk-*", "file-*.parquet")))
    episodes = []
    for p in eps_path:
        for r in pq.read_table(p, columns=["episode_index", "tasks", "length"]).to_pylist():
            if any(name in t for t in r["tasks"]):
                episodes.append((int(r["episode_index"]), int(r["length"])))
    episodes.sort()
    return name, episodes


def load_segments(labels_dir, ep):
    p = os.path.join(labels_dir, f"ep{ep:06d}.json")
    if not os.path.exists(p):
        return []
    d = json.load(open(p))
    return [s for s in d.get("segments", []) if s.get("caption", "").strip()]


def frame_labels(segs, n_frames):
    """帧 -> 段序号；重叠区记为 -1（歧义，训练时剔除），空隙归最近段。"""
    lab = np.full(n_frames, -1, dtype=np.int32)
    cover = np.zeros(n_frames, dtype=np.int16)
    for i, s in enumerate(segs):
        a = max(0, min(int(round(s["start_s"] * FPS)), n_frames))
        b = max(0, min(int(round(s["end_s"] * FPS)), n_frames))
        if b > a:
            lab[a:b] = i
            cover[a:b] += 1
    known = np.where(lab >= 0)[0]
    if len(known) == 0:
        return lab, np.ones(n_frames, bool)
    unk = np.where(lab < 0)[0]
    if len(unk):
        idx = np.clip(np.searchsorted(known, unk), 0, len(known) - 1)
        left = known[np.clip(idx - 1, 0, len(known) - 1)]
        right = known[idx]
        pick = np.where(np.abs(unk - left) <= np.abs(right - unk), left, right)
        lab[unk] = lab[pick]
    return lab, cover > 1


# ----------------------------------------------------------------- 第 1 步
def embed_captions(captions, model_dir, device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    mdl = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(captions), 64):
            batch = captions[i:i + 64]
            enc = tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    X = np.concatenate(out).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X


def spherical_kmeans(X, k, iters=150, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.full(len(X), -1)
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
    return lab, C


# ----------------------------------------------------------------- 视频解码
def decode_views(clips_dir, ep, stride, keep_all=False):
    import av
    import cv2
    frames = {}
    for v in VIEWS:
        p = os.path.join(clips_dir, f"ep{ep:06d}", f"{v}.mp4")
        if not os.path.exists(p):
            return None
        out = []
        try:
            with av.open(p) as c:
                for i, f in enumerate(c.decode(video=0)):
                    if not keep_all and (i % stride):
                        continue
                    img = f.to_ndarray(format="rgb24")
                    out.append(cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA))
        except Exception as e:
            print(f"  [warn] decode {p}: {type(e).__name__} {e}")
            return None
        frames[v] = np.stack(out) if out else None
    if any(frames[v] is None for v in VIEWS):
        return None
    n = min(len(frames[v]) for v in VIEWS)
    return np.stack([frames[v][:n] for v in VIEWS], axis=1)   # (T, 3views, H, W, 3) uint8


def set_pred_k(k):
    global PRED_K, ST_OUT
    PRED_K = int(k)
    ST_OUT = NS if PRED_K > 0 else NS * len(HIST_OFFSETS)
    print(f"预测步长 k={PRED_K}（{'预测 t+k 的观测' if PRED_K else '重构输入本身'}），"
          f"状态输出维度 {ST_OUT}")


def set_img_hist(mode):
    global IMG_OFFSETS, NT, TILES
    if mode == "vlm3":
        IMG_OFFSETS = [50, 12, 0]
    elif mode == "single":
        IMG_OFFSETS = [0]
    else:
        sys.exit(f"未知 --img-hist {mode}")
    NT = len(IMG_OFFSETS)
    TILES = NT * 3
    print(f"图像时间点 {mode}：偏移 {IMG_OFFSETS}，共 {TILES} 格（{NT} 时刻 × 3 相机）")


def set_hist(mode):
    global HIST_OFFSETS, SH
    if mode == "vlm3":
        HIST_OFFSETS = [50, 12, 0]
    elif mode == "window4":
        HIST_OFFSETS = [15, 10, 5, 0]
    elif mode == "single":
        HIST_OFFSETS = [0]
    else:
        sys.exit(f"未知 --hist {mode}")
    SH = len(HIST_OFFSETS)
    print(f"历史方案 {mode}：帧偏移 {HIST_OFFSETS}（"
          f"约 {[round(o/25,2) for o in HIST_OFFSETS]} 秒）")


def load_all_states(dataset, eps):
    """一次扫过所有 parquet，收集指定 episode 的完整状态序列。"""
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


def state_windows(states_map, EP, FIDX):
    """按 HIST_OFFSETS 从完整状态序列中取历史；早于起点则重复首帧（与 VLM 规格一致）。"""
    Xs = np.zeros((len(EP), SH, NS), np.float32)
    for i, (e, f) in enumerate(zip(EP, FIDX)):
        st = states_map.get(int(e))
        if st is None:
            continue
        for j, off in enumerate(HIST_OFFSETS):
            idx = min(max(0, int(f) - off), len(st) - 1)
            Xs[i, j] = st[idx]
    return Xs


def state_target(states_map, EP, FIDX):
    """目标状态：t+k 那一刻的 14 维（越界取末帧）。PRED_K==0 时返回整个输入窗口。"""
    if PRED_K <= 0:
        return None
    out = np.zeros((len(EP), NS), np.float32)
    for i, (e, f) in enumerate(zip(EP, FIDX)):
        st = states_map.get(int(e))
        if st is None:
            continue
        out[i] = st[min(int(f) + PRED_K, len(st) - 1)]
    return out


def load_states(dataset, ep):
    import pyarrow.parquet as pq
    for p in sorted(glob.glob(os.path.join(dataset, "data", "chunk-*", "*.parquet"))):
        t = pq.read_table(p, columns=["episode_index", "frame_index", "observation.state"])
        epi = np.asarray(t["episode_index"].to_pylist())
        m = epi == ep
        if not m.any():
            continue
        fi = np.asarray(t["frame_index"].to_pylist())[m]
        rows = t["observation.state"].to_pylist()
        st = np.stack([np.asarray(x, np.float32) for x, keep in zip(rows, m) if keep])
        o = np.argsort(fi)
        return st[o]
    return None


# ----------------------------------------------------------------- 第 2 步 VAE
def check_prep_img():
    """回归自检：通道前置是否做对（左上角像素应等于原图左上角）。"""
    probe = np.zeros((1, NT, 3, IMG_H, IMG_W, 3), np.uint8)  # (B, NT, 3视角, H, W, C)
    probe[0, 0, 0, 0, 0] = [10, 200, 30]          # 视角 0 左上角像素
    probe[0, 0, 0, IMG_H - 1, IMG_W - 1] = [1, 2, 250]   # 视角 0 右下角像素
    probe[0, 0, 2, 0, 0] = [7, 8, 9]              # 视角 2 左上角像素
    out = (prep_img(probe, "cpu") * 255.0).numpy()   # prep_img 返回 0–1 浮点
    def near(actual, want):
        return abs(float(actual) - want) < 1.0
    ok = (near(out[0, 0, 0, 0], 10) and near(out[0, 1, 0, 0], 200)
          and near(out[0, 2, 0, 0], 30)
          and near(out[0, 2, IMG_H - 1, IMG_W - 1], 250)
          and near(out[2, 0, 0, 0], 7) and near(out[2, 1, 0, 0], 8)
          and out.shape[0] == TILES)
    if not ok:
        raise RuntimeError("prep_img 通道顺序错误")
    print("prep_img 自检通过（通道前置正确）")


def prep_img(x_np, device):
    """(B, 3视角, H, W, C) uint8 numpy -> (B*3, C, H, W) float。
    注意：必须做通道前置（permute），否则 view() 会把像素重排成垃圾。"""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(x_np)).float().to(device) / 255.0
    B = t.shape[0]                       # (B, NT, 3视角, H, W, C)
    # 必须先把通道维移到前面，再摊平成 (B*TILES, C, H, W)；直接 reshape 会重排像素
    return t.permute(0, 1, 2, 5, 3, 4).reshape(B * TILES, 3, IMG_H, IMG_W)


USE_IMAGE = True
USE_STATE = True
STATE_W = 2.5


def build_model(d=16):
    import torch
    import torch.nn as nn

    class VAE(nn.Module):
        """编码器做激进池化：640x480 -> 48x64 -> 24x32 -> 12x16 -> 自适应均值池化 3x4。
        参数量约 5 万，用来在每类 ~1-3 千帧的预算下避免过拟合。"""

        def __init__(self):
            super().__init__()
            self.use_image = USE_IMAGE
            self.c1 = nn.Conv2d(3, 16, 3, 2, 1)                 # -> 16x24x32
            self.c2 = nn.Conv2d(16, 32, 3, 2, 1)                # -> 32x12x16
            self.pool = nn.AdaptiveAvgPool2d((POOL_H, POOL_W))  # 激进池化 -> 32x3x4
            self.vproj = nn.Linear(32 * POOL_H * POOL_W, 24)
            self.sproj = nn.Linear(NS * SH, 24)
            din = (24 * TILES if USE_IMAGE else 0) + (24 if USE_STATE else 0)
            self.head = nn.Linear(din, 2 * d)
            self.v = nn.Linear(d, TILES * 32 * POOL_H * POOL_W)
            # 从 (POOL_H, POOL_W) 逐级上采样到 (REC_H, REC_W)，层数随池化强度自动决定
            # 逐级 2 倍上采样，直到两个方向都不小于目标，再插值到精确尺寸
            # （池化到 1x1 时 2 的幂次永远凑不出 24x32，所以必须用插值兜底）
            stages, ch, hh, ww = [], 32, POOL_H, POOL_W
            while hh < REC_H or ww < REC_W:
                nxt = 16 if ch == 32 else (8 if ch == 16 else 3)
                stages += [nn.ConvTranspose2d(ch, nxt, 4, 2, 1), nn.ReLU()]
                ch, hh, ww = nxt, hh * 2, ww * 2
            stages = stages[:-1]
            self.up = nn.Sequential(*stages)
            self.to3 = nn.Conv2d(ch, 3, 1) if ch != 3 else nn.Identity()
            self.sdec = nn.Linear(d, ST_OUT)

        def forward(self, img_flat, st, B):
            if self.use_image:
                x = torch.relu(self.c1(img_flat))
                x = torch.relu(self.c2(x))
                x = self.pool(x).reshape(B * TILES, -1)
                x = self.vproj(x).reshape(B, TILES * 24)
            else:
                x = torch.zeros(B, 0, device=st.device)
            if USE_STATE:
                s = torch.relu(self.sproj(st.reshape(B, -1)))
            else:
                s = torch.zeros(B, 0, device=st.device)
            h = torch.cat([x, s], 1)
            mu, lv = self.head(h).chunk(2, 1)
            lv = lv.clamp(-6, 4)
            z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
            v = self.v(z).reshape(B * TILES, 32, POOL_H, POOL_W)
            v = self.to3(self.up(v))
            if v.shape[-2:] != (REC_H, REC_W):
                v = torch.nn.functional.interpolate(v, size=(REC_H, REC_W), mode="bilinear",
                                                    align_corners=False)
            rec_img = torch.sigmoid(v)                         # (3B, 3, 24, 32)
            rec_st = self.sdec(z)
            return mu, lv, z, rec_img, rec_st

    return VAE()


def vae_losses(vae, img_in, img_tgt, st, st_tgt, B, beta=1.0):
    """img_in/st = 编码器输入（t 时刻）；img_tgt/st_tgt = 目标（PRED_K==0 时就是输入本身）。"""
    import torch
    import torch.nn.functional as F
    mu, lv, z, ri, rs = vae(img_in, st, B)
    tgt = F.adaptive_avg_pool2d(img_tgt, (REC_H, REC_W))
    assert ri.shape == tgt.shape, f"重建形状不一致 {tuple(ri.shape)} vs {tuple(tgt.shape)}"
    rec_img = F.binary_cross_entropy(ri, tgt)
    if USE_STATE and STATE_W > 0:
        rec_st = F.mse_loss(rs, st_tgt.reshape(B, -1))
        loss = rec_img + STATE_W * rec_st
    else:
        rec_st = torch.zeros((), device=st.device)
        loss = rec_img
    kl = (-0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1)).mean() / (NS * SH)
    return loss + beta * kl, rec_img.item(), rec_st.item(), kl.item()


def sample_nll(vae, img_in, img_tgt, st, st_tgt, B, beta=1.0):
    """逐样本非一致性分数（越大越不像）：预测误差 + KL 项。"""
    import torch
    import torch.nn.functional as F
    mu, lv, z, ri, rs = vae(img_in, st, B)
    tgt = F.adaptive_avg_pool2d(img_tgt, (REC_H, REC_W))
    per_img = F.binary_cross_entropy(ri, tgt, reduction="none").flatten(1).mean(1).reshape(B, TILES).mean(1)
    per_kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1) / (NS * SH)
    out = per_img + beta * per_kl
    if USE_STATE and STATE_W > 0:
        out = out + STATE_W * F.mse_loss(rs, st_tgt.reshape(B, -1), reduction="none").mean(1)
    return out.detach().cpu().numpy()


def train_vae(Xi, Xt, Xs, Xst, iters, beta, device, seed, log_every=0):
    import torch
    torch.manual_seed(seed)
    vae = build_model(d=16).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=2e-3)
    n = len(Xi)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    for it in range(1, iters + 1):
        rng.shuffle(idx)
        b = idx[:64]
        img_in = prep_img(Xi[b], device)
        img_tg = prep_img(Xt[b], device)
        st = torch.from_numpy(Xs[b]).float().to(device)
        st_tg = torch.from_numpy(Xst[b]).float().to(device)
        loss, ri, rs, kl = vae_losses(vae, img_in, img_tg, st, st_tg, len(b), beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and it % log_every == 0:
            print(f"    it {it:5d}  img {ri:.4f}  st {rs:.4f}  kl {kl:.4f}")
    return vae, sum(p.numel() for p in vae.parameters())


def nll_of(vae, Xi, Xt, Xs, Xst, device, beta=1.0, bs=512):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(Xi), bs):
            b = slice(i, i + bs)
            img_in = prep_img(Xi[b], device)
            img_tg = prep_img(Xt[b], device)
            st = torch.from_numpy(Xs[b]).float().to(device)
            st_tg = torch.from_numpy(Xst[b]).float().to(device)
            out.append(sample_nll(vae, img_in, img_tg, st, st_tg, len(Xi[b]), beta))
    return np.concatenate(out)


def conformal_p(score, cal_scores):
    """p = (1 + #{cal >= score}) / (n + 1)；score 越大越不像。"""
    return (1.0 + (cal_scores[None, :] >= score[:, None]).sum(1)) / (len(cal_scores) + 1.0)


# ----------------------------------------------------------------- 分三段执行
def stage_prepare(a):
    """labeler 环境：定位任务、列 episode、抽标注 -> spec json"""
    name, episodes = resolve_task(a.dataset, TASK_KEY)
    print(f"任务: {name}")
    print(f"该任务的 episode 数: {len(episodes)}")
    if a.limit_episodes:
        episodes = episodes[:a.limit_episodes]
        print(f"限用 {len(episodes)} 集")
    segs_of_ep = {}
    for ep, _ in episodes:
        segs = load_segments(a.labels, ep)
        if segs:
            segs_of_ep[ep] = [dict(start_s=float(s["start_s"]), end_s=float(s["end_s"]),
                                   caption=s["caption"].strip().lower()) for s in segs]
    caps = sorted({s["caption"] for v in segs_of_ep.values() for s in v})
    print(f"有标注 episode {len(segs_of_ep)}；段数 {sum(len(v) for v in segs_of_ep.values())}；不同句子 {len(caps)}")
    json.dump(dict(task=name, segs={str(k): v for k, v in segs_of_ep.items()}, caps=caps),
              open(a.spec, "w"), ensure_ascii=False)
    print(f"写出 {a.spec}")


def stage_embed(a):
    """vllm 环境：句子 -> 句向量 -> npz"""
    import torch
    spec = json.load(open(a.spec))
    caps = spec["caps"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E = embed_captions(caps, a.encoder, dev)
    print(f"句向量 {E.shape}（bge-small-en-v1.5，L2 归一化）")
    np.savez(a.emb, caps=np.array(caps, dtype=object), emb=E)
    print(f"写出 {a.emb}")


def _decode_all(a, segs_of_ep, cap_index):
    """解码三视角 + 对齐关节状态，返回 (Xi, Xs, EP, CI)。CI 是句子在 caps 中的下标。"""
    Xi, Xt, EP, FIDX, CI = [], [], [], [], []
    for n_done, ep in enumerate(sorted(segs_of_ep), 1):
        segs = segs_of_ep[ep]
        st = load_states(a.dataset, ep)
        if st is None:
            continue
        views = decode_views(a.clips, ep, a.stride, keep_all=True)
        if views is None:
            print(f"  [skip] ep{ep:06d} 解码失败"); continue
        T = min((len(views) - 1) // a.stride + 1, (len(st) - 1) // a.stride + 1)
        lab, amb = frame_labels(segs, len(st))
        for i in range(T):
            f = i * a.stride
            if f >= len(views):
                continue
            if f >= len(st) or amb[f]:
                continue
            j = lab[f]
            if j < 0:
                continue
            tiles = views[[max(0, f - o) for o in IMG_OFFSETS]]      # (NT, 3, H, W, C)
            tf = min(f + PRED_K, len(views) - 1)                     # 越界取末帧
            tgt = views[[max(0, tf - o) for o in IMG_OFFSETS]]
            if tf <= f and PRED_K > 0:
                CLAMPED[0] += 1
            CLAMPED[1] += 1
            Xi.append(tiles); Xt.append(tgt); EP.append(ep); FIDX.append(f)
            CI.append(cap_index[segs[j]["caption"]])
        if n_done % 20 == 0 or n_done == len(segs_of_ep):
            print(f"  解码 {n_done}/{len(segs_of_ep)} 集，样本 {len(Xi)}")
    if not Xi:
        z0 = np.zeros((0, NT, 3, IMG_H, IMG_W, 3), np.uint8)
        return z0, z0, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    return (np.stack(Xi), np.stack(Xt), np.asarray(EP, np.int64),
            np.asarray(FIDX, np.int64), np.asarray(CI, np.int64))


def stage_run(a):
    """labeler 环境：聚类 -> 解码 -> 训 VAE -> 共形 p 值评估"""
    import collections
    import torch
    import random as _random
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = json.load(open(a.spec))
    bundle = np.load(a.emb, allow_pickle=True)
    caps = list(bundle["caps"]); E = bundle["emb"]
    segs_of_ep = {int(k): v for k, v in spec["segs"].items()}
    if a.limit_episodes:
        keep = sorted(segs_of_ep)[:a.limit_episodes]
        segs_of_ep = {k: segs_of_ep[k] for k in keep}
        print(f"[提示] --limit-episodes={a.limit_episodes}，实际只用 {len(segs_of_ep)} 集")
    print(f"任务: {spec['task']}   有标注 episode: {len(segs_of_ep)}   句子: {len(caps)}")

    # ---- 第 1 步 ----
    cap_lab, _ = spherical_kmeans(E, min(a.k, len(caps)))
    cap2clu = {c: int(l) for c, l in zip(caps, cap_lab)}
    sizes = collections.Counter(cap_lab.tolist())
    print("\n[第1步] 各类句子数:", dict(sorted(sizes.items())))
    for c in range(a.k):
        ex = [caps[i] for i in range(len(caps)) if cap_lab[i] == c][:2]
        print(f"  c{c} ({sizes.get(c,0)} 句): {' | '.join(x[:72] for x in ex)}")

    # ---- 取帧（带缓存，避免每个配置重解码 300 个视频）----
    cap_index = {c: i for i, c in enumerate(caps)}
    if os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        Xi, Xt, EP, FIDX, CI = z["Xi"], z["Xt"], z["EP"], z["FIDX"], z["CI"]
        print(f"复用解码缓存 {a.cache}：样本 {len(CI)}")
    else:
        Xi, Xt, EP, FIDX, CI = _decode_all(a, segs_of_ep, cap_index)
        np.savez(a.cache, Xi=Xi, Xt=Xt, EP=EP, FIDX=FIDX, CI=CI)
        print(f"解码缓存写出 {a.cache}")
    states = load_all_states(a.dataset, np.unique(EP))
    Xs = state_windows(states, EP, FIDX)
    Xst = state_target(states, EP, FIDX)
    if Xst is None:
        Xst = Xs.reshape(len(Xs), -1)
    print(f"输入 图像 {Xi.shape} 状态 {Xs.shape}；目标 图像 {Xt.shape} 状态 {Xst.shape}")
    if PRED_K > 0:
        print(f"目标越界被截到末帧的样本：{CLAMPED[0]}/{CLAMPED[1]} = {CLAMPED[0]/max(CLAMPED[1],1)*100:.1f}%")
    if len(CI) == 0:
        sys.exit("没有可用样本")
    Y = cap_lab[CI]
    import collections as _c
    print(f"\n样本 图像 {Xi.shape}  状态 {Xs.shape}  类别分布 {dict(_c.Counter(Y.tolist()))}")
    prefix = "img+state" if (USE_IMAGE and USE_STATE) else ("img-only" if USE_IMAGE else "state-only")
    print(f"输入配置: {prefix}  状态权重 {STATE_W if USE_STATE else '—'}  池化 {POOL_H}x{POOL_W}")

    # 关节状态标准化：各维量纲差两个数量级（有的维极差 0.001，有的 2.3），
    # 不归一化会让重建/打分被大范围维度独占。
    mu_s = Xs.reshape(-1, NS).mean(0)
    sd_s = Xs.reshape(-1, NS).std(0) + 1e-6
    Xs = ((Xs - mu_s) / sd_s).astype(np.float32)
    print(f"状态已标准化（各维均值绝对值 {np.abs(mu_s).mean():.3f}，标准差范围 "
          f"{sd_s.min():.4f}–{sd_s.max():.3f}）")

    # ---- 按 episode 划分 ----
    eps_sorted = sorted(set(EP.tolist())); _random.Random(0).shuffle(eps_sorted)
    n_val = max(1, int(round(len(eps_sorted) * 0.2)))
    val_eps = set(eps_sorted[:n_val])
    tr = np.array([e not in val_eps for e in EP]); va = ~tr
    print(f"按 episode 划分: 训练 {len(eps_sorted)-n_val} 集 / 留出 {n_val} 集；"
          f"样本 训练 {tr.sum()} / 留出 {va.sum()}")

    # ---- 第 2 步 ----
    models, params = {}, {}
    for c in range(a.k):
        m = tr & (Y == c)
        if m.sum() < 200:
            print(f"  [跳过] 类 {c}: 训练样本仅 {m.sum()}"); continue
        print(f"\n[第2步] 训练类 {c} 的 VAE（{m.sum()} 帧，{a.iters} 步）...")
        vae, np_ = train_vae(Xi[m], Xt[m], Xs[m], Xst[m], a.iters, a.beta, device, seed=c,
                             log_every=a.iters)
        models[c] = vae; params[c] = np_
    print(f"每个 VAE 参数 {list(params.values())}")

    # ---- 第 3 步 ----
    cal_scores = {}
    for c, vae in models.items():
        idx = np.where(tr & (Y == c))[0]
        cut = int(len(idx) * 0.7)
        cal_scores[c] = nll_of(vae, Xi[idx[:cut]], Xt[idx[:cut]], Xs[idx[:cut]], Xst[idx[:cut]], device, a.beta)
        rest = idx[cut:]
        if len(rest):
            print(f"  类 {c}: 训练段校准 {len(idx[:cut])} / 自检 {len(rest)} 的分数均值 "
                  f"{cal_scores[c].mean():.3f} vs "
                  f"{nll_of(vae, Xi[rest], Xt[rest], Xs[rest], Xst[rest], device, a.beta).mean():.3f}")
    idxv = np.where(va)[0]
    P = np.zeros((len(idxv), a.k))
    for c, vae in models.items():
        s = nll_of(vae, Xi[idxv], Xt[idxv], Xs[idxv], Xst[idxv], device, a.beta)
        P[:, c] = conformal_p(s, cal_scores[c])
    pred = P.argmax(1); yv = Y[idxv]
    acc = float((pred == yv).mean())
    rec = [float((pred[yv == c] == c).mean()) for c in range(a.k) if (yv == c).any()]
    bal = float(np.mean(rec)) if rec else 0.0
    cov = float(np.mean(P.max(1) >= 0.05))
    cm = np.zeros((a.k, a.k), int)
    for t_, p_ in zip(yv, pred):
        cm[t_, p_] += 1
    print(f"\n[第3步] 留出样本 {len(idxv)}，随机 {100/a.k:.1f}%")
    print(f"  准确率 {acc*100:.1f}%   平衡准确率 {bal*100:.1f}%")
    print(f"  各类召回 {[f'{r*100:.0f}%' for r in rec]}")
    print(f"  p>=0.05 的覆盖率 {cov*100:.1f}%")
    # 只在"有模型"的类上算平衡准确率（未建模的稀有类永远判不出来，会污染这个数）
    modeled = sorted(models)
    rb = [float((pred[yv == c] == c).mean()) for c in modeled if (yv == c).any()]
    bal_m = float(np.mean(rb)) if rb else 0.0
    print(f"  仅在有模型的类 {modeled} 上：平衡准确率 {bal_m*100:.1f}%（随机 {100/len(modeled):.1f}%）")
    print('  风险-覆盖率（把「愿意作答」的门槛提高）：')
    for tau in (0.02, 0.05, 0.1, 0.2, 0.3):
        keep = P.max(1) >= tau
        if keep.sum():
            a_ = float((pred[keep] == yv[keep]).mean())
            print(f"    tau={tau:<5} 覆盖 {keep.mean()*100:5.1f}%  该子集准确率 {a_*100:5.1f}%")
    print("  混淆矩阵(行=真值):\n", cm)
    json.dump(dict(task=spec["task"], n_episodes=len(segs_of_ep), n_samples=int(len(Y)),
                   k=a.k, cluster_sizes={int(k): int(v) for k, v in sizes.items()},
                   cluster_examples={int(c): [caps[i] for i in range(len(caps)) if cap_lab[i] == c][:3] for c in range(a.k)},
                   params_per_vae={int(k): int(v) for k, v in params.items()},
                   acc=acc, balanced_acc=bal, balanced_acc_modeled=bal_m,
                   per_class_recall=rec, modeled_classes=modeled,
                   coverage_at_p005=cov, random=1.0 / a.k, confusion=cm.tolist(),
                   stride=a.stride, iters=a.iters, beta=a.beta),
              open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n写出 {a.out}")


def _auroc(pos, neg):
    """rank-based AUROC：pos 应大于 neg。"""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 并列取平均秩
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _all_task_names(dataset):
    import pyarrow.parquet as pq
    rows = pq.read_table(os.path.join(dataset, "meta", "tasks.parquet")).to_pylist()
    out = set()
    for r in rows:
        s = list(r.values())[-1]
        if isinstance(s, str) and "Current subtask:" in s:
            head = s.split("Current subtask:")[0].strip()
            if head.lower().startswith("task:"):
                head = head[5:].strip()
            out.add(head)
    return sorted(out)


def _episodes_of(dataset, name, limit):
    import pyarrow.parquet as pq
    eps = []
    for p in sorted(glob.glob(os.path.join(dataset, "meta", "episodes", "chunk-*", "file-*.parquet"))):
        for r in pq.read_table(p, columns=["episode_index", "tasks"]).to_pylist():
            if any(name in t for t in r["tasks"]):
                eps.append(int(r["episode_index"]))
    return sorted(eps)[:limit]


def _decode_episodes(a, eps, stride, tag):
    """解码一组 episode 的 (图像, 状态)，不需要标注。用于 OOD 取帧。"""
    cache = f"/tmp/pilot_{tag}_{IMG_OFFSETS[0] if NT > 1 else 'single'}_s{stride}_k{PRED_K}.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"  复用 {cache}（{len(z['EP'])} 帧）")
        return z["Xi"], z["Xt"], z["EP"], z["FIDX"]
    Xi, Xt, EP, FIDX = [], [], [], []
    for ep in eps:
        st = load_states(a.dataset, ep)
        if st is None:
            continue
        views = decode_views(a.clips, ep, stride, keep_all=True)
        if views is None:
            continue
        T = min((len(views) - 1) // stride + 1, (len(st) - 1) // stride + 1)
        for i in range(T):
            f = i * stride
            if f >= len(views):
                continue
            tf = min(f + PRED_K, len(views) - 1)
            Xi.append(views[[max(0, f - o) for o in IMG_OFFSETS]])
            Xt.append(views[[max(0, tf - o) for o in IMG_OFFSETS]])
            EP.append(ep); FIDX.append(f)
    if not Xi:
        z0 = np.zeros((0, NT, 3, IMG_H, IMG_W, 3), np.uint8)
        return z0, z0, np.zeros(0, np.int64), np.zeros(0, np.int64)
    Xi = np.stack(Xi); Xt = np.stack(Xt)
    EP = np.asarray(EP, np.int64); FIDX = np.asarray(FIDX, np.int64)
    np.savez(cache, Xi=Xi, Xt=Xt, EP=EP, FIDX=FIDX)
    return Xi, Xt, EP, FIDX


def _render_grid(rows, path, cell=(128, 96), pad=6, header=26):
    """rows: list of (label, [np.uint8 HxWx3, ...])；每行左侧写标签。"""
    import cv2
    cw, ch = cell
    ncol = max(len(r[1]) for r in rows)
    W = pad + 150 + ncol * (cw + pad)
    H = header + len(rows) * (ch + pad) + pad
    canvas = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(canvas, "input(48x64)   target(24x32)   recon(24x32)", (pad, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    for r, (label, tiles) in enumerate(rows):
        y = header + r * (ch + pad)
        cv2.putText(canvas, label[:34], (pad, y + ch // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 0, 0), 1, cv2.LINE_AA)
        for c, im in enumerate(tiles):
            x = pad + 150 + c * (cw + pad)
            big = cv2.resize(im, (cw, ch), interpolation=cv2.INTER_NEAREST)
            canvas[y:y + ch, x:x + cw] = big[:, :, ::-1] if big.shape[2] == 3 else big
    cv2.imwrite(path, canvas)
    print(f"写出 {path}  ({W}x{H})")


def stage_recon(a):
    """把输入 / 目标 / VAE 复原图并排画出来，看它到底学到了什么。"""
    import torch
    import torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = json.load(open(a.spec))
    segs_of_ep = {int(k): v for k, v in spec["segs"].items()}
    bundle = np.load(a.emb, allow_pickle=True)
    caps = list(bundle["caps"]); E = bundle["emb"]
    cap_lab, _ = spherical_kmeans(E, min(a.k, len(caps)))
    z = np.load(a.cache, allow_pickle=True)
    Xi, Xt, EP, FIDX, CI = z["Xi"], z["Xt"], z["EP"], z["FIDX"], z["CI"]
    st = load_all_states(a.dataset, np.unique(EP))
    Xs = state_windows(st, EP, FIDX)
    Xst = state_target(st, EP, FIDX)
    mu_s = Xs.reshape(-1, NS).mean(0); sd_s = Xs.reshape(-1, NS).std(0) + 1e-6
    Xs = ((Xs - mu_s) / sd_s).astype(np.float32)
    if Xst is None:
        Xst = Xs.reshape(len(Xs), -1)
    Y = cap_lab[CI]
    eps_sorted = sorted(set(EP.tolist()))
    import random as _r
    _r.Random(0).shuffle(eps_sorted)
    val = set(eps_sorted[:max(1, int(round(len(eps_sorted) * 0.2)))])
    tr = np.array([e not in val for e in EP])

    for c in a.recon_classes:
        m = tr & (Y == c)
        if m.sum() < 200:
            print(f"[跳过] 类 {c} 训练样本不足"); continue
        print(f"\n[类 {c}] 训练 {m.sum()} 帧 ...")
        vae, npar = train_vae(Xi[m], Xt[m], Xs[m], Xst[m], a.iters, a.beta, device, seed=c,
                              log_every=a.iters)
        mt = (~tr) & (Y == c)
        idx = np.where(mt)[0]
        if len(idx) == 0:
            continue
        s = nll_of(vae, Xi[idx], Xt[idx], Xs[idx], Xst[idx], device, a.beta)
        order = np.argsort(s)
        picks = [(f"typical #{k+1} s={s[order[k]]:.3f}", idx[order[k]]) for k in range(min(3, len(idx)))]
        picks += [(f"atypical #{k+1} s={s[order[-1-k]]:.3f}", idx[order[-1-k]]) for k in range(min(3, len(idx)))]
        rows = []
        for label, i in picks:
            img_in = prep_img(Xi[i:i + 1], device)
            with torch.no_grad():
                mu, lv, zz, ri, rs = vae(img_in, torch.from_numpy(Xs[i:i + 1]).to(device), 1)
                tgt = F.adaptive_avg_pool2d(img_in, (REC_H, REC_W))
            for v in range(TILES):
                inp = Xi[i, v % NT, v // NT]
                tg = (tgt[v].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                rc = (ri[v].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                rows.append((f"{label} v{v}" if v == 0 else f"v{v}", [inp, tg, rc]))
        _render_grid(rows, f"/tmp/recon_c{c}.png")
        # 每类再画一张"平均图"：看他是不是只在学平均
        N = min(64, len(idx))
        ii = idx[:N]
        with torch.no_grad():
            img_in = prep_img(Xi[ii], device)
            mu, lv, zz, ri, rs = vae(img_in, torch.from_numpy(Xs[ii]).to(device), N)
            tgt = F.adaptive_avg_pool2d(img_in, (REC_H, REC_W))
        mean_in = (Xi[ii].astype(np.float32).mean(0)).astype(np.uint8)
        mean_tg = (tgt.view(N, TILES, 3, REC_H, REC_W).mean(0).permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        mean_rc = (ri.view(N, TILES, 3, REC_H, REC_W).mean(0).permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        rows = [(f"MEAN v{v}", [mean_in[v % NT, v // NT], mean_tg[v], mean_rc[v]]) for v in range(TILES)]
        _render_grid(rows, f"/tmp/recon_mean_c{c}.png")


def stage_ood(a):
    """三路分流里的第 3 路判决：喂别的任务的帧，看能不能正确"说不会"。"""
    import collections
    import random as _random
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = json.load(open(a.spec))
    bundle = np.load(a.emb, allow_pickle=True)
    caps = list(bundle["caps"]); E = bundle["emb"]
    segs_of_ep = {int(k): v for k, v in spec["segs"].items()}
    if a.limit_episodes:
        keep = sorted(segs_of_ep)[:a.limit_episodes]
        segs_of_ep = {k: segs_of_ep[k] for k in keep}
    task_name = spec["task"]

    cap_lab, _ = spherical_kmeans(E, min(a.k, len(caps)))
    cap2clu = {c: int(l) for c, l in zip(caps, cap_lab)}
    print(f"[ID] 任务: {task_name}")
    print(f"[ID] 各类句子数: {dict(sorted(collections.Counter(cap_lab.tolist()).items()))}")

    cap_index = {c: i for i, c in enumerate(caps)}
    if os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        Xi, Xt, EP, FIDX, CI = z["Xi"], z["Xt"], z["EP"], z["FIDX"], z["CI"]
    else:
        Xi, Xt, EP, FIDX, CI = _decode_all(a, segs_of_ep, cap_index)
        np.savez(a.cache, Xi=Xi, Xt=Xt, EP=EP, FIDX=FIDX, CI=CI)
    st_id = load_all_states(a.dataset, np.unique(EP))
    Xs = state_windows(st_id, EP, FIDX)
    Xst = state_target(st_id, EP, FIDX)
    if Xst is None:
        Xst = Xs.reshape(len(Xs), -1)
    print(f"输入 图像 {Xi.shape} 状态 {Xs.shape}；目标 图像 {Xt.shape} 状态 {Xst.shape}")
    Y = cap_lab[CI]
    mu_s = Xs.reshape(-1, NS).mean(0); sd_s = Xs.reshape(-1, NS).std(0) + 1e-6
    Xs = ((Xs - mu_s) / sd_s).astype(np.float32)
    print(f"[ID] 帧 {len(Y)}  类别分布 {dict(collections.Counter(Y.tolist()))}")

    eps_sorted = sorted(set(EP.tolist())); _random.Random(0).shuffle(eps_sorted)
    n_val = max(1, int(round(len(eps_sorted) * 0.2)))
    val_eps = set(eps_sorted[:n_val])
    tr = np.array([e not in val_eps for e in EP]); va = ~tr

    # 每类 VAE + 任务级 VAE
    models, params = {}, {}
    for c in range(a.k):
        m = tr & (Y == c)
        if m.sum() < 200:
            continue
        models[c], params[c] = train_vae(Xi[m], Xt[m], Xs[m], Xst[m], a.iters, a.beta, device, seed=c)
    print(f"[ID] 子任务级 VAE {len(models)} 个，参数 {list(params.values())}")
    mt = tr
    task_vae, npt = train_vae(Xi[mt], Xt[mt], Xs[mt], Xst[mt], a.iters, a.beta, device, seed=99)
    print(f"[ID] 任务级 VAE 参数 {npt}")

    # 校准（训练集的 70%）与 ID 测试（留出 episode）
    idx_tr = np.where(tr)[0]
    cut = int(len(idx_tr) * 0.7)
    cal_idx, rest_idx = idx_tr[:cut], idx_tr[cut:]
    cal = {c: nll_of(m, Xi[cal_idx], Xt[cal_idx], Xs[cal_idx], Xst[cal_idx], device, a.beta)
           for c, m in models.items()}
    cal_task = nll_of(task_vae, Xi[cal_idx], Xt[cal_idx], Xs[cal_idx], Xst[cal_idx], device, a.beta)
    idxv = np.where(va)[0]
    print(f"[ID] 校准帧 {len(cal_idx)}  留出帧 {len(idxv)}")

    # OOD：别的任务
    names = _all_task_names(a.dataset)
    keys = [s.strip() for s in a.ood_tasks.split(",") if s.strip()]
    ood_names = []
    for k_ in keys:
        kk = k_.replace("_", " ").lower()
        hit = [n for n in names if kk in n.lower() and kk not in task_name.lower()]
        ood_names += hit[:1]
    ood_names = [n for n in ood_names if n]
    print(f"\n[OOD] 使用 {len(ood_names)} 个其它任务：")
    OX, OXt, OE, OF, OT = [], [], [], [], []
    for i, n in enumerate(ood_names):
        eps = _episodes_of(a.dataset, n, a.ood_episodes)
        ox, oxt, oe, of_ = _decode_episodes(a, eps, a.ood_stride, f"ood{i}")
        if len(oe) == 0:
            print(f"  [skip] {n[:60]}"); continue
        print(f"  {n[:66]:68} {len(oe):5d} 帧 ({len(eps)} 集)")
        OX.append(ox); OXt.append(oxt); OE.append(oe); OF.append(of_); OT += [n] * len(oe)
    if not OX:
        sys.exit("没有 OOD 数据")
    OX = np.concatenate(OX); OXt = np.concatenate(OXt)
    OE = np.concatenate(OE); OF = np.concatenate(OF)
    OT = np.array(OT, dtype=object)
    # 关键：OOD 必须用与 ID **完全相同**的历史帧偏移，否则时间跨度不同会污染判别
    st_ood = load_all_states(a.dataset, np.unique(OE))
    OS = ((state_windows(st_ood, OE, OF) - mu_s) / sd_s).astype(np.float32)
    OSt = state_target(st_ood, OE, OF)
    OSt = ((OSt - mu_s) / sd_s).astype(np.float32) if OSt is not None else OS.reshape(len(OS), -1)
    print(f"OOD 输入 图像 {OX.shape} 状态 {OS.shape}；目标 图像 {OXt.shape} 状态 {OSt.shape}")
    print(f"[OOD] 合计 {len(OX)} 帧")

    # 打分
    def scores(Xi_, Xt_, Xs_, Xst_):
        pk = np.zeros((len(Xs_), a.k))
        for c, m in models.items():
            pk[:, c] = conformal_p(nll_of(m, Xi_, Xt_, Xs_, Xst_, device, a.beta), cal[c])
        pt = conformal_p(nll_of(task_vae, Xi_, Xt_, Xs_, Xst_, device, a.beta), cal_task)
        return pk, pt

    pk_id, pt_id = scores(Xi[idxv], Xt[idxv], Xs[idxv], Xst[idxv])
    pk_ood, pt_ood = scores(OX, OXt, OS, OSt)
    maxp_id, maxp_ood = pk_id.max(1), pk_ood.max(1)

    auroc_sub = _auroc(maxp_id, maxp_ood)      # ID 的 max p 应高于 OOD
    auroc_task = _auroc(pt_id, pt_ood)
    print(f"\n=== OOD 判别（ID 留出 {len(maxp_id)} 帧 vs OOD {len(maxp_ood)} 帧）===")
    print(f"AUROC(子任务级 max p): {auroc_sub:.3f}   AUROC(任务级 p): {auroc_task:.3f}   (0.5 = 完全分不开)")
    print(f"ID  max p 中位 {np.median(maxp_id):.3f}   任务级 p 中位 {np.median(pt_id):.3f}")
    print(f"OOD max p 中位 {np.median(maxp_ood):.3f}   任务级 p 中位 {np.median(pt_ood):.3f}")

    np.savez("/tmp/pilot_ood_scores.npz", maxp_id=maxp_id, pt_id=pt_id,
             maxp_ood=maxp_ood, pt_ood=pt_ood, OT=OT)
    print("分数已落盘 /tmp/pilot_ood_scores.npz")

    print("\n拒绝前沿（只用子任务级 max p）：")
    print("   τ_sub    ID 接受    OOD 误收")
    for tau in (0.1, 0.3, 0.5, 0.7, 0.9):
        print(f"   {tau:<6} {(maxp_id>=tau).mean()*100:7.1f}%   {(maxp_ood>=tau).mean()*100:7.1f}%")
    print("\n拒绝前沿（只用任务级 p）：")
    print("   τ_task   ID 保留    OOD 误收")
    for tau in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        print(f"   {tau:<6} {(pt_id>=tau).mean()*100:7.1f}%   {(pt_ood>=tau).mean()*100:7.1f}%")

    # 三路分流：在网格上选阈值，约束 OOD 误收 ≤5%
    best = None
    for taus in np.linspace(0.01, 0.9, 45):
        for taut in np.linspace(0.01, 0.9, 45):
            fa = float(np.mean(~((maxp_ood < taus) & (pt_ood < taut))))   # OOD 被分到子任务或总任务的比例
            if fa > 0.05:
                continue
            cap_ok = float(np.mean((maxp_ood < taus) & (pt_ood < taut)))
            id_sub = float(np.mean(maxp_id >= taus))
            id_task = float(np.mean((maxp_id < taus) & (pt_id >= taut)))
            id_cap = float(np.mean((maxp_id < taus) & (pt_id < taut)))
            score = id_sub * cap_ok
            if best is None or score > best[0]:
                best = (score, taus, taut, cap_ok, fa, id_sub, id_task, id_cap)
    if best is None:
        print("在 5% 误收约束下网格内无解（用上面的前沿表另选工作点）")
        best = (0.0, float("nan"), float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"), float("nan"))
    _, taus, taut, cap_ok, fa, id_sub, id_task, id_cap = best
    print(f"\n三路分流（阈值 τ_sub={taus:.2f}, τ_task={taut:.2f}，取 OOD 误收≤5% 中最好的一档）：")
    print(f"  OOD  → 判为'不认识'→ 升级 CAP : {cap_ok*100:5.1f}%   误收 {fa*100:4.1f}%")
    print(f"  ID   → 子任务 : {id_sub*100:5.1f}%    退回总任务 : {id_task*100:5.1f}%    误升级 CAP : {id_cap*100:5.1f}%")

    # 每个 OOD 任务的任务级 p 中位数（看哪些任务最难分）
    print("\n各家 OOD 任务的任务级 p 中位数（越低越好分）：")
    for n in ood_names:
        m = OT == n
        if m.sum():
            print(f"  {n[:64]:66} p_task 中位 {np.median(pt_ood[m]):.3f}  max p_sub 中位 {np.median(maxp_ood[m]):.3f}")

    print("\n拒绝前沿（只用子任务级 max p，两路：接受/退回）：")
    print("   τ_sub    ID 接受    OOD 误收")
    for tau in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        print(f"   {tau:<6} {(maxp_id>=tau).mean()*100:7.1f}%   {(maxp_ood>=tau).mean()*100:7.1f}%")
    print("\n拒绝前沿（只用任务级 p，两路：认作本任务 / 升级）：")
    print("   τ_task   ID 保留    OOD 误收")
    for tau in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        print(f"   {tau:<6} {(pt_id>=tau).mean()*100:7.1f}%   {(pt_ood>=tau).mean()*100:7.1f}%")

    json.dump(dict(task=task_name, auroc_subtask=auroc_sub, auroc_task=auroc_task,
                   tau_sub=float(taus), tau_task=float(taut),
                   ood_cap_rate=cap_ok, ood_false_accept=fa,
                   id_subtask_rate=id_sub, id_task_rate=id_task, id_cap_rate=id_cap,
                   n_id=int(len(maxp_id)), n_ood=int(len(maxp_ood)),
                   ood_tasks=ood_names), open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n写出 {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--clips", default=DEFAULT_CLIPS)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--stage", default="run", choices=["prepare", "embed", "run", "ood", "recon"])
    ap.add_argument("--recon-classes", default="0,4", help="要画复原图的类，逗号分隔")
    # 刻意混难易：鼠标/键盘、挂杯子、笔筒最像本任务；井字棋、倒球最不像
    ap.add_argument("--ood-tasks", default="mouse pad,mug rack,pen holder,eggs from the basket,"
                                        "fold the clothes,throw them into the dustbin,tic-tac-toe,balls from the cup",
                    help="逗号分隔的任务名关键词（下划线会自动转空格），用它们的帧做 OOD")
    ap.add_argument("--ood-episodes", type=int, default=6)
    ap.add_argument("--ood-stride", type=int, default=10)
    ap.add_argument("--spec", default="/tmp/pilot_spec.json")
    ap.add_argument("--emb", default="/tmp/pilot_emb.npz")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--limit-episodes", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-image", action="store_true", help="消融：只用关节状态，不用图像")
    ap.add_argument("--pool", default="3x4", help="编码器池化后的空间尺寸，如 1x1 / 3x4 / 6x8；越大越不激进")
    ap.add_argument("--no-state", action="store_true", help="消融：完全不输入关节角度（只用图像）")
    ap.add_argument("--state-weight", type=float, default=4.0,
                    help="关节状态重建项的权重（同时作用于非一致性分数）；默认 2.5")
    ap.add_argument("--cache", default="", help="解码缓存路径；留空按 stride/集数自动命名")
    ap.add_argument("--pred-k", type=int, default=0,
                    help="预测步长：0 = 重构输入本身；>0 = 预测 t+k 的观测（越界取末帧）")
    ap.add_argument("--img-hist", default="vlm3", choices=["vlm3", "single"],
                    help="图像时间点：vlm3 = 3 相机 × (t,t-12,t-50) 共 9 格；single = 只看当前帧 3 格")
    ap.add_argument("--hist", default="vlm3", choices=["vlm3", "window4", "single"],
                    help="历史方案：vlm3 = t/t-12/t-50（与 VLM 规格一致）；window4 = 连续 0.6s 四帧；single = 只看当前帧")
    ap.add_argument("--out", default="/tmp/pilot_store_laptop.json")
    a = ap.parse_args()
    global USE_IMAGE, USE_STATE, STATE_W, POOL_H, POOL_W
    set_hist(a.hist)
    set_img_hist(a.img_hist)
    set_pred_k(a.pred_k)
    check_prep_img()
    USE_IMAGE = not a.no_image
    USE_STATE = not a.no_state
    STATE_W = a.state_weight
    if not USE_IMAGE and not USE_STATE:
        sys.exit("--no-image 与 --no-state 不能同时开：那样没有任何输入")
    if not a.cache:
        a.cache = f"/tmp/pilot_frames_{a.img_hist}_s{a.stride}_k{PRED_K}_n{a.limit_episodes or 'all'}.npz"
    POOL_H, POOL_W = (int(v) for v in a.pool.lower().split("x"))
    if a.smoke:
        a.limit_episodes = a.limit_episodes or 8
        a.stride = 20
        a.iters = 200
    if a.stage == "recon":
        a.recon_classes = [int(x) for x in a.recon_classes.split(",") if x.strip()]
    {"prepare": stage_prepare, "embed": stage_embed, "run": stage_run,
     "ood": stage_ood, "recon": stage_recon}[a.stage](a)


if __name__ == "__main__":
    main()
