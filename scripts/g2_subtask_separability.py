#!/usr/bin/env python3
"""G2 · 子任务可分性探针（go/no-go 判决实验）

目的（见 docs/05 §2.2）：在**不训练 VAE** 的前提下，先回答"子任务信息在不在观测里"。
若冻结特征 + 线性探针都分不开子任务，那么 350 个 VAE 也学不出来 —— 项目当场转向。

做三件事：
  1. 从 Flora 段级标注构造"每条 caption → 子任务簇 id"（句向量 K-means，可用 tf-idf 兜底）；
  2. 用**关节状态**（默认）和可选的**图像**特征，按 episode 不相交切分，训线性探针；
  3. 报 balanced accuracy / 每类召回 / 混淆矩阵 / 随机基线 1/K。

用法（6c，labeler 环境）：
    P=/hdfs/users/peize/vlm_vla/env/labeler/bin/python
    $P g2_subtask_separability.py --dry-run
    $P g2_subtask_separability.py --n-episodes 200 --k 8

判读（docs/05 §2.2）：
    bal.acc <= 1/K + 5pt   -> 判负，转 docs/04 §5
    bal.acc >= 1/K + 15pt  -> 判正，进入 G3
    中间 -> 灰区，加测边界指标后再定
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
TASK_OF = {}
FPS = 25
MIN_SEG_S = 0.2   # 碎段并入前一段（docs/03 §2.1 规则 4）


# ---------------------------------------------------------------- labels
def load_segments(labels_dir, episodes):
    """episodes 是整型 id 列表（ep000123 -> 123），与 lerobot 的 episode_index 对齐。"""
    out = {}
    for ep in episodes:
        p = os.path.join(labels_dir, f"ep{ep:06d}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        segs = [s for s in d.get("segments", []) if s.get("caption", "").strip()]
        if segs:
            out[ep] = segs
    return out


def build_frame_labels(segs, n_frames):
    """帧 -> 子任务簇 id；重叠区返回 -1（歧义，剔除），空隙归最近段。

    返回 (labels[n_frames], ambig[n_frames])。labels 里存的是"段序"，簇 id 在后面再映射。
    """
    n = len(segs)
    labels = np.full(n_frames, -1, dtype=np.int64)
    cover = np.zeros(n_frames, dtype=np.int16)
    for i, s in enumerate(segs):
        a = int(round(s["start_s"] * FPS))
        b = int(round(s["end_s"] * FPS))
        a = max(0, min(a, n_frames))
        b = max(0, min(b, n_frames))
        labels[a:b] = i
        cover[a:b] += 1
    # 空隙帧 -> 最近段
    known = np.where(labels >= 0)[0]
    if len(known) == 0:
        return labels, np.ones(n_frames, dtype=bool)
    unknown = np.where(labels < 0)[0]
    if len(unknown):
        idx = np.searchsorted(known, unknown)
        idx = np.clip(idx, 0, len(known) - 1)
        left = known[np.clip(idx - 1, 0, len(known) - 1)]
        right = known[idx]
        pick = np.where(np.abs(unknown - left) <= np.abs(right - unknown), left, right)
        labels[unknown] = labels[pick]
    ambig = cover > 1
    return labels, ambig


def merge_short_segments(segs):
    """<20 帧的碎段并入前一段（只影响索引，不改原始标注）。返回 seg->canonical 段序映射。"""
    canon = list(range(len(segs)))
    for i, s in enumerate(segs):
        if (s["end_s"] - s["start_s"]) < MIN_SEG_S and i > 0:
            canon[i] = canon[i - 1]
    return canon


# ---------------------------------------------------------------- text -> clusters
def tfidf_embed(texts):
    import re
    docs = [re.findall(r"[a-z]+", t.lower()) for t in texts]
    df = collections.Counter()
    for d in docs:
        df.update(set(d))
    vocab = {t: i for i, t in enumerate(sorted(df))}
    X = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for i, d in enumerate(docs):
        for t in d:
            X[i, vocab[t]] += 1
    idf = np.array([np.log((1 + len(docs)) / (1 + df[t])) + 1 for t in sorted(df)], dtype=np.float32)
    X *= idf
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X, "tfidf(1gram)"


def dense_embed(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-small-en-v1.5")
    X = m.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    return np.asarray(X, dtype=np.float32), "bge-small-en-v1.5"


def encode(texts, mode):
    if mode in ("auto", "dense"):
        try:
            return dense_embed(texts)
        except Exception as e:
            if mode == "dense":
                raise
            print(f"[warn] dense encoder unavailable ({type(e).__name__}), falling back to tf-idf")
    return tfidf_embed(texts)


def spherical_kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), dtype=np.int64)
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


# ---------------------------------------------------------------- state features
def load_states(dataset, episodes, stride, h):
    """读 lerobot parquet，返回 {ep: (feats[T,D], frame_idx[T])}。特征 = 历史窗口统计。"""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(dataset, "data", "chunk-*", "*.parquet")))
    want = set(episodes)
    per_ep = collections.defaultdict(list)
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state", "task_index"])
        epi = np.asarray(t["episode_index"].to_pylist())
        if not want.intersection(set(epi.tolist())):
            continue
        fi = np.asarray(t["frame_index"].to_pylist())
        st = np.stack([np.asarray(x, dtype=np.float32) for x in t["observation.state"].to_pylist()])
        ti = np.asarray(t["task_index"].to_pylist())
        global TASK_OF
        for e in np.unique(epi):
            if int(e) not in want:
                continue
            m = epi == e
            o = np.argsort(fi[m])
            per_ep[int(e)].append((fi[m][o], st[m][o]))
            TASK_OF[int(e)] = int(ti[m][0])
    feats = {}
    for e, chunks in per_ep.items():
        fi = np.concatenate([c[0] for c in chunks])
        st = np.concatenate([c[1] for c in chunks])
        T = len(fi)
        rows, idx = [], []
        for t in range(h - 1, T, stride):
            w = st[t - h + 1: t + 1]
            rows.append(np.concatenate([st[t], w.mean(0), w.std(0), st[t] - w[0]]))
            idx.append(t)
        if rows:
            feats[e] = (np.stack(rows), np.asarray(idx, dtype=np.int64))
    return feats


# ---------------------------------------------------------------- probe
def balanced_acc(y, p, k):
    rec = []
    for c in range(k):
        m = y == c
        if m.sum():
            rec.append((p[m] == c).mean())
    return float(np.mean(rec)) if rec else 0.0


def train_probe(Xtr, ytr, Xte, yte, k, iters=800, seed=0):
    import torch
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    net = torch.nn.Linear(d, k)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    # 类别均衡权重
    cnt = np.bincount(ytr, minlength=k).astype(np.float64)
    w = torch.tensor((cnt.sum() / (k * np.maximum(cnt, 1))).astype(np.float32))
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(Xtr_t), ytr_t, weight=w)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(Xte_t).argmax(1).numpy()
    return pred, float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--k", type=int, default=8, help="子任务簇数 N")
    ap.add_argument("--stride", type=int, default=5, help="帧采样步长")
    ap.add_argument("--h", type=int, default=3, help="历史窗口帧数")
    ap.add_argument("--encoder", default="auto", choices=["auto", "dense", "tfidf"])
    ap.add_argument("--out", default="")
    ap.add_argument("--split", default="episode", choices=["episode", "task"],
                    help="episode=按集不相交(默认); task=按任务不相交(更严格, 排除任务身份泄漏)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # ep000123.json -> 123，必须整型，才能与 parquet 的 episode_index 对上
    eps = sorted(int(os.path.basename(p)[2:-5])
                 for p in glob.glob(os.path.join(a.labels, "ep*.json")))
    print(f"label files: {len(eps)}")
    if not eps:
        sys.exit("no labels found")
    eps = eps[: a.n_episodes]
    segs = load_segments(a.labels, eps)
    print(f"episodes with segments: {len(segs)}")

    if a.dry_run:
        feats = load_states(a.dataset, list(segs)[:4], a.stride, a.h)
        print(f"dry-run states: {len(feats)} episodes")
        for e, (F, I) in list(feats.items())[:4]:
            print(f"  ep{e:06d}: feats {F.shape} frames {len(I)} segments {len(segs[e])}")
        texts = sorted({s["caption"].strip().lower() for v in segs.values() for s in v})
        print(f"distinct captions: {len(texts)}  e.g. {texts[:2]}")
        enc_mode = "tfidf" if a.encoder == "tfidf" else "auto"
        X, used = encode(texts, enc_mode)
        print(f"encoder used: {used}  X={X.shape}")
        print("dry-run OK")
        return

    # 1) caption -> cluster
    cap_list = sorted({s["caption"].strip().lower() for v in segs.values() for s in v})
    X, used = encode(cap_list, a.encoder)
    print(f"encoder: {used}  captions: {len(cap_list)}  dim: {X.shape[1]}")
    cap_lab, _ = spherical_kmeans(X, min(a.k, len(cap_list)))
    cap2clu = {c: int(l) for c, l in zip(cap_list, cap_lab)}
    sizes = collections.Counter(cap_lab.tolist())
    print("cluster sizes:", dict(sorted(sizes.items())))

    # 2) state features
    feats = load_states(a.dataset, list(segs), a.stride, a.h)
    print(f"episodes with states: {len(feats)}")

    # 3) 帧 -> 簇
    ep_train, ep_test = [], []
    test_eps = set()
    all_tasks = sorted({TASK_OF[e] for e in feats if e in TASK_OF})
    heldout_tasks = set(all_tasks[::5])   # 每 5 个任务取 1 个作测试任务
    if a.split == "task":
        print(f"task-disjoint: {len(heldout_tasks)}/{len(all_tasks)} tasks held out")
    Xs, ys, es = [], [], []
    for i, (e, (F, I)) in enumerate(sorted(feats.items())):
        seg = segs[e]
        canon = merge_short_segments(seg)
        lab, ambig = build_frame_labels(seg, int(I.max()) + 1 + a.h)
        fl = lab[I]
        fa = ambig[I]
        keep = (fl >= 0) & (~fa)
        if not keep.any():
            continue
        clu = np.array([cap2clu[seg[canon[j]]["caption"].strip().lower()] for j in fl[keep]])
        Xs.append(F[keep]); ys.append(clu); es.append(np.full(keep.sum(), i))
        if a.split == "task":
            is_test = TASK_OF.get(e, -1) in heldout_tasks
        else:
            is_test = (i % 5 == 0)
        if is_test:
            ep_test.append(e); test_eps.add(i)
        else:
            ep_train.append(e)
    if not Xs:
        sys.exit("no usable frames")
    X = np.concatenate(Xs); y = np.concatenate(ys); grp = np.concatenate(es)
    print(f"frames used: {len(y)}  dropped(overlap/none): {sum(len(v[1]) for v in feats.values()) - len(y)}")

    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    te = np.isin(grp, list(test_eps))
    if te.sum() < 50 or (~te).sum() < 50:
        sys.exit(f"split too small: train {(~te).sum()} test {te.sum()}")
    pred, loss = train_probe(X[~te], y[~te], X[te], y[te], a.k)

    base = 1.0 / a.k
    ba = balanced_acc(y[te], pred, a.k)
    acc = float((pred == y[te]).mean())
    print(f"\n--- G2 result (k={a.k}, {used}, h={a.h}, stride={a.stride}) ---")
    print(f"train frames {(~te).sum()}  test frames {te.sum()}  episodes train {len(ep_train)} test {len(ep_test)}")
    print(f"random baseline : {base*100:5.1f}%")
    print(f"test accuracy   : {acc*100:5.1f}%")
    print(f"balanced acc    : {ba*100:5.1f}%   (delta vs random {(ba-base)*100:+.1f} pt)")
    cm = np.zeros((a.k, a.k), dtype=int)
    for t_, p_ in zip(y[te], pred):
        cm[t_, p_] += 1
    print("confusion (row=true):")
    print(cm)
    per = [f"{(pred[y[te]==c]==c).mean()*100:4.0f}" if (y[te]==c).any() else "  --" for c in range(a.k)]
    print("per-class recall %:", per)

    verdict = "PASS" if (ba - base) >= 0.15 else ("FAIL" if (ba - base) <= 0.05 else "GREY")
    print(f"\nVERDICT: {verdict}  (docs/05 §2.2: >=+15pt PASS, <=+5pt FAIL)")
    if a.out:
        json.dump(dict(k=a.k, encoder=used, h=a.h, stride=a.stride, acc=acc,
                       balanced_acc=ba, random=base, verdict=verdict,
                       confusion=cm.tolist(), n_test=int(te.sum())),
                  open(a.out, "w"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
