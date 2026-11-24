#!/usr/bin/env python3
"""标签结构实测：为 docs/02 与 docs/04 里的数字提供可复现来源。

只读，不写任何文件。仅依赖 numpy（本机无 sklearn/torch）。

用法:
    python3 docs/scripts/caption_structure.py
    python3 docs/scripts/caption_structure.py --root /path/to/samples35

数据源: flora_samples35/samples35/taskNN/segments.json
        （35 个任务各 1 集，schema flora_demo_annotation_v1）
"""

import argparse
import collections
import glob
import json
import math
import os
import re
import statistics as st

import numpy as np


def load(root):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "task*/segments.json"))):
        task = os.path.basename(os.path.dirname(p))
        d = json.load(open(p))
        segs = d["segments"]
        caps = [s["caption"].strip() for s in segs]
        overlap = sum(1 for a, b in zip(segs, segs[1:]) if b["start_s"] < a["end_s"] - 1e-6)
        rows.append(
            dict(
                task=task,
                dur=d["duration_s"],
                segs=segs,
                caps=caps,
                n=len(segs),
                distinct=len({c.lower() for c in caps}),
                overlap=overlap,
            )
        )
    return rows


def table_per_task(rows):
    print(f"{'task':8} {'dur_s':>6} {'segs':>5} {'distinct':>8} {'repeat':>7} {'ovl':>4}")
    for r in rows:
        rep = (1 - r["distinct"] / r["n"]) * 100
        print(f"{r['task']:8} {r['dur']:6.1f} {r['n']:5d} {r['distinct']:8d} {rep:6.0f}% {r['overlap']:4d}")

    ns = [r["n"] for r in rows]
    ds = [r["distinct"] for r in rows]
    print()
    print(f"segments/task      : mean {st.mean(ns):.1f} median {st.median(ns)} "
          f"min {min(ns)} max {max(ns)} total {sum(ns)}")
    print(f"distinct/task      : mean {st.mean(ds):.1f} median {st.median(ds)} "
          f"min {min(ds)} max {max(ds)}")
    print(f"intra-episode repeat: {(1 - sum(ds) / sum(ns)) * 100:.1f}%  "
          f"(=> 集内子任务几乎是单向推进的时序相位, 不是重复的语义类)")
    print(f"episodes with >=1 segment overlap: {sum(1 for r in rows if r['overlap'])}/{len(rows)}, "
          f"overlapping boundaries {sum(r['overlap'] for r in rows)}")


def table_corpus(rows):
    allcaps = [c.lower() for r in rows for c in r["caps"]]
    uniq = set(allcaps)
    print()
    print(f"segments {len(allcaps)} / distinct captions {len(uniq)} "
          f"= {len(uniq) / len(allcaps) * 100:.1f}% unique within this 35-episode sample")
    verbl = collections.Counter(re.findall(r"[a-z]+", c)[0] for c in allcaps if re.findall(r"[a-z]+", c))
    print("first word (verb family):", verbl.most_common(10))
    seglens = [s["end_s"] - s["start_s"] for r in rows for s in r["segs"]]
    print(f"seg duration: mean {st.mean(seglens):.2f}s median {st.median(seglens):.2f} "
          f"min {min(seglens):.2f} max {max(seglens):.2f}")
    print("NOTE 全库口径 (f35, 3,500 集): 31,968 段 / 12,242 条不同 prompt "
          f"=> 唯一率 {12242 / 31968 * 100:.1f}%, 平均每条 prompt 承载 {31968 / 12242:.2f} 段")
    print("     即: 跨集高度复用(语义类), 集内几乎不复用(时序相位) -> docs/02 §2 的分叉")


def tfidf(docs):
    df = collections.Counter()
    for d in docs:
        df.update(set(d))
    vocab = {t: i for i, t in enumerate(sorted(df))}
    X = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for t in d:
            X[i, vocab[t]] += 1
    idf = np.array([math.log((1 + len(docs)) / (1 + df[t])) + 1 for t in sorted(df)])
    X = X * idf
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X, vocab


def spherical_kmeans(X, k, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = None
    for _ in range(iters):
        lab = (X @ C.T).argmax(1)
        newC = C.copy()
        for c in range(k):
            m = lab == c
            if m.any():
                v = X[m].mean(0)
                newC[c] = v / (np.linalg.norm(v) + 1e-12)
        if np.allclose(newC, C, atol=1e-9):
            C = newC
            break
        C = newC
    return lab, C


def table_kmeans(rows):
    texts, w, docs = [], [], []
    for r in rows:
        for s in r["segs"]:
            t = s["caption"].strip().lower()
            texts.append(t)
            w.append(s["end_s"] - s["start_s"])
            ws = re.findall(r"[a-z]+", t)
            docs.append(ws + [f"{a}_{b}" for a, b in zip(ws, ws[1:])])
    w = np.array(w)
    X, vocab = tfidf(docs)
    inv = {i: t for t, i in vocab.items()}
    print(f"\ncaption tf-idf matrix: {X.shape}  (注: 未去停用词, 质心被 'the/on' 主导)")
    for k in (4, 5, 8, 12):
        lab, C = spherical_kmeans(X, k)
        intra = (X * C[lab]).sum(1).mean()
        print(f"\n--- k={k}  mean intra-cluster cosine={intra:.3f} ---")
        for c in range(k):
            idx = np.where(lab == c)[0]
            verbs = collections.Counter(texts[i].split()[0] for i in idx).most_common(3)
            top = [inv[j] for j in C[c].argsort()[::-1][:4]]
            print(f"  c{c}: n={len(idx):3d} segs {len(idx)/len(texts)*100:4.1f}% "
                  f"dur {w[idx].sum()/w.sum()*100:4.1f}%  verbs={verbs}  terms={top}")
    print("\n结论: k=4/5 退化为 pick/place/move/handover 的'动词族', "
          "k=8/12 开始按左右手与物体切 -> 文本 K-means 给的是语义类, 不是集内时序相位。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="flora_samples35/samples35")
    args = ap.parse_args()
    if not os.path.isdir(args.root):
        ap.error(f"找不到 {args.root}; 请在 lite_agent 根目录运行, 或用 --root 指定")
    rows = load(args.root)
    print(f"tasks found: {len(rows)}\n")
    table_per_task(rows)
    table_corpus(rows)
    table_kmeans(rows)


if __name__ == "__main__":
    main()
