#!/usr/bin/env python3
"""在 6 卡机上按任务抽取 π0 中间表征（每任务一个 demo.npz），支持分片并行与断点续跑。

任务→episode 的映射来自交接包的 manifest.csv（`episode_id, task, duration_s, ...`）。

用法（在这台机器上，每张卡一个 shard）：
  CUDA_VISIBLE_DEVICES=4 python3 extract_all_tasks.py --shard 0 --nshard 2
  CUDA_VISIBLE_DEVICES=5 python3 extract_all_tasks.py --shard 1 --nshard 2

输出：<out>/<task_slug>/demo.npz  （E (n,2048), EP, T）
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from routeC_vae_failure import load_policy, make_extractor  # noqa: E402

FLORA = "/hdfs/users/peize/vlm_vla/flora"


def slug(task):
    s = re.sub(r"[^A-Za-z0-9]+", "_", task.strip().lower()).strip("_")
    return s[:48] or "task"


def load_tasks(manifest):
    tasks = {}
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            ep = int(row["episode_id"].replace("ep", ""))
            tasks.setdefault(row["task"], []).append(ep)
    for t in tasks:
        tasks[t].sort()
    return dict(sorted(tasks.items(), key=lambda kv: kv[1][0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/root/rtc/manifest.csv")
    ap.add_argument("--clips", default=f"{FLORA}/clips")
    ap.add_argument("--ckpt", default="/hdfs/users/peize/vlm_vla/ckpt/"
                                    "RoboDojo35FloraU_train_20261117_032827/30000")
    ap.add_argument("--out", default="/root/rtc/latent")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--episodes-per-task", type=int, default=0, help="0=全部")
    ap.add_argument("--train-config", default="pi05_base_aloha_full_sim_arx-x5_seed_0")
    ap.add_argument("--repo-id", default="arx_x5_sim")
    a = ap.parse_args()

    import cv2
    tasks = load_tasks(a.manifest)
    print(f"任务 {len(tasks)} 个，共 {sum(len(v) for v in tasks.values())} 集", flush=True)

    names = list(tasks)[a.shard::a.nshard]
    my_eps = sum(len(tasks[t][:a.episodes_per_task] if a.episodes_per_task else tasks[t])
                 for t in names)
    print(f"shard {a.shard}/{a.nshard}: {len(names)} 个任务 / {my_eps} 集", flush=True)

    a.source = "demo"
    policy = load_policy(a)
    hidden_batch = make_extractor(policy, a)
    print("模型就绪", flush=True)

    for ti, task in enumerate(names, 1):
        out = os.path.join(a.out, slug(task), "demo.npz")
        if os.path.exists(out):
            print(f"[{ti}/{len(names)}] 跳过（已有）{slug(task)}", flush=True)
            continue
        eps = tasks[task][:a.episodes_per_task] if a.episodes_per_task else tasks[task]
        t0 = time.time()
        FE, EP, T = [], [], []
        for ep in eps:
            d = os.path.join(a.clips, f"ep{ep:06d}")
            paths = [os.path.join(d, f"{c}.mp4") for c in ("main", "left_wrist", "right_wrist")]
            if not all(os.path.exists(p) for p in paths):
                print(f"  [skip] ep{ep}", flush=True)
                continue
            caps = [cv2.VideoCapture(p) for p in paths]
            n = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)
            cur, triples, ts = 0, [], []
            for t in range(0, n, a.stride):
                for c in caps:
                    for _ in range(t - cur):
                        c.grab()
                frames, ok_all = [], True
                for c in caps:
                    ok, f_ = c.read()
                    if not ok:
                        ok_all = False
                        break
                    frames.append(cv2.cvtColor(f_, cv2.COLOR_BGR2RGB))
                cur = t + 1
                if ok_all:
                    triples.append(tuple(frames))
                    ts.append(t)
            for c in caps:
                c.release()
            if not triples:
                continue
            for i in range(0, len(triples), a.batch):
                chunk = triples[i:i + a.batch]
                pad = a.batch - len(chunk)
                if pad:
                    chunk = chunk + [chunk[-1]] * pad
                h = hidden_batch(chunk)
                if pad:
                    h = h[:len(chunk) - pad]
                FE.append(h)
            EP += [ep] * len(triples)
            T += ts
        if not FE:
            print(f"[{ti}/{len(names)}] {slug(task)} 无数据", flush=True)
            continue
        E = np.concatenate(FE).astype(np.float32)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, E=E, EP=np.asarray(EP), T=np.asarray(T))
        dt = time.time() - t0
        print(f"[{ti}/{len(names)}] {slug(task)}: {len(EP)} 帧 / {len(set(EP))} 集 "
              f"{dt/60:.1f} min（{dt/max(1,len(set(EP))):.1f} s/集）→ {out}", flush=True)

    print(f"shard {a.shard} 完成", flush=True)


if __name__ == "__main__":
    main()
