#!/usr/bin/env python3
"""生成「按任务浏览」页面：每个任务一套独立 VAE 的潜空间（32 维）+ 演示视频。

输入（由流水线产出）：
  --pages   工作站导出的 <task>/demo.page.npz（pc3 / ep / t / p / pc_var / is_cal）
  --manifest 交接包的 manifest.csv（episode → task 提示词）
  --videos  本地已转码的演示视频目录（<slug>/epXXXXXX.mp4）

输出：
  docs/assets/flora_tasks/<slug>.js   每个任务一套数据
  docs/assets/flora_tasks/manifest.js 分组清单（一档一个任务）
  docs/assets/flora_task_explorer.html 页面外壳（单栏 + 任务下拉）

用法：
  python3 scripts/build_multitask_page.py --pages /tmp/pages --videos docs/assets/videos/flora
"""
import argparse
import base64
import csv
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def b64(a, dt):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dt).tobytes()).decode()


def slugify(task):
    s = re.sub(r"[^A-Za-z0-9]+", "_", task.strip().lower()).strip("_")
    return s[:48]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="/tmp/pages")
    ap.add_argument("--manifest", default="/tmp/manifest.csv")
    ap.add_argument("--videos", default="docs/assets/videos/flora")
    ap.add_argument("--out", default="docs/assets/latents")
    ap.add_argument("--html", default="docs/assets/flora_task_explorer.html")
    a = ap.parse_args()

    tasks, eps_of = {}, {}
    for r in csv.DictReader(open(a.manifest)):
        t = r["task"]
        tasks.setdefault(t, []).append(int(r["episode_id"].replace("ep", "")))
    for t in tasks:
        tasks[t].sort()

    os.makedirs(a.out, exist_ok=True)
    entries = []
    for f in sorted(os.listdir(a.pages)):
        if not f.endswith(".npz"):
            continue
        key = f[:-4]
        task = next((t for t in tasks if slugify(t) == key), key)
        z = np.load(os.path.join(a.pages, f))
        pc3, ep, t_, p = z["pc3"], z["ep"], z["t"], z["p"]
        order = np.lexsort((t_, ep))
        pc3, ep, t_, p = pc3[order], ep[order], t_[order], p[order]

        vdir = os.path.join(a.videos, key)
        vids = sorted(x[:-4] for x in os.listdir(vdir) if x.endswith(".mp4")) if os.path.isdir(vdir) else []
        eps = np.unique(ep)
        ep_names = [f"ep{int(e):06d}" for e in eps]
        video_idx = [ep_names.index(v) for v in vids if v in ep_names]
        offs = [0]
        for e in eps:
            offs.append(offs[-1] + int((ep == e).sum()))

        ds = dict(
            meta=dict(
                title=f"「{task}」 · 该任务自己的 VAE（32 维潜变量）→ demo 拟合 PCA 前 3 维",
                pageTitle="Flora · 按任务浏览",
                selectorLabel="任务",
                paneATitle=(f"<b>{task}</b> · {len(pc3):,} 帧 / {len(eps)} 集 · "
                            f"按时间着色 · 视频 {len(vids)} 集"),
                onePane=True,
                videoDir=f"videos/flora/{key}",
                videoEpisodes=video_idx,
                n=int(len(pc3)), n_ep=int(len(eps)),
                ep_names=ep_names,
                pc_var=[round(float(v) * 100, 2) for v in z["pc_var"]],
                center=[0.0, 0.0], t_max=float(t_.max() / 25.0),
                cls_counts={}, no_cls=True, cmode0="t",
                ev_n=0, ev_rollouts=[], ev2_n=0, ev2_rollouts=[],
                latent=32,
                note=f"共形 p（LOF，局部密度归一化）· 校准集 {int(z['is_cal'].sum())} 帧",
            ),
            pc=b64(pc3, np.float32),
            cls=b64(np.zeros(len(pc3), np.uint8), np.uint8),
            ep=b64(np.array([list(eps).index(x) for x in ep], np.int32), np.int32),
            off=b64(np.asarray(offs, np.int32), np.int32),
            t=b64((t_ / 25.0).astype(np.float32), np.float32),
            score=b64(p, np.float32),
            p=b64(p, np.float32),
            ang=b64(np.arctan2(pc3[:, 1], pc3[:, 0]).astype(np.float32), np.float32),
            ev=dict(n=0), ev2=dict(n=0),
        )
        open(os.path.join(a.out, f"{key}.js"), "w").write(
            "window.ROUTEC_DATA = window.ROUTEC_DATA || {};\n"
            f"window.ROUTEC_DATA[\"{key}\"] = {json.dumps(ds, ensure_ascii=False)};\n")
        entries.append(dict(key=key, latent=32, n_tr=int(len(pc3)), n_ev=0,
                            pc_var=ds["meta"]["pc_var"], n_ep=int(len(eps)),
                            label=task, note=f"{len(eps)} 集 · {len(pc3):,} 帧 · 视频 {len(vids)} 集"))
        print(f"  {key}: {len(pc3):,} 帧 / {len(eps)} 集 · 视频 {len(vids)} · "
              f"{os.path.getsize(os.path.join(a.out, key + '.js'))/1024:.0f} KB")

    entries.sort(key=lambda e: e["key"])
    man = dict(default=entries[0]["key"] if entries else "",
               dims=entries,
               groups=[dict(label="按任务（各自独立 VAE · 32 维 · LOF 共形 p）",
                            keys=[e["key"] for e in entries])],
               note="每个任务一套独立 VAE 与 PCA 基；任务之间不可比坐标，只比结构。")
    open(os.path.join(a.out, "manifest.js"), "w").write(
        "window.ROUTEC_MANIFEST = " + json.dumps(man, ensure_ascii=False) + ";\n")

    shell = dict(pageTitle="Flora · 按任务浏览",
                 title="35 个任务 · 每任务独立 VAE（32 维）→ demo PCA 前 3 维",
                 selectorLabel="任务",
                 videoDir="", ev_videoDir="", eval25VideoDir="",
                 train_cams=["head"], ev_cams=["head"],
                 onePane=True,
                 defaultDim=man["default"], dimDir=os.path.relpath(a.out, os.path.dirname(a.html)))
    tpl = open(os.path.join(HERE, "explorer_template.html"), encoding="utf-8").read()
    pre = f'<script src="{shell["dimDir"]}/manifest.js"></script>'
    html = (tpl.replace("__PRELOAD__", pre)
               .replace("__META__", json.dumps(shell, ensure_ascii=False))
               .replace("__DATA__", "{}")
               .replace("__TITLE__", shell["pageTitle"]))
    open(a.html, "w").write(html)
    print(f"\n{len(entries)} 个任务 → {a.out}/manifest.js")
    print(f"页面：{a.html}（{os.path.getsize(a.html)/1024:.0f} KB）")


if __name__ == "__main__":
    main()
