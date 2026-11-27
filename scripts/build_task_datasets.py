#!/usr/bin/env python3
"""把每个任务的 demo.page.npz 转成前端数据集，观感与单任务页对齐。

  左栏 = 拟合集（该任务 VAE 的参考集），按阶段着色（阶段 = 标注 caption 的开头动词）
  右栏 = 留出集（25% 的集，检测器没见过）当测试轨迹：白点 + 橙星（OOD 触发帧）
  OOD 面板：逐帧共形 p 随数据给出，α 滑块 / 触发清单可用
  （留出集全是成功集，所以这里的"检出"其实是误报，面板如实显示为"成功 N 局中误报"）

输入：--pages /tmp/pages（<task>.npz: pc3/ep/t/p/pc_var/is_cal）
      --labels /tmp/labels（manifest.csv + labels/epNNNNNN.json）
      --videos docs/assets/videos/flora
输出：docs/assets/latents/<task>/{lof-d32.js, datasets.json}
"""
import argparse
import base64
import csv
import json
import os
import re

import numpy as np

WIN, NEED, ALPHA = 5, 3, 0.12


def b64(a, dt):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dt).tobytes()).decode()


def verb_of(caption):
    w = re.split(r"\s+", caption.strip().lower())[:2]
    return (" ".join(w).rstrip(",.")) or "other"


def load_labels(labels_dir):
    seg = {}
    d0 = os.path.join(labels_dir, "labels")
    for f in os.listdir(d0):
        if not (f.startswith("ep") and f.endswith(".json")):
            continue
        ep = int(f[2:-5])
        d = json.load(open(os.path.join(d0, f)))
        seg[ep] = [(s["start_s"], s["end_s"], verb_of(s["caption"])) for s in d["segments"]]
    return seg


def stage_of(segments, t_sec):
    for a, b, v in segments:
        if a <= t_sec < b:
            return v
    return "other"


def fires(P, EP, T, a=ALPHA):
    out = {}
    for ep in sorted(set(EP.tolist())):
        m = EP == ep
        o = np.where(m)[0]
        o = o[np.argsort(T[o])]
        p = P[o]
        f = -1
        for i in range(len(p) - WIN + 1):
            if int((p[i:i + WIN] < a).sum()) >= NEED:
                f = i
                break
        out[str(int(ep))] = (dict(idx=int(f), t=round(float(T[o][f]) / 25.0, 2),
                                  pmin=round(float(p[f:f + WIN].min()), 3))
                             if f >= 0 else dict(idx=-1, t=None, pmin=None))
    return out


def slug(t):
    return re.sub(r"[^A-Za-z0-9]+", "_", t.strip().lower()).strip("_")[:48]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="/tmp/pages")
    ap.add_argument("--labels", default="/tmp/labels")
    ap.add_argument("--videos", default="docs/assets/videos/flora")
    ap.add_argument("--out", default="docs/assets/latents")
    a = ap.parse_args()

    tasks_eps = {}
    for r in csv.DictReader(open(os.path.join(a.labels, "manifest.csv"))):
        tasks_eps.setdefault(r["task"], []).append(int(r["episode_id"].replace("ep", "")))
    segs = load_labels(a.labels)

    for fname in sorted(os.listdir(a.pages)):
        if not fname.endswith(".npz"):
            continue
        key = fname[:-4]
        task = next((t for t in tasks_eps if slug(t) == key), key)
        z = np.load(os.path.join(a.pages, fname))
        pc3, ep, t_, p = z["pc3"], z["ep"], z["t"], z["p"]
        is_cal = z["is_cal"].astype(bool)

        stages = np.array([stage_of(segs[int(e)], float(tt) / 25.0) if int(e) in segs else "other"
                           for e, tt in zip(ep, t_)])
        names = sorted(set(stages.tolist()))
        sid = {n: i for i, n in enumerate(names)}
        cls = np.array([sid[s] for s in stages], np.uint8)

        # 左栏 = **全部**演示集（100 集）；75/25 只是训练 VAE 时的验证划分，不作为可视化轴
        ids_fit = np.arange(len(pc3))[np.lexsort((t_, ep))]
        eps_fit = np.unique(ep[ids_fit])

        # 右栏 = 该任务的**真实 VLA rollout**（若已编码：<pages>/<task>.eval.npz）
        evf = os.path.join(a.pages, f"{key}.eval.npz")
        has_ev = os.path.exists(evf)
        if has_ev:
            zz = np.load(evf)
            ids_cal = np.arange(len(zz["pc3"]))[np.lexsort((zz["t"], zz["ep"]))]
            eps_cal = sorted(set(zz["ep"].tolist()))
        else:
            ids_cal, eps_cal = np.array([], int), []

        offs = [0]
        for e in eps_fit:
            offs.append(offs[-1] + int((ep[ids_fit] == e).sum()))

        vdir = os.path.join(a.videos, key)
        vids = set(x[:-4] for x in os.listdir(vdir) if x.endswith(".mp4")) if os.path.isdir(vdir) else set()

        offs_ev, rollouts = [0], []
        for e in eps_cal:
            n = int((zz["ep"] == e).sum())
            offs_ev.append(offs_ev[-1] + n)
            k = int(np.where(zz["ep"] == e)[0][0])
            ok = bool(zz["ok"][k]) if "ok" in zz.files else False
            name = f"episode_{int(e):07d}_cam_head_{'success' if ok else 'fail'}"
            rollouts.append(dict(layout=int(e), n=n, ok=ok,
                                 official=(float(zz["score"][k]) if "score" in zz.files else None),
                                 score=None, cfg="VLA rollout",
                                 group=("成功" if ok else "失败"),
                                 file=(name if name in vids else "")))

        loc = fires(zz["p"][ids_cal], zz["ep"][ids_cal], zz["t"][ids_cal]) if has_ev else {}
        ev_fire = {}
        for i, e in enumerate(eps_cal):
            rec = dict(loc[str(int(e))])
            if rec["idx"] >= 0:
                rec["idx"] = offs_ev[i] + rec["idx"]
            ev_fire[str(i)] = rec
        tp = sum(1 for v in ev_fire.values() if v["idx"] >= 0)

        # 集级误报率：留出集全是成功集，逐集"有没有触发"就是误报
        def ep_fire_rate(alpha):
            hit = 0
            if not has_ev:
                return 0, 0
            for e in eps_cal:
                q = zz["p"][zz["ep"] == e]
                tt = zz["t"][zz["ep"] == e]
                q = q[np.argsort(tt)]
                h = any(int((q[i:i + WIN] < alpha).sum()) >= NEED
                        for i in range(len(q) - WIN + 1))
                hit += bool(h)
            return hit, len(eps_cal)
        ep_fpr = ({str(al): ep_fire_rate(al) for al in (0.01, 0.05, 0.12)}
                  if has_ev else {})

        ds = dict(
            meta=dict(
                title=f"「{task}」· 该任务独立 VAE（32 维潜变量）→ demo 拟合 PCA 前 3 维",
                pageTitle="Flora · 潜空间浏览器",
                paneATitle=(f"<b>拟合集（参考）</b> · {len(ids_fit):,} 帧 / {len(eps_fit)} 集 · 按阶段着色"),
                paneBTitle=(f"<b>VLA 真实 rollout · {len(eps_cal)} 局</b> · "
                            f"{len(ids_cal):,} 帧 · 白点=轨迹 "
                            "<span style='color:#ff7a18'>橙星=OOD 触发帧</span>"
                            if has_ev else "<b>VLA 真实 rollout</b> · 暂无数据"),
                n=int(len(ids_fit)), n_ep=int(len(eps_fit)),
                ep_names=[f"ep{int(e):06d}" for e in eps_fit],
                pc_var=[round(float(v) * 100, 2) for v in z["pc_var"]],
                center=[0.0, 0.0], t_max=float(t_.max() / 25.0),
                no_cls=False, cmode0="cls",
                cls_counts={int(sid[n]): int((stages == n).sum()) for n in names},
                cls_names=names,
                ev_n=int(len(zz["pc3"])) if has_ev else 0, ev_rollouts=rollouts,
                evNote=("" if has_ev else "该任务暂无 VLA 评测 rollout（左栏为全部演示）"), ev2_n=0, ev2_rollouts=[],
                ev_fire={"0": ev_fire}, batch0Label=f"留出集（{len(eps_cal)} 集）",
                latent=32, alpha0=ALPHA, onlyCurOODDefault=True, autoplayFirstFire=True,
                epFpr=({k: f"{(v[0]/max(1, v[1])*100):.0f}%" for k, v in ep_fpr.items()}
                       if ep_fpr else {}),
                pSource="共形 p（局部密度归一化 kNN）",
                videoDir=f"videos/flora/{key}", ev_videoDir=f"videos/flora/{key}",
                videoEpisodes=[i for i, n in
                               enumerate([f"ep{int(e):06d}" for e in eps_fit]) if n in vids],
                onePane=False,
            ),
            pc=b64(pc3[ids_fit], np.float32),
            cls=b64(cls[ids_fit], np.uint8),
            ep=b64(np.array([list(eps_fit).index(x) for x in ep[ids_fit]], np.int32), np.int32),
            off=b64(np.asarray(offs, np.int32), np.int32),
            t=b64((t_[ids_fit] / 25.0).astype(np.float32), np.float32),
            score=b64(p[ids_fit], np.float32),
            p=b64(p[ids_fit], np.float32),
            ang=b64(np.arctan2(pc3[ids_fit][:, 1], pc3[ids_fit][:, 0]).astype(np.float32), np.float32),
            ev=(dict(n=int(len(zz["pc3"])), z=b64(zz["pc3"][ids_cal], np.float32),
                     t=b64((zz["t"][ids_cal] / 25.0).astype(np.float32), np.float32),
                     ep=b64(np.array([eps_cal.index(x) for x in zz["ep"][ids_cal]], np.int32), np.int32),
                     off=b64(np.asarray(offs_ev, np.int32), np.int32),
                     ok=b64(np.array([1 if rollouts[eps_cal.index(e)]["ok"] else 0
                                      for e in zz["ep"][ids_cal]], np.uint8), np.uint8),
                     cfg=b64(np.zeros(len(ids_cal), np.uint8), np.uint8),
                     p=b64(zz["p"][ids_cal], np.float32))
                 if has_ev else dict(n=0)),
            ev2=dict(n=0),
        )
        outdir = os.path.join(a.out, key)
        os.makedirs(outdir, exist_ok=True)
        open(os.path.join(outdir, "lof-d32.js"), "w").write(
            "window.ROUTEC_DATA = window.ROUTEC_DATA || {};\n"
            f"window.ROUTEC_DATA[\"{key}@lof-d32\"] = {json.dumps(ds, ensure_ascii=False)};\n")
        json.dump(dict(key=key, label=(task[:44] + "…") if len(task) > 46 else task,
                       videoDir=f"videos/flora/{key}", evVideoDir=f"videos/flora/{key}",
                       datasets=[dict(key="lof-d32", group="按任务（该任务独立 VAE · 32 维）",
                                      file=f"latents/{key}/lof-d32.js", latent=32,
                                      n_tr=int(len(ids_fit)), n_ev=int(len(ids_cal)),
                                      n_ep=int(len(eps_fit)), pc_var=ds["meta"]["pc_var"],
                                      note=(f"{len(eps_fit)} 集演示 · {len(eps_cal)} 局 rollout · "
                                            f"集级误报@0.12 {ep_fpr['0.12'][0]}/{ep_fpr['0.12'][1]}"
                                            if has_ev else
                                            f"{len(eps_fit)} 集演示 · 暂无 rollout"))]),
                  open(os.path.join(outdir, "datasets.json"), "w"), ensure_ascii=False, indent=1)
        extra = (f" · 集级误报 α=0.12: {ep_fpr['0.12'][0]}/{ep_fpr['0.12'][1]}" if has_ev else " · 暂无 rollout")
        print(f"  {key}: 演示 {len(ids_fit):,} 帧 / {len(eps_fit)} 集 · 阶段 {len(names)} 类"
              f"{extra} · {os.path.getsize(os.path.join(outdir, 'lof-d32.js'))/1024:.0f} KB")


if __name__ == "__main__":
    main()
