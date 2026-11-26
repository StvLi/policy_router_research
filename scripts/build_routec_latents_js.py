#!/usr/bin/env python3
"""把不同潜空间维数的 VAE 潜变量打包成前端可直接加载的 JS 数据文件。

输入目录约定（ws 上 encode 之后按维数分目录）：
    <src>/d16/{demo.npz, eval25.npz, model.json}
    <src>/d32/...
每个 npz 需含键：Z (N×d, 后验均值), EP, T, score, p。

输出（默认 docs/assets/routeC_latents/）：
    d16.js, d32.js ...     window.ROUTEC_DATA["d16"] = {meta, pc, ...}
    manifest.js            window.ROUTEC_MANIFEST = {default, dims:[...]}

这样一页可以挂任意多套（每个维数一套 PCA 基），前端按需注入 <script src>，
file:// 下也能用（fetch 会被 CORS 拦，script 标签不会）。
"""
import argparse
import base64
import glob
import json
import os
import re

import numpy as np

IN_SAMPLE = {2, 8, 11, 12, 13, 17, 19, 24}      # α 调参用的 8 局
ALPHA_FIRE = 0.12
WIN, NEED = 5, 3


def b64(a, dt):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dt).tobytes()).decode()


def fire_point(p, alpha=ALPHA_FIRE):
    for i in range(0, len(p) - WIN + 1):
        if int((p[i:i + WIN] < alpha).sum()) >= NEED:
            return i
    return -1


def pca3(X):
    mu = X.mean(0, keepdims=True)
    _, sv, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = (sv ** 2) / (sv ** 2).sum()
    return mu, Vt[:3].T, var


def build_dim(src, dim, official, videos_note=""):
    dd = np.load(os.path.join(src, f"d{dim}", "demo.npz"))
    de = np.load(os.path.join(src, f"d{dim}", "eval25.npz"))
    mj = os.path.join(src, f"d{dim}", "model.json")
    model = json.load(open(mj)) if os.path.exists(mj) else {}

    Zd, EPd, Td, Sd, Pd = dd["Z"].astype(np.float64), dd["EP"], dd["T"], dd["score"], dd["p"]
    Ze, EPe, Te, Se, Pe = de["Z"].astype(np.float64), de["EP"], de["T"], de["score"], de["p"]

    mu, basis, var = pca3(Zd)
    Ztr, Zev = (Zd - mu) @ basis, (Ze - mu) @ basis

    o = np.lexsort((Td, EPd))
    Ztr, EPd, Td, Sd, Pd = Ztr[o], EPd[o], Td[o], Sd[o], Pd[o]
    offs = [0]
    for ep in np.unique(EPd):
        offs.append(offs[-1] + int((EPd == ep).sum()))

    # 评测：按 (rollout 序, t) 排序
    eps = sorted(set(EPe.tolist()))
    order = np.lexsort((Te, np.array([eps.index(x) for x in EPe])))
    Zev, EPe_s, Te_s, Se_s, Pe_s = Zev[order], EPe[order], Te[order], Se[order], Pe[order]

    rollouts, fires, offs_e = [], {}, [0]
    for r, ep in enumerate(eps):
        a0 = offs_e[-1]
        n = int((EPe_s == ep).sum())
        a1 = a0 + n
        offs_e.append(a1)
        f = fire_point(Pe_s[a0:a1])
        fires[str(r)] = (dict(idx=int(f), t=round(float(Te_s[a0 + f]) / 25.0, 2),
                              pmin=round(float(Pe_s[a0 + f:a0 + f + WIN].min()), 3))
                         if f >= 0 else dict(idx=-1, t=None, pmin=None))
        sc = official.get(int(ep))
        ok = bool(sc is not None and sc >= 0.999)
        rollouts.append(dict(layout=int(ep), n=n, ok=ok, official=sc,
                             score=round(float(Se_s[a0:a1].mean()), 3), cfg="路线 C",
                             group=("样本内" if int(ep) in IN_SAMPLE else "样本外"),
                             file=(f"episode_{int(ep):07d}_cam_{{cam}}_"
                                   f"{'success' if ok else 'fail'}")))

    n_fail = sum(1 for r in rollouts if not r["ok"])
    tp = sum(1 for i, r in enumerate(rollouts) if not r["ok"] and fires[str(i)]["idx"] >= 0)
    tp_in = sum(1 for i, r in enumerate(rollouts)
                if not r["ok"] and r["group"] == "样本内" and fires[str(i)]["idx"] >= 0)
    tp_out = sum(1 for i, r in enumerate(rollouts)
                 if not r["ok"] and r["group"] == "样本外" and fires[str(i)]["idx"] >= 0)

    ds = dict(
        meta=dict(
            title=f"π0 中间表征 → VAE（{dim} 维潜变量）→ 训练集 PCA 前 3 维",
            pageTitle="路线 C · 潜空间浏览器",
            paneATitle=(f"<b>训练数据 · 演示</b> · {len(Ztr):,} 帧 / {len(offs)-1} 集 · "
                        f"VAE 潜变量 {dim} 维"),
            paneBTitle=(f"<b>测试轨迹 · {len(rollouts)} 局</b> · {len(Zev):,} 帧 · 同一 PCA 基投影 · "
                        "<span style='color:#ff7a18'>橙星=OOD 触发帧</span>"),
            n=int(len(Ztr)), n_ep=int(len(offs) - 1),
            ep_names=[f"ep{int(e):06d}" for e in np.unique(EPd)],
            pc_var=[round(float(v) * 100, 2) for v in var[:3]],
            center=[0.0, 0.0], t_max=float(Td.max() / 25.0),
            cls_counts={}, no_cls=True, cmode0="t", ev_palette=True,
            alpha0=ALPHA_FIRE, onlyCurOODDefault=True, autoplayFirstFire=True,
            ev_n=int(len(Zev)), ev_rollouts=rollouts, ev2_n=0, ev2_rollouts=[],
            ev_fire={"0": fires}, batch0Label=f"路线 C 测试轨迹（{len(rollouts)} 局）",
            latent=int(dim), cal=model.get("cal"), model=model,
            stats=dict(n_fail=n_fail, tp=tp, tp_in=tp_in, tp_out=tp_out,
                       n_in=sum(1 for r in rollouts if r["group"] == "样本内" and not r["ok"]),
                       n_out=sum(1 for r in rollouts if r["group"] == "样本外" and not r["ok"])),
        ),
        pc=b64(Ztr, np.float32),
        cls=b64(np.zeros(len(Ztr), np.uint8), np.uint8),
        ep=b64(np.array([list(np.unique(EPd)).index(x) for x in EPd], np.int32), np.int32),
        off=b64(np.asarray(offs, np.int32), np.int32),
        t=b64(Td.astype(np.float32) / 25.0, np.float32),
        score=b64(Sd, np.float32),
        p=b64(Pd, np.float32),
        ang=b64(np.arctan2(Ztr[:, 1], Ztr[:, 0]).astype(np.float32), np.float32),
        ev=dict(n=int(len(Zev)), z=b64(Zev, np.float32),
                t=b64(Te_s.astype(np.float32) / 25.0, np.float32),
                ep=b64(np.array([eps.index(x) for x in EPe_s], np.int32), np.int32),
                off=b64(np.asarray(offs_e, np.int32), np.int32),
                ok=b64(np.array([next(r["ok"] for r in rollouts if r["layout"] == ep) for ep in EPe_s],
                                np.uint8), np.uint8),
                cfg=b64(np.zeros(len(Zev), np.uint8), np.uint8),
                p=b64(Pe_s, np.float32)),
        ev2=dict(n=0),
    )
    note = (f"{len(offs)-1} 集训练 · α={ALPHA_FIRE} 检出 {tp}/{n_fail}"
            + (f"（样本外 {tp_out}/{sum(1 for r in rollouts if r['group']=='样本外' and not r['ok'])}）"
               if any(r["group"] == "样本外" for r in rollouts) else ""))
    man = dict(key=f"d{dim}", latent=int(dim), n_tr=int(len(Ztr)), n_ev=int(len(Zev)),
               pc_var=ds["meta"]["pc_var"], n_fail=n_fail, tp=tp, tp_in=tp_in, tp_out=tp_out,
               cal=model.get("cal"), note=note)
    return ds, man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/latent")
    ap.add_argument("--out", default="docs/assets/routeC_latents")
    ap.add_argument("--result", default="data/eval25/result.json")
    ap.add_argument("--default", default="")
    a = ap.parse_args()

    official = {}
    if os.path.exists(a.result):
        det = json.load(open(a.result)).get("details", {})
        official = {int(k): float(v["score"]) for k, v in det.items() if "score" in v}

    dims = sorted(int(re.search(r"d(\d+)$", p).group(1))
                  for p in glob.glob(os.path.join(a.src, "d*")) if os.path.isdir(p))
    if not dims:
        raise SystemExit(f"{a.src} 下没有 d<N> 目录")
    os.makedirs(a.out, exist_ok=True)

    entries = []
    for dim in dims:
        ds, man = build_dim(a.src, dim, official)
        js = ("window.ROUTEC_DATA = window.ROUTEC_DATA || {};\n"
              f"window.ROUTEC_DATA[\"{man['key']}\"] = {json.dumps(ds, ensure_ascii=False)};\n")
        path = os.path.join(a.out, f"d{dim}.js")
        open(path, "w").write(js)
        entries.append(man)
        print(f"d{dim}: {man['n_tr']} 训练帧 / {man['n_ev']} 评测帧 · PC1-3 {man['pc_var']} · "
              f"{man['note']} · {os.path.getsize(path)/1024:.0f} KB")

    default = a.default or (f"d{32}" if 32 in dims else entries[0]["key"])
    man = dict(default=default, dims=entries,
               note="每个维数一套独立 PCA 基（同一维数内训练/测试共用）")
    open(os.path.join(a.out, "manifest.js"), "w").write(
        "window.ROUTEC_MANIFEST = " + json.dumps(man, ensure_ascii=False) + ";\n")
    print(f"写出 {a.out}/manifest.js（{len(entries)} 套：{[e['key'] for e in entries]}，默认 {default}）")


if __name__ == "__main__":
    main()
