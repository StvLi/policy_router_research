#!/usr/bin/env python3
"""把交接包（docs/HANDOVER_LATENT_VIS.md 描述的 `data/routeC_latents/`）转成前端数据。

输入（交接包，按维数一套）：
    latent{N}_demo.npz   Z(N,?) PC3(N,3) EP T CAP
    latent{N}_ood.npz    Z PC3 EP T
    latent{N}_eval.npz   Z PC3 EP T SUCCESS SCORE PROGRESS
    latent{N}_vae.pt     权重（本脚本不加载 torch，只读 export_index.json 的元数据）

输出（前端可直接 <script src> 的 JS）：
    docs/assets/routeC_latents/d{N}.js
    docs/assets/routeC_latents/manifest.js

关于逐帧 `p`：
  * 若 npz 里**已经有** `S`（重建分）与 `P`（共形 p），直接用（交接方按约定补上即可，无需改代码）。
  * 否则退回**潜空间马氏距离 + 共形 p**：在 demo 的 Z 上估计均值/协方差（收缩正则），
    用"每 4 集留 1 集"做校准集，`p = (1 + #{cal >= s}) / (1 + |cal|)`。
    这一版与 VAE 重建分**不是同一个量**，页面与文档里必须标注清楚。
"""
import argparse
import base64
import glob
import json
import os
import re

import numpy as np

ALPHA_FIRE = 0.12
WIN, NEED = 5, 3
CAL_EVERY = 4          # 每 4 集留 1 集做校准（近似交接包的 75/25 划分）


def b64(a, dt):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dt).tobytes()).decode()


def fire_point(p, a=ALPHA_FIRE):
    for i in range(0, len(p) - WIN + 1):
        if int((p[i:i + WIN] < a).sum()) >= NEED:
            return i
    return -1


def progress_knn_p(Z_ref, prog_ref, ep_ref, Z_cal, prog_cal, ep_cal, Z_tgt, prog_tgt, ep_tgt,
                   k=5, band=0.10, nbins=20):
    """按**进度条件化**的 kNN 新颖度 + 共形 p。

    why：潜空间是多峰的（不同阶段各占一坨），全局高斯/全局 kNN 会把"正常但换阶段"也当成异常。
          先按集内归一化进度切带，再在同带内找最近邻，才是"这个进度上有没有像样的演示帧"。
    返回 (score_tgt, p_tgt)，p 用 cal 的分数做共形。
    """
    edges = np.linspace(0.0, 1.0, nbins + 1)
    def binof(p):
        b = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
        return b
    ref_bins = binof(prog_ref)
    groups = [np.where(ref_bins == b)[0] for b in range(nbins)]

    def stat(Q, QP, QEP):
        out = np.empty(len(Q), np.float32)
        qb = binof(QP)
        for i in range(len(Q)):
            b = qb[i]
            lo, hi = max(0, b - int(round(band * nbins))), min(nbins, b + int(round(band * nbins)) + 1)
            idx = np.concatenate([groups[j] for j in range(lo, hi) if len(groups[j])])
            idx = idx[ep_ref[idx] != QEP[i]]            # 留一集，避免同集自比
            if len(idx) < k + 1:
                idx = np.where(ep_ref != QEP[i])[0]
            if len(idx) == 0:
                out[i] = 1e9; continue
            diff = Z_ref[idx] - Q[i]
            dd = (diff * diff).sum(1)
            kk = min(k, len(dd)) - 1
            out[i] = np.sqrt(np.partition(dd, kk)[kk])
        return out

    sc = stat(Z_cal, prog_cal, ep_cal)
    st = stat(Z_tgt, prog_tgt, ep_tgt)
    p = (1.0 + (sc[None, :] >= st[:, None]).sum(1)) / (len(sc) + 1.0)
    return st, p


def progress_of(T, EP):
    out = np.zeros(len(T), np.float32)
    for ep in np.unique(EP):
        m = EP == ep
        out[m] = (T[m] - T[m].min()) / max(1, T[m].max() - T[m].min())
    return out


def build(files, info):
    demo = np.load(files["demo"], allow_pickle=True)
    ood = np.load(files["ood"], allow_pickle=True)
    ev = np.load(files["eval"], allow_pickle=True)
    N = int(info["latent"])

    Ztr, PCtr, EPd, Td = demo["Z"], demo["PC3"], demo["EP"], demo["T"]
    CAP = demo["CAP"] if "CAP" in demo.files else np.array(["unknown"] * len(EPd))
    Zev, PCev, EPe, Te = ev["Z"], ev["PC3"], ev["EP"], ev["T"]
    Zoo, PCoo = ood["Z"], ood["PC3"]
    SCORE = ev["SCORE"] if "SCORE" in ev.files else np.zeros(len(EPe))
    SUCC = ev["SUCCESS"] if "SUCCESS" in ev.files else np.zeros(len(EPe), bool)

    # ---- 逐帧 p：优先用交接包给的 S/P，否则潜空间马氏距离 ----
    p_src = "共形 p（进度条件 kNN）"
    if "S" in demo.files and "P" in demo.files:
        S_tr, P_tr = demo["S"], demo["P"]
        S_ev, P_ev = ev["S"], ev["P"]
    else:
        cal_mask = (EPd % CAL_EVERY) == 0
        Pd = progress_of(Td, EPd)
        Pe = progress_of(Te, EPe)
        S_tr, P_tr = progress_knn_p(Ztr[~cal_mask], Pd[~cal_mask], EPd[~cal_mask],
                                    Ztr[cal_mask], Pd[cal_mask], EPd[cal_mask],
                                    Ztr, Pd, EPd)
        S_ev, P_ev = progress_knn_p(Ztr[~cal_mask], Pd[~cal_mask], EPd[~cal_mask],
                                    Ztr[cal_mask], Pd[cal_mask], EPd[cal_mask],
                                    Zev, Pe, EPe)

    # ---- 训练集排序 ----
    o = np.lexsort((Td, EPd))
    PCtr, EPd, Td, CAP, S_tr, P_tr = PCtr[o], EPd[o], Td[o], CAP[o], S_tr[o], P_tr[o]
    eps_tr = np.unique(EPd)
    offs = [0]
    for e in eps_tr:
        offs.append(offs[-1] + int((EPd == e).sum()))

    caps = sorted(set(CAP.tolist()))
    cap_idx = {c: i for i, c in enumerate(caps)}

    # ---- 评测（按 rollout, t 排序）----
    eps_ev = sorted(set(EPe.tolist()))
    o2 = np.lexsort((Te, np.array([eps_ev.index(x) for x in EPe])))
    PCev, EPe, Te, SCORE, SUCC, P_ev = PCev[o2], EPe[o2], Te[o2], SCORE[o2], SUCC[o2], P_ev[o2]
    rollouts, fires, offs_e = [], {}, [0]
    for r, ep in enumerate(eps_ev):
        a0 = offs_e[-1]
        n = int((EPe == ep).sum())
        a1 = a0 + n
        offs_e.append(a1)
        f = fire_point(P_ev[a0:a1])
        fires[str(r)] = (dict(idx=int(f), t=round(float(Te[a0 + f]) / 25.0, 2),
                              pmin=round(float(P_ev[a0 + f:a0 + f + WIN].min()), 3))
                         if f >= 0 else dict(idx=-1, t=None, pmin=None))
        ok = bool(SUCC[a0])
        sc = float(SCORE[a0])
        rollouts.append(dict(layout=int(ep), n=n, ok=ok, official=sc,
                             score=round(float(np.mean(SCORE[a0:a1])), 3), cfg="路线 C",
                             group="",
                             file=(f"episode_{int(ep):07d}_cam_{{cam}}_"
                                   f"{'success' if ok else 'fail'}")))

    n_fail = sum(1 for r in rollouts if not r["ok"])
    tp = sum(1 for i, r in enumerate(rollouts) if not r["ok"] and fires[str(i)]["idx"] >= 0)
    pcvar = [round(float(v) * 100, 2) for v in info.get("pca_explained", [0, 0, 0])]

    ds = dict(
        meta=dict(
            title=(f"π0 中间表征 → VAE（{N} 维潜变量）→ demo 拟合 PCA 前 3 维"
                   f"（3D 代表 {sum(pcvar):.0f}% 的结构）"),
            pageTitle="路线 C · 隐空间浏览器",
            paneATitle=(f"<b>演示（ID）</b> · {len(PCtr):,} 帧 / {len(eps_tr)} 集 · 按阶段着色 · "
                        f"<span style='color:#e74c3c'>红=另一个任务（OOD）</span>"),
            paneBTitle=(f"<b>真实评测 · {len(rollouts)} 局</b> · {len(PCev):,} 帧 · 同一 PCA 基 · "
                        "<span style='color:#ff7a18'>橙星=OOD 触发帧</span>"),
            n=int(len(PCtr)), n_ep=int(len(eps_tr)),
            ep_names=[f"ep{int(e)}" for e in eps_tr],
            pc_var=pcvar, center=[0.0, 0.0], t_max=float(Td.max() / 25.0),
            no_cls=False, cmode0="cls",
            cls_counts={i: int((CAP == c).sum()) for c, i in cap_idx.items()},
            cls_names=caps,
            ev_n=int(len(PCev)), ev_rollouts=rollouts, ev2_n=0, ev2_rollouts=[],
            ev_fire={"0": fires}, batch0Label=f"真实评测（{len(rollouts)} 局）",
            latent=N, alpha0=ALPHA_FIRE, onlyCurOODDefault=True, autoplayFirstFire=True,
            pSource=p_src,
            stats=dict(n_fail=n_fail, tp=tp),
        ),
        pc=b64(PCtr, np.float32),
        cls=b64(np.array([cap_idx[c] for c in CAP], np.uint8), np.uint8),
        ep=b64(np.array([list(eps_tr).index(x) for x in EPd], np.int32), np.int32),
        off=b64(np.asarray(offs, np.int32), np.int32),
        t=b64((Td / 25.0).astype(np.float32), np.float32),
        score=b64(np.asarray(S_tr, np.float32), np.float32),
        p=b64(np.asarray(P_tr, np.float32), np.float32),
        ang=b64(np.arctan2(PCtr[:, 1], PCtr[:, 0]).astype(np.float32), np.float32),
        ood=dict(n=int(len(PCoo)), z=b64(PCoo, np.float32), label="另一任务（OOD）"),
        ev=dict(n=int(len(PCev)), z=b64(PCev, np.float32),
                t=b64((Te / 25.0).astype(np.float32), np.float32),
                ep=b64(np.array([eps_ev.index(x) for x in EPe], np.int32), np.int32),
                off=b64(np.asarray(offs_e, np.int32), np.int32),
                ok=b64(np.array([1 if SUCC[a] else 0 for a in range(len(SUCC))], np.uint8), np.uint8),
                cfg=b64(np.zeros(len(PCev), np.uint8), np.uint8),
                p=b64(np.asarray(P_ev, np.float32), np.float32)),
        ev2=dict(n=0),
    )
    note = (f"{len(eps_tr)} 集 demo · α={ALPHA_FIRE} 检出 {tp}/{n_fail} · 3D≈{sum(pcvar):.0f}%"
            + (" · p=进度条件kNN" if "latent" in p_src else " · p=真实重建分"))
    man = dict(key=f"d{N}", latent=N, n_tr=int(len(PCtr)), n_ev=int(len(PCev)),
               n_ood=int(len(PCoo)), n_ep=int(len(eps_tr)), pc_var=pcvar,
               n_fail=n_fail, tp=tp, note=note, p_src=p_src)
    return ds, man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/routeC_latents")
    ap.add_argument("--out", default="docs/assets/routeC_latents")
    ap.add_argument("--default", default="")
    a = ap.parse_args()

    idx = json.load(open(os.path.join(a.src, "export_index.json")))
    os.makedirs(a.out, exist_ok=True)
    entries = []
    for info in idx:
        N = int(info["latent"])
        files = {k: os.path.join(a.src, info["files"][k]) for k in ("demo", "ood", "eval")}
        if not all(os.path.exists(v) for v in files.values()):
            print(f"跳过 d{N}：文件缺失"); continue
        ds, man = build(files, info)
        js = ("window.ROUTEC_DATA = window.ROUTEC_DATA || {};\n"
              f"window.ROUTEC_DATA[\"{man['key']}\"] = {json.dumps(ds, ensure_ascii=False)};\n")
        p = os.path.join(a.out, f"d{N}.js")
        open(p, "w").write(js)
        entries.append(man)
        print(f"d{N}: {man['n_tr']} demo 帧/{man['n_ep']} 集 · {man['n_ood']} OOD 帧 · "
              f"{man['n_ev']} 评测帧 · PC1-3 {man['pc_var']} · α=0.12 检出 {man['tp']}/{man['n_fail']} "
              f"· p 来源 {man['p_src']} · {os.path.getsize(p)/1024:.0f} KB")

    entries.sort(key=lambda e: -e["latent"])
    default = a.default or f"d{entries[0]['latent']}"
    man = dict(default=default, dims=entries,
               note="每个维数一套 demo 拟合的 PCA 基；3D 只代表 |pc_var| 之和比例的结构")
    open(os.path.join(a.out, "manifest.js"), "w").write(
        "window.ROUTEC_MANIFEST = " + json.dumps(man, ensure_ascii=False) + ";\n")
    print(f"写出 {a.out}/manifest.js（{[e['key'] for e in entries]}，默认 {default}）")


if __name__ == "__main__":
    main()
