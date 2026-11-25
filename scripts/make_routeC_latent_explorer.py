#!/usr/bin/env python3
"""生成「路线 C · VAE 潜空间」交互页（与路线 A 页面同一套模板，左右两栏）。

数据口径：
  demo_latent.npz / eval_latent.npz 由 ws 侧 encode_routeC.py 产出 ——
  π0 中间表征(2048) → 路线 C 的 VAE（2048→512→256→32）的**后验均值** 32 维潜变量，
  同时给出逐帧异常分数 score 与共形 p。

  这里做的只是"可视化"这一步：在**训练集**的潜变量上拟合 PCA（3 维），
  训练集与测试轨迹都用同一个基投影 → 两栏可比。

输出：docs/assets/routeC_latent_explorer.html
用法：python3 scripts/make_routeC_latent_explorer.py
"""
import argparse
import base64
import json
import os

import numpy as np

ALPHA, WIN, NEED = 0.05, 5, 3


def b64(arr, dtype):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode()


def fire_point(p):
    """滑窗 5、窗口内 ≥3 个 p<0.05 → 返回窗口起点（采样点序号），否则 -1。"""
    p = np.asarray(p)
    for i in range(0, len(p) - WIN + 1):
        if int((p[i:i + WIN] < ALPHA).sum()) >= NEED:
            return i
    return -1


def pca3(X):
    mu = X.mean(0, keepdims=True)
    _, sv, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = (sv ** 2) / (sv ** 2).sum()
    return mu, Vt[:3].T, var


def phase3(Z3):
    """PC1-PC2 平面上的绕环角度（相位）；无环时它退化为方向角，仅作参考量。"""
    ang = np.arctan2(Z3[:, 1], Z3[:, 0])
    return ang.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="/tmp/demo_latent.npz")
    ap.add_argument("--eval", default="/tmp/eval_latent.npz")
    ap.add_argument("--official", default="/tmp/routeC_layer.npz",
                    help="提供 official 分数的 npz（layouts/official 两个键）")
    ap.add_argument("--eval25-videos", default="videos/eval25")
    ap.add_argument("--out", default="docs/assets/routeC_latent_explorer.html")
    a = ap.parse_args()

    d = np.load(a.demo)
    e = np.load(a.eval)
    Zd, EPd, Td = d["Z"].astype(np.float64), d["EP"], d["T"]
    Ze, EPe, Te = e["Z"].astype(np.float64), e["EP"], e["T"]
    Sc_d, Sc_e = d["score"], e["score"]
    P_d, P_e = d["p"], e["p"]
    print(f"潜变量：训练 {Zd.shape} / 测试 {Ze.shape}（{len(set(EPd.tolist()))} / "
          f"{len(set(EPe.tolist()))} 集）")

    mu, basis, var = pca3(Zd)
    Ztr = (Zd - mu) @ basis
    Zev = (Ze - mu) @ basis
    print(f"训练集潜空间 PCA：PC1-3 解释 {var[0]*100:.1f}/{var[1]*100:.1f}/{var[2]*100:.1f}%")

    # 逐集按时间排序
    ord_d = np.lexsort((Td, EPd))
    ord_e = np.lexsort((Te, EPe))
    Ztr, EPd, Td, Sc_d, P_d = Ztr[ord_d], EPd[ord_d], Td[ord_d], Sc_d[ord_d], P_d[ord_d]

    offs = [0]
    for ep in np.unique(EPd):
        offs.append(offs[-1] + int((EPd == ep).sum()))
    offs = np.asarray(offs, np.int32)

    # 测试轨迹：逐集切分 + 触发点
    rollouts, fire, offs_e = [], {}, [0]
    for r, ep in enumerate(sorted(set(EPe.tolist()))):
        idx = np.where(EPe == ep)[0]
        idx = idx[np.argsort(Te[idx])]
        p = P_e[idx]
        f = fire_point(p)
        fire[str(r)] = dict(idx=int(f), t=(round(float(Te[idx][f]) / 25.0, 2) if f >= 0 else None),
                            p=(round(float(p[f:f + WIN].min()), 3) if f >= 0 else None))
        rollouts.append(dict(layout=int(ep), n=int(len(idx)),
                             score=round(float(Sc_e[idx].mean()), 3),
                             cfg="路线 C"))
        offs_e.append(offs_e[-1] + int(len(idx)))
    offs_e = np.asarray(offs_e, np.int32)

    # 官方得分 → 成功/失败标签与视频文件名
    off = {}
    if a.official and os.path.exists(a.official):
        o = np.load(a.official, allow_pickle=True)
        off = {int(L): float(s) for L, s in zip(o["layouts"].tolist(), o["official"].tolist())}
    for ro in rollouts:
        sc = off.get(ro["layout"])
        ro["official"] = sc
        ro["ok"] = bool(sc is not None and sc >= 0.999)
        ro["file"] = (f"episode_{ro['layout']:07d}_cam_{{cam}}_"
                      f"{'success' if ro['ok'] else 'fail'}")

    # 测试轨迹按 (rollout, t) 排好，供右栏逐帧投影
    ord_e2 = np.lexsort((Te, np.array([sorted(set(EPe.tolist())).index(x) for x in EPe])))
    Zev, EPe_s, Te_s = Zev[ord_e2], EPe[ord_e2], Te[ord_e2]
    okf = np.array([next(r["ok"] for r in rollouts if r["layout"] == ep) for ep in EPe_s], np.uint8)
    cfgs = np.zeros(len(EPe_s), np.uint8)

    META = dict(
        title="π0 中间表征 → 路线 C 的 VAE（32 维潜变量）→ 训练集 PCA 前 3 维",
        pageTitle="路线 C · 潜空间浏览器",
        paneATitle=("<b>训练数据 · 演示</b> · "
                    f"{len(Ztr):,} 帧 / {len(offs)-1} 集 · 路线 C VAE 潜变量"),
        paneBTitle=("<b>测试轨迹 · 8 局</b> · "
                    f"{len(Zev):,} 帧 · 同一 PCA 基投影 · "
                    "<span style='color:#ff7a18'>橙星=OOD 触发帧</span>"),
        videoDir="videos/head",                       # 演示集头部视频（仓库内，100 集从 ep003100 起）
        ev_videoDir=a.eval25_videos,
        ev_cams=["head", "left_wrist", "right_wrist"], train_cams=["head"],
        eval25VideoDir=a.eval25_videos,
        n=int(len(Ztr)), n_ep=int(len(offs) - 1),
        ep_names=[f"ep{int(ep):06d}" for ep in np.unique(EPd)],
        pc_var=[round(float(v) * 100, 2) for v in var[:3]],
        center=[0.0, 0.0],
        t_max=float(Td.max() / 25.0),
        cls_counts={}, no_cls=True, cmode0="t", ev_palette=True,
        batch0Label="路线 C 测试轨迹（8 局）", autoplayFirstFire=True,
        ev_n=int(len(Zev)), ev_rollouts=rollouts, ev2_n=0, ev2_rollouts=[],
        ev_fire={"0": fire},
        approx=False,
        note="VAE 权重为 ws 上按同脚本重训（原 checkpoint 已丢失），逐帧 p 与原跑相关 0.954",
    )
    DATA = dict(
        pc=b64(Ztr, np.float32),
        cls=b64(np.zeros(len(Ztr), np.uint8), np.uint8),
        ep=b64(np.array([list(np.unique(EPd)).index(x) for x in EPd], np.int32), np.int32),
        off=b64(offs, np.int32),
        t=b64(Td.astype(np.float32) / 25.0, np.float32),
        score=b64(Sc_d, np.float32),
        p=b64(P_d, np.float32),
        ang=b64(phase3(Ztr), np.float32),
        ev=dict(n=int(len(Zev)), z=b64(Zev, np.float32),
                t=b64(Te_s.astype(np.float32) / 25.0, np.float32),
                ep=b64(np.array([sorted(set(EPe.tolist())).index(x) for x in EPe_s], np.int32), np.int32),
                off=b64(offs_e, np.int32), ok=b64(okf, np.uint8), cfg=b64(cfgs, np.uint8)),
        ev2=dict(n=0),
    )

    here = os.path.dirname(os.path.abspath(__file__))
    tpl = open(os.path.join(here, "explorer_template.html"), encoding="utf-8").read()
    html = tpl.replace("__META__", json.dumps(META, ensure_ascii=False)) \
              .replace("__DATA__", json.dumps(DATA))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(html)
    nf = sum(1 for v in fire.values() if v["idx"] >= 0)
    print(f"触发 {nf}/8 局：" + "，".join(
        f"layout {rollouts[r]['layout']}@{fire[str(r)]['t']}s" for r in range(8) if fire[str(r)]["idx"] >= 0))
    print(f"写出 {a.out}  ({os.path.getsize(a.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
