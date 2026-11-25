#!/usr/bin/env python3
"""生成可拖动交互的 3D 潜空间浏览器（单文件 HTML，零依赖，数据内嵌）

输入：routeA_task_latent.npz（scripts/routeA_task_latent.py 在 6c 生成）
输出：docs/assets/latent_explorer.html（双击即可打开，不需要服务器、不需要联网）

内容：把 16 维潜变量 PCA 到 3 维，画成可旋转/缩放的点云；
     可按子任务类别 / 相位角 / 时间 / 不典型度着色；
     可选中单条 episode 看它的执行轨迹（按时间渐变）。
"""
import argparse
import base64
import json
import os

import numpy as np


def pca3(X):
    Xc = X - X.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    return Xc @ Vt[:3].T, ev[:3]


def b64(arr, dtype):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", default="/tmp/routeA_latents3.npz")
    ap.add_argument("--spec", default="/tmp/pilot_spec_local.json")
    ap.add_argument("--videos", default="videos/head")
    ap.add_argument("--eval-videos", default="videos/eval")
    ap.add_argument("--eval25-videos", default="videos/eval25")
    ap.add_argument("--routeC-layer", default="/tmp/routeC_layer.npz")
    ap.add_argument("--out", default="docs/assets/latent_explorer.html")
    ap.add_argument("--title", default="收纳耳机与笔记本电脑 · 整任务 VAE 潜空间（PCA 3 维）")
    a = ap.parse_args()

    z = np.load(a.latent, allow_pickle=True)
    keys = set(z.files)
    if "tr_z3" in keys:                       # 新格式：训练+评测同一坐标系
        Z3, Y, EP, TT0, SCORE = z["tr_z3"], z["tr_y"], z["tr_ep"], z["tr_t"], z["tr_score"]
        pcvar = z["pc_var"]
        caps = [c.decode() if isinstance(c, bytes) else c for c in z["caps"]]
        cap_labels = z["cap_labels"]
        FIDX = (TT0 * 25).round().astype(np.int32)
        approx = bool(np.asarray(z["approx"]).reshape(-1)[0]) if "approx" in keys else False
    else:                                      # 旧格式（仅训练）
        MU = z["mu"]; Y, EP, FIDX, SCORE = z["y"], z["ep"], z["fidx"], z["score"]
        caps = [c.decode() if isinstance(c, bytes) else c for c in z["caps"]]
        cap_labels = z["cap_labels"]
        Z3, pcvar = pca3(MU)
        approx = False
    N = len(Z3)

    order = np.lexsort((FIDX, EP))
    Z3, Y, EP, FIDX, SCORE = Z3[order], Y[order], EP[order], FIDX[order], SCORE[order]

    eps = np.unique(EP)
    ep_remap = {int(e): i for i, e in enumerate(eps)}
    EPI = np.array([ep_remap[int(e)] for e in EP], np.int32)
    OFF = np.zeros(len(eps) + 1, np.int32)
    for i, c in enumerate(np.bincount(EPI, minlength=len(eps))):
        OFF[i + 1] = OFF[i] + c

    center = np.median(Z3[:, :2], axis=0)
    ang = np.arctan2(Z3[:, 1] - center[1], Z3[:, 0] - center[0])

    # ---- 每集分段标注 ----
    segs_by_ep = {}
    if os.path.exists(a.spec):
        spec = json.load(open(a.spec))
        cap2cls = {c: int(l) for c, l in zip(caps, cap_labels)}
        for i, e in enumerate(eps):
            out = []
            for s in spec["segs"].get(str(int(e))) or []:
                cap = s["caption"]
                out.append([round(float(s["start_s"]), 2), round(float(s["end_s"]), 2),
                            cap, cap2cls.get(cap.lower(), -1)])
            segs_by_ep[str(i)] = out
        print(f"嵌入分段标注：{len(segs_by_ep)} 集")

    # ---- 评测轨迹（同一坐标基）----
    ev = dict(n=0)
    if "ev_z3" in keys and len(z["ev_z3"]):
        EZ, ET, EEPI = z["ev_z3"], z["ev_t"], z["ev_ep"]
        EOK, ECFG = z["ev_ok"].astype(np.uint8), z["ev_cfg"]
        ELAY = z["ev_layout"]
        order_e = np.lexsort((ET, EEPI))
        EZ, ET, EEPI, EOK, ECFG, ELAY = (EZ[order_e], ET[order_e], EEPI[order_e],
                                         EOK[order_e], ECFG[order_e], ELAY[order_e])
        remap = {int(v): i for i, v in enumerate(np.unique(EEPI))}
        EEP = np.array([remap[int(v)] for v in EEPI], np.int32)
        OFF_EV = np.zeros(int(EEP.max()) + 2, np.int32)
        for i, c in enumerate(np.bincount(EEP, minlength=int(EEP.max()) + 1)):
            OFF_EV[i + 1] = OFF_EV[i] + c
        rollouts = []
        for r in range(int(EEP.max()) + 1):
            m = EEP == r
            if not m.any():
                continue
            fname = f"{'no_vlm' if ECFG[m][0] == 0 else 'with_vlm'}_{ELAY[m][0]}_{'success' if EOK[m][0] else 'fail'}"
            rollouts.append(dict(cfg="no_vlm" if ECFG[m][0] == 0 else "with_vlm",
                                 ok=bool(EOK[m][0]), layout=str(ELAY[m][0]),
                                 n=int(m.sum()), file=fname))
        ev = dict(n=int(len(EZ)), z=b64(EZ, np.float32), t=b64(ET, np.float32),
                  ep=b64(EEP, np.int32), off=b64(OFF_EV, np.int32),
                  ok=b64(EOK, np.uint8), cfg=b64(ECFG, np.uint8),
                  rollouts=rollouts, videoDir=a.eval_videos)
        print(f"评测轨迹：{len(EZ)} 帧 / {len(rollouts)} 个 rollout "
              f"(成功 {sum(1 for x in rollouts if x['ok'])})")

    # ---- 第二批：data/eval25（真实观测）----
    ev2 = dict(n=0)
    if "ev2_z3" in keys and len(z["ev2_z3"]):
        E2Z, E2T, E2EP = z["ev2_z3"], z["ev2_t"], z["ev2_ep"]
        E2OK, E2LAY, E2SC = z["ev2_ok"].astype(np.uint8), z["ev2_layout"], z["ev2_score"]
        o2 = np.lexsort((E2T, E2EP))
        E2Z, E2T, E2EP, E2OK, E2LAY, E2SC = (E2Z[o2], E2T[o2], E2EP[o2], E2OK[o2], E2LAY[o2], E2SC[o2])
        remap2 = {int(v): i for i, v in enumerate(np.unique(E2EP))}
        E2P = np.array([remap2[int(v)] for v in E2EP], np.int32)
        OFF2 = np.zeros(int(E2P.max()) + 2, np.int32)
        for i, c in enumerate(np.bincount(E2P, minlength=int(E2P.max()) + 1)):
            OFF2[i + 1] = OFF2[i] + c
        roll2 = []
        for r in range(int(E2P.max()) + 1):
            m = E2P == r
            if not m.any():
                continue
            lay, ok = int(E2LAY[m][0]), bool(E2OK[m][0])
            roll2.append(dict(layout=lay, ok=ok, score=float(E2SC[m][0]), n=int(m.sum()),
                              file=f"episode_{lay:07d}_cam_head_{'success' if ok else 'fail'}"))
        ev2 = dict(n=int(len(E2Z)), z=b64(E2Z, np.float32), t=b64(E2T, np.float32),
                   ep=b64(E2P, np.int32), off=b64(OFF2, np.int32),
                   ok=b64(E2OK, np.uint8), layout=b64(E2LAY.astype(np.int32), np.int32),
                   rollouts=roll2, videoDir=a.eval25_videos)
        print(f"第二批 eval25：{len(E2Z)} 帧 / {len(roll2)} 集 (成功 {sum(1 for r in roll2 if r['ok'])})")

    # ---- 路线 C：独立 PCA 坐标系（在演示集上拟合，评测集投影） ----
    rc = dict(n=0, nd=0)
    rc_meta = {}
    if a.routeC_layer and os.path.exists(a.routeC_layer):
        C = np.load(a.routeC_layer, allow_pickle=True)
        DZ, EZ = C["demo_z3"].astype(np.float64), C["ev_z3"].astype(np.float64)
        CE, CT, CS, CP = C["ev_ep"], C["ev_t"], C["ev_score"], C["ev_p"]
        LAY, FIRE = C["layouts"], C["fire"]
        allz = np.vstack([DZ, EZ])
        cc, cs = allz.mean(0), float(np.abs(allz - allz.mean(0)).max()) or 1.0
        DZn, EZn = (DZ - cc) / cs, (EZ - cc) / cs
        # 逐集按时间排序 → 局部位次（供游标/触发帧定位）
        SE, ST, SS, SP = [], [], [], []
        offs, firemap = [0], {}
        for L, f in zip(LAY.tolist(), FIRE.tolist()):
            idx = np.where(CE == L)[0]
            idx = idx[np.argsort(CT[idx])]
            SE.append(EZn[idx]); ST.append(CT[idx])
            SS.append(CS[idx]); SP.append(CP[idx])
            offs.append(offs[-1] + len(idx))
            firemap[str(int(L))] = dict(
                idx=int(f), t=(round(float(ST[-1][f]), 2) if f >= 0 else None),
                score=(round(float(SS[-1][f]), 3) if f >= 0 else None))
        rc = dict(n=int(len(EZ)), nd=int(len(DZ)),
                  demo=b64(DZn, np.float32), z=b64(np.vstack(SE), np.float32),
                  t=b64(np.concatenate(ST), np.float32),
                  ep=b64(np.repeat(LAY, [o2 - o1 for o1, o2 in zip(offs, offs[1:])]), np.int32),
                  score=b64(np.concatenate(SS), np.float32),
                  p=b64(np.concatenate(SP), np.float32),
                  off=b64(np.array(offs, np.int32), np.int32))
        rc_meta = dict(rc_layouts=[int(x) for x in LAY.tolist()], rc_fire=firemap,
                       rc_pcvar=[round(float(v) * 100, 1) for v in C["pc_var"]],
                       rc_official=[None if np.isnan(float(v)) else round(float(v), 2)
                                    for v in C["official"]])
        print(f"路线 C 图层：演示 {rc['nd']} 帧 + 评测 {rc['n']} 帧 / {len(LAY)} 局 "
              f"(触发 {sum(1 for v in firemap.values() if v['idx'] >= 0)} 局)")

    meta = dict(title=a.title, videoDir=a.videos, segs=segs_by_ep,
                n=int(N), n_ep=int(len(eps)),
                ep_names=[f"ep{int(e):06d}" for e in eps],
                pc_var=[round(float(v) * 100, 2) for v in pcvar],
                center=[float(center[0]), float(center[1])],
                t_max=float(FIDX.max() / 25.0),
                cls_counts={int(c): int((Y == c).sum()) for c in np.unique(Y)},
                ev_n=ev.get("n", 0), ev_rollouts=ev.get("rollouts", []),
                ev2_n=ev2.get("n", 0), ev2_rollouts=ev2.get("rollouts", []),
                ev_videoDir=a.eval_videos, eval25VideoDir=a.eval25_videos,
                approx=approx, **rc_meta)

    DATA = dict(pc=b64(Z3, np.float32), cls=b64(Y, np.uint8), ep=b64(EPI, np.int32),
                off=b64(OFF, np.int32), t=b64(FIDX.astype(np.float32) / 25.0, np.float32),
                score=b64(SCORE, np.float32), ang=b64(ang, np.float32), ev=ev, ev2=ev2, rc=rc)

    html = load_template().replace("__META__", json.dumps(meta, ensure_ascii=False)) \
                          .replace("__DATA__", json.dumps(DATA))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(html)
    print(f"写出 {a.out}  ({os.path.getsize(a.out)/1024:.0f} KB)")


TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "explorer_template.html")


def load_template():
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    main()
