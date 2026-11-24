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
    ap.add_argument("--latent", default="/tmp/routeA_latents.npz")
    ap.add_argument("--spec", default="/tmp/pilot_spec_local.json")
    ap.add_argument("--videos", default="videos/head")
    ap.add_argument("--eval-videos", default="videos/eval")
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

    meta = dict(title=a.title, videoDir=a.videos, segs=segs_by_ep,
                n=int(N), n_ep=int(len(eps)),
                ep_names=[f"ep{int(e):06d}" for e in eps],
                pc_var=[round(float(v) * 100, 2) for v in pcvar],
                center=[float(center[0]), float(center[1])],
                t_max=float(FIDX.max() / 25.0),
                cls_counts={int(c): int((Y == c).sum()) for c in np.unique(Y)},
                ev_n=ev.get("n", 0), ev_rollouts=ev.get("rollouts", []),
                ev_videoDir=a.eval_videos, approx=approx)

    DATA = dict(pc=b64(Z3, np.float32), cls=b64(Y, np.uint8), ep=b64(EPI, np.int32),
                off=b64(OFF, np.int32), t=b64(FIDX.astype(np.float32) / 25.0, np.float32),
                score=b64(SCORE, np.float32), ang=b64(ang, np.float32), ev=ev)

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
