#!/usr/bin/env python3
"""E1 · 对象遮挡敏感度：路线 A 与路线 C 用同一把无量纲尺比较。

尺：

    sensitivity = median |f(原帧) − f(去掉物体)|
                  ──────────────────────────────
                  median |f(帧 i) − f(帧 j)|      （自然帧间差异）

≈0：该表征几乎不编码对象；≈1：去掉物体与换一帧的影响相当。

物体定位：红色耳机 = **所有**红色连通域的并集 bbox，外扩 15%（只用最大连通域会
漏掉头梁，遮挡不彻底，两边敏感度都会被低估）。找不到就跳过该帧。

三个 stage：
  masks  任意机器：抽帧、定位、inpaint，写紧凑缓存（48×64，供路线 A）
  routeA 6c：冻结路线 A 的 VAE，比较图像项 per_img
  routeC ws：冻结 VLA，比较 transformer hidden states（头部视角 256 个图像 token）

用法：
  python3 object_occlusion_probe.py masks  --out /tmp/e1
  python3 object_occlusion_probe.py routeA --cache /tmp/e1/occl.npz --models /tmp/eval25_routeA.pt
  python3 object_occlusion_probe.py routeC --out /tmp/e1 --videos <评测目录>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


# ------------------------------------------------------------------ 定位与遮挡
def red_box(bgr, expand=0.15, cap=0.25, min_area=120):
    """红色耳机框：所有红色连通域的并集 bbox 再放大 expand 倍。"""
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (cv2.inRange(hsv, (0, 185, 50), (6, 255, 255)) |
         cv2.inRange(hsv, (174, 185, 50), (180, 255, 255)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    boxes = [stats[i][:4] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not boxes:
        return None, m
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    dx = int((x1 - x0) * expand); dy = int((y1 - y0) * expand)
    x0 = max(0, x0 - dx); y0 = max(0, y0 - dy)
    x1 = min(bgr.shape[1], x1 + dx); y1 = min(bgr.shape[0], y1 + dy)
    if (x1 - x0) * (y1 - y0) > cap * bgr.shape[0] * bgr.shape[1]:
        return None, m
    return (x0, y0, x1 - x0, y1 - y0), m


def occlude_box(bgr, box, dilate=5):
    """用 inpaint 抹掉物体框（比涂纯色自然，避免模型只检测"有个方块"）。"""
    import cv2
    m = np.zeros(bgr.shape[:2], np.uint8)
    x, y, w, h = box
    m[y:y + h, x:x + w] = 255
    m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8), iterations=2)
    return cv2.inpaint(bgr, m, 7, cv2.INPAINT_TELEA), m


def stage_masks(a):
    import cv2
    os.makedirs(a.out, exist_ok=True)
    eps = [int(x) for x in a.episodes.split(",")] if a.episodes else None
    picks = []
    for f in sorted(glob.glob(os.path.join(a.videos, "episode_*_cam_head_*.mp4"))):
        ep = int(os.path.basename(f).split("_")[1])
        if eps is None or ep in eps:
            picks.append((ep, f))
    store = dict(ep=[], t=[], orig=[], occ=[], box=[])
    for ep, f in picks:
        cap = cv2.VideoCapture(f)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        hit = 0
        for t in range(0, n, a.stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, fr = cap.read()
            if not ok:
                continue
            box, _ = red_box(fr)
            if box is None:
                continue
            occ, _ = occlude_box(fr, box)
            store["ep"].append(ep); store["t"].append(t); store["box"].append(box)
            store["orig"].append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB),
                                            (a.w, a.h), interpolation=cv2.INTER_AREA))
            store["occ"].append(cv2.resize(cv2.cvtColor(occ, cv2.COLOR_BGR2RGB),
                                           (a.w, a.h), interpolation=cv2.INTER_AREA))
            hit += 1
        cap.release()
        print(f"  ep{ep:03d}: {hit} 帧命中", flush=True)
    out = os.path.join(a.out, "occl.npz")
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in store.items()})
    print(f"写出 {out}：{len(store['ep'])} 帧 / {len(picks)} 集")
    for k in range(min(4, len(store["orig"]))):
        row = np.hstack([store["orig"][k], store["occ"][k]])
        cv2.imwrite(os.path.join(a.out, f"sample_{k}.jpg"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))


# ------------------------------------------------------------------ 路线 A
def stage_routeA(a):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pilot_store_laptop as A

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A.set_hist("vlm3"); A.set_img_hist("single"); A.set_pred_k(0)
    z = torch.load(a.models, map_location=dev, weights_only=False)
    info = z.get("info", {})
    c_main = max(z["models"], key=lambda c: info.get(c, {}).get("train_frames", 0))
    m = A.build_model().to(dev)
    m.load_state_dict(z["models"][c_main]); m.eval()

    d = np.load(a.cache)
    orig, occ = d["orig"], d["occ"]        # (N, 48, 64, 3) RGB uint8
    print(f"载入 {len(orig)} 帧，类 c{c_main}")

    def per_img(rgb):
        X = np.repeat(rgb[:, None, None], 3, axis=2)   # 三视角都用同一张头部图（两次调用一致）
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                x = A.prep_img(X[i:i + 256], dev)
                b = x.shape[0] // A.TILES
                st = torch.zeros((b, A.SH, A.NS), device=dev)
                mu, lv, zz, ri, rs = m(x, st, b)
                tgt = F.adaptive_avg_pool2d(x, (A.REC_H, A.REC_W))
                pi = F.binary_cross_entropy(ri, tgt, reduction="none").flatten(1).mean(1)
                out.append(pi.reshape(b, A.TILES).mean(1).cpu().numpy())
        return np.concatenate(out)

    po, pc = per_img(orig), per_img(occ)
    dlt = pc - po
    sub = po[:400]
    nat = float(np.median(np.abs(sub[:, None] - sub[None, :])))
    res = dict(scorer="routeA_per_img", cls=int(c_main), n=int(len(po)),
               per_img_orig=float(np.median(po)), per_img_occ=float(np.median(pc)),
               delta_abs_median=float(np.median(np.abs(dlt))), natural_spread=nat,
               sensitivity=float(np.median(np.abs(dlt)) / (nat + 1e-12)),
               frac_increase=float((dlt > 0).mean()))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(os.path.join(os.path.dirname(a.cache), "sens_routeA.json"), "w"),
              ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ 路线 C
def stage_routeC(a):
    """ws：定位/遮挡 → 冻结 VLA → transformer hidden states（头部视角图像 token）。"""
    import copy
    import cv2
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, "/srv/vla_shared/robodojo/RoboDojo/XPolicyLab/policy/Pi_05/openpi/src")
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import normalize as _normalize
    from openpi.training import config as _config

    cfg = _config.get_config(a.train_config)
    ns = _normalize.load(os.path.join(a.ckpt, "assets", a.repo_id))
    policy = _policy_config.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    mdl = policy._model
    prompt = ("Hang the headphones on the headphone stand, close the laptop, "
              "then place it into the vertical laptop stand.")

    def hidden_batch(frames):
        """一批帧 → 头部视角图像 token 的 hidden states (B, 256, 2048)。"""
        prepared = []
        for fr in frames:
            chw = np.ascontiguousarray(np.transpose(fr, (2, 0, 1)))
            obs = dict(state=np.zeros(14, np.float32),
                       images={"cam_high": chw,
                               "cam_left_wrist": chw.copy(),
                               "cam_right_wrist": chw.copy()},
                       prompt=prompt)
            prepared.append(policy._input_transform(copy.deepcopy(obs)))
        stacked = jax.tree.map(
            lambda *xs: jnp.asarray(np.stack([np.asarray(x) for x in xs])), *prepared)
        observation = _model.Observation.from_dict(stacked)
        toks, mask, ar = mdl.embed_prefix(observation)
        out, _ = mdl.PaliGemma.llm([toks, None], mask=make_attn_mask(mask, ar),
                                   positions=jnp.cumsum(mask, axis=1) - 1)
        return np.asarray(out[0])[:, :256]       # 三路各 256，取头部

    eps = [int(x) for x in a.episodes.split(",")] if a.episodes else None
    picks = []
    for f in sorted(glob.glob(os.path.join(a.videos, "episode_*_cam_head_*.mp4"))):
        ep = int(os.path.basename(f).split("_")[1])
        if eps is None or ep in eps:
            picks.append((ep, f))

    E_o, E_c = [], []
    for ep, f in picks:
        cap = cv2.VideoCapture(f)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        po, pc = [], []
        for t in range(0, n, a.stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, fr = cap.read()
            if not ok:
                continue
            box, _ = red_box(fr)
            if box is None:
                continue
            occ, _ = occlude_box(fr, box)
            po.append(fr); pc.append(occ)
            if a.max_frames and len(po) >= a.max_frames:
                break
        cap.release()
        if not po:
            print(f"  ep{ep:03d}: 0 帧命中", flush=True)
            continue
        for i in range(0, len(po), a.batch):
            E_o.append(hidden_batch(po[i:i + a.batch]))
            E_c.append(hidden_batch(pc[i:i + a.batch]))
        print(f"  ep{ep:03d}: {len(po)} 帧", flush=True)

    if not E_o:
        raise SystemExit("没有命中帧")
    Eo, Ec = np.concatenate(E_o), np.concatenate(E_c)
    Vo, Vc = Eo.mean(1), Ec.mean(1)                 # (N,2048) 每帧均值池化
    d_obj = np.linalg.norm(Vo - Vc, axis=1)
    sub = Vo[:min(300, len(Vo))]
    nat = float(np.median(np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)))
    dtok = np.linalg.norm(Eo - Ec, axis=2).mean(1)
    res = dict(scorer="routeC_hidden", n=int(len(Vo)),
               delta_median=float(np.median(d_obj)), token_delta_median=float(np.median(dtok)),
               natural_spread=nat, sensitivity=float(np.median(d_obj) / (nat + 1e-12)),
               frac_increase=float((d_obj > 0).mean()))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(os.path.join(a.out, "sens_routeC.json"), "w"),
              ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["masks", "routeA", "routeC"])
    ap.add_argument("--videos", default="data/eval25/videos")
    ap.add_argument("--out", default="/tmp/e1")
    ap.add_argument("--cache", default="")
    ap.add_argument("--episodes", default="")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--h", type=int, default=48)
    ap.add_argument("--models", default="/tmp/eval25_routeA.pt")
    ap.add_argument("--ckpt", default="/home/syt/tmp_ws/floraU_eval/30000")
    ap.add_argument("--train-config", default="pi05_base_aloha_full_sim_arx-x5_seed_0")
    ap.add_argument("--repo-id", default="arx_x5_sim")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=0, help="每集最多取多少帧（冒烟用）")
    a = ap.parse_args()
    a.cache = a.cache or os.path.join(a.out, "occl.npz")
    {"masks": stage_masks, "routeA": stage_routeA, "routeC": stage_routeC}[a.stage](a)


if __name__ == "__main__":
    main()
