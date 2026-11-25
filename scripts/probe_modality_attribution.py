#!/usr/bin/env python3
"""归因实验：路线 A 的分数到底在读"机械臂状态"还是"被操作对象"？

背景：可视化显示潜空间轨迹几乎就是关节角轨迹。若判据主要读状态，那么
"机械臂照轨迹走、但对象出问题（耳机掉了）"就不会触发 OOD —— 这正是要测的。

四个测量：
  A. 分数分解：per_img（图像重建 BCE） / STATE_W*per_st（状态 MSE） / kl 各占多少
  B. 扰动敏感度：只打乱状态 vs 只打乱图像，看分数变化幅度
  C. 对象区域扰动：只替换图像中央"桌面/物体带"的像素（保留机械臂区域），看分数变化
     对照：替换同样大小的边缘区域（远离物体）
  D. 评测帧上的交叉表：状态残差正常但图像残差异常的帧有多少（对象级 OOD 的可用信号量）

用法（6c）：
  python3 probe_modality_attribution.py --models /tmp/eval25_routeA.pt --episodes 30
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_store_laptop as A          # noqa: E402
import routeA_eval25_ood as R           # noqa: E402


def terms(vae, Xi, Xs, Xst, dev, bs=256):
    """返回 (per_img, per_st_mse, per_kl) 三个分量的逐样本值。"""
    import torch
    import torch.nn.functional as F
    Pi, Ps, Pk = [], [], []
    with torch.no_grad():
        for i in range(0, len(Xi), bs):
            b = slice(i, i + bs)
            x = A.prep_img(Xi[b], dev)
            st = torch.from_numpy(Xs[b]).float().to(dev)
            mu, lv, z, ri, rs = vae(x, st, len(Xi[b]))
            tgt = F.adaptive_avg_pool2d(x, (A.REC_H, A.REC_W))
            per_img = F.binary_cross_entropy(ri, tgt, reduction="none").flatten(1).mean(1)
            per_img = per_img.reshape(len(Xi[b]), A.TILES).mean(1)
            per_st = F.mse_loss(rs, torch.from_numpy(Xst[b]).float().to(dev), reduction="none").mean(1)
            per_kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1) / (A.NS * A.SH)
            Pi.append(per_img.cpu().numpy())
            Ps.append(per_st.cpu().numpy())
            Pk.append(per_kl.cpu().numpy())
    return np.concatenate(Pi), np.concatenate(Ps), np.concatenate(Pk)


def score_from(pimg, pst, pkl, w):
    return pimg + w * pst + pkl


def patch_swap(X, idx2, y0, y1, x0, x1, views=None):
    """把图像的 [y0:y1, x0:x1] 区域换成另一批样本的同一区域（其余像素保持）。"""
    out = X.copy()
    src = X[idx2]
    if views is None:
        out[:, :, :, y0:y1, x0:x1, :] = src[:, :, :, y0:y1, x0:x1, :]
    else:
        out[:, :, views, y0:y1, x0:x1, :] = src[:, :, views, y0:y1, x0:x1, :]
    return out



def build_eval_cache(a, dev):
    """把 25 集评测轨迹按 stride 采样成 (Xi, Xs, F, EP) 缓存，供 D 段使用。"""
    eps = R.list_eval_episodes(a.eval_videos)
    official = R.load_result_json(getattr(a, "result_json", ""))
    raw = R.load_state_dump(a.states)
    segs = R.split_states_by_reset(raw)
    assert len(segs) == len(eps), f"切出 {len(segs)} 段但视频 {len(eps)} 集"
    Xi, Xs, F, EP, SUCC = [], [], [], [], []
    for i, (ep, info) in enumerate(sorted(eps.items())):
        v = R.decode_eval_episode(info, a.stride)
        if v is None:
            continue
        st = segs[i]
        T = len(v)
        need = T * a.stride
        if len(st) < need:
            st = np.concatenate([st, np.repeat(st[-1:], need - len(st), axis=0)], axis=0)
        f_full = np.arange(T) * a.stride
        Xi.append(v[:, None])
        Xs.append(A.state_windows({0: st[:need]}, np.zeros(T, np.int64), f_full))
        F.append(f_full)
        EP.append(np.full(T, ep))
        SUCC.append(np.full(T, bool(official.get(ep, info)["success"])))
    np.savez_compressed(a.eval_frames,
                        Xi=np.concatenate(Xi), Xs=np.concatenate(Xs),
                        F=np.concatenate(F), EP=np.concatenate(EP), SUCC=np.concatenate(SUCC))
    print(f"评测缓存写出 {a.eval_frames}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="/tmp/eval25_routeA.pt")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--emb", default="/tmp/caption_emb_bge_small.npz")
    ap.add_argument("--eval-videos", default="/tmp/eval25_videos")
    ap.add_argument("--states", default="/tmp/states_rd25.jsonl")
    ap.add_argument("--eval-frames", default="/tmp/eval25_npz.npz",
                    help="缓存的评测图像/状态（首次会从视频重建）")
    ap.add_argument("--out", default="/tmp/probe_attrib.json")
    ap.add_argument("--result-json", default="/tmp/eval25_result.json")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--build-eval-cache", action="store_true")
    a = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A.set_hist("vlm3"); A.set_img_hist("single"); A.set_pred_k(0); A.STATE_W = 4.0
    w = float(A.STATE_W)

    z = torch.load(a.models, map_location=dev, weights_only=False)
    mu_s, sd_s = z["mu_s"], z["sd_s"]
    models = {}
    for c, sd in z["models"].items():
        m = A.build_model().to(dev); m.load_state_dict(sd); m.eval(); models[c] = m

    if a.build_eval_cache:
        build_eval_cache(a, dev)

    # ---------- 演示帧 ----------
    a.task = "headphones"
    a.dataset, a.labels, a.clips = A.DEFAULT_DATASET, A.DEFAULT_LABELS, A.DEFAULT_CLIPS
    a.k = z["k"]; a.emb = a.emb
    d = R.gather_demo(a)
    Xi, Xs = d["Xi"], d["Xs"]
    Xs_n = ((Xs - mu_s) / sd_s).astype(np.float32)
    Xst_n = Xs_n.reshape(len(Xs_n), -1)
    print(f"演示样本 {len(Xi)}")

    # 该任务里帧数最多的类当"主模型"
    c_main = max(models, key=lambda c: len(models))
    vae = models[c_main]

    pimg, pst, pkl = terms(vae, Xi, Xs_n, Xst_n, dev)
    base = score_from(pimg, pst, pkl, w)
    out = dict(n_demo=int(len(Xi)), state_w=w, rec=f"REC {A.REC_H}x{A.REC_W}", pool=f"{A.POOL_H}x{A.POOL_W}",
               demo_terms=dict(per_img=float(np.mean(pimg)),
                               state_term=float(w * np.mean(pst)),
                               kl=float(np.mean(pkl)),
                               state_share=float(w * np.mean(pst) / base.mean()),
                               img_share=float(np.mean(pimg) / base.mean())))
    print(f"[A] 演示分数分解：img {np.mean(pimg):.4f} + 4*st {w*np.mean(pst):.4f} "
          f"+ kl {np.mean(pkl):.4f} = {base.mean():.4f}  → 状态占 {out['demo_terms']['state_share']:.1%}")

    # ---------- B. 扰动敏感度 ----------
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xi))

    pimg_s, pst_s, pkl_s = terms(vae, Xi, Xs_n[perm], Xst_n[perm], dev)
    d_state = (w * pst_s + pkl_s) - (w * pst + pkl)
    print(f"[B] 只打乱状态：Δscore 中位 {np.median(d_state):+.3f}  分数 {np.median(score_from(pimg_s,pst_s,pkl_s,w)):.3f} "
          f"(基线 {np.median(base):.3f})")

    pimg_i, pst_i, pkl_i = terms(vae, Xi[perm], Xs_n, Xst_n, dev)
    d_img = score_from(pimg_i, pst_i, pkl_i, w) - base
    print(f"[B] 只打乱整幅图像：Δscore 中位 {np.median(d_img):+.3f}")

    # ---------- C. 对象区域 vs 边缘区域 ----------
    # 48x64 输入：中央带（桌面/物体）约 rows 16..38, cols 14..50；边缘对照取同尺寸的左上角
    obj = patch_swap(Xi, perm, 16, 38, 14, 50)
    edge = patch_swap(Xi, perm, 0, 22, 0, 36)
    pimg_o, _, _ = terms(vae, obj, Xs_n, Xst_n, dev)
    pimg_e, _, _ = terms(vae, edge, Xs_n, Xst_n, dev)
    d_obj = pimg_o - pimg
    d_edge = pimg_e - pimg
    # 按视角拆分：物体带只在头部相机 vs 只在两个手腕相机
    pimg_h, _, _ = terms(vae, patch_swap(Xi, perm, 16, 38, 14, 50, views=[0]), Xs_n, Xst_n, dev)
    pimg_w, _, _ = terms(vae, patch_swap(Xi, perm, 16, 38, 14, 50, views=[1, 2]), Xs_n, Xst_n, dev)
    print(f"[C'] 只换头部相机的物体带：Δscore 中位 {np.median(pimg_h-pimg):+.4f}")
    print(f"[C'] 只换两个手腕相机的物体带：Δscore 中位 {np.median(pimg_w-pimg):+.4f}")
    out["patch_views"] = dict(head=float(np.median(pimg_h - pimg)),
                              wrists=float(np.median(pimg_w - pimg)))
    out["patch"] = dict(obj_delta=float(np.median(d_obj)), edge_delta=float(np.median(d_edge)),
                        obj_p90=float(np.percentile(d_obj, 90)), edge_p90=float(np.percentile(d_edge, 90)))
    print(f"[C] 只换中央物体带：Δscore 中位 {np.median(d_obj):+.4f}（p90 {np.percentile(d_obj,90):+.4f}）")
    print(f"[C] 只换同尺寸边缘区：Δscore 中位 {np.median(d_edge):+.4f}（p90 {np.percentile(d_edge,90):+.4f}）")

    # ---------- D. 评测帧：状态残差 vs 图像残差 ----------
    ev = np.load(a.eval_frames) if os.path.exists(a.eval_frames) else None
    if ev is None:
        print("[D] 缺评测帧缓存，跳过（先跑 cache_eval_frames）")
        out["eval"] = None
    else:
        Ev, Es, Ef = ev["Xi"], ev["Xs"], ev["F"]
        Es_n = ((Es - mu_s) / sd_s).astype(np.float32)
        Est_n = Es_n.reshape(len(Es_n), -1)
        epimg, epst, epkl = terms(vae, Ev, Es_n, Est_n, dev)
        # 用演示分布定 z（每个分量各自标准化）
        z_img = (epimg - np.median(pimg)) / (np.percentile(pimg, 90) - np.median(pimg) + 1e-9)
        z_st = (epst - np.median(pst)) / (np.percentile(pst, 90) - np.median(pst) + 1e-9)
        both = (z_img > 1) & (z_st > 1)
        img_only = (z_img > 1) & (z_st <= 1)
        st_only = (z_st > 1) & (z_img <= 1)
        out["eval"] = dict(n=int(len(Ev)),
                           state_share=float(w * epst.mean() / score_from(epimg, epst, epkl, w).mean()),
                           img_share=float(epimg.mean() / score_from(epimg, epst, epkl, w).mean()),
                           frac_img_hi=float((z_img > 1).mean()),
                           frac_st_hi=float((z_st > 1).mean()),
                           frac_both=float(both.mean()), frac_img_only=float(img_only.mean()),
                           frac_st_only=float(st_only.mean()),
                           z_img_median=float(np.median(z_img)), z_st_median=float(np.median(z_st)))
        print(f"[D] 评测 {len(Ev)} 帧：图像残差超演示 p90 {out['eval']['frac_img_hi']:.1%}、"
              f"状态残差超 p90 {out['eval']['frac_st_hi']:.1%}；"
              f"仅图像异常 {img_only.mean():.1%}、仅状态异常 {st_only.mean():.1%}、两者都异常 {both.mean():.1%}")
        print(f"[D] 评测分数构成：状态占 {out['eval']['state_share']:.1%}、图像占 {out['eval']['img_share']:.1%}")

    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("写出", a.out)


if __name__ == "__main__":
    main()
