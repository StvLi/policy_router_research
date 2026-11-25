#!/usr/bin/env python3
"""在 eval25 全量 25 局上核验 α=0.12（及阈值扫描），并做逐局/逐案分析。

输入（本地）：
  /tmp/eval25_score_frames.npz   ws 上 `routeC_vae_failure.py score` 的产物（EP/T/score/p）
  data/eval25/result.json        官方逐局部分得分
输出：
  docs/assets/routeC_alpha_sweep.png     阈值扫描曲线（检出数 / 误报数 vs α）
  docs/assets/routeC_alpha_timing.png    每局触发时刻 vs 进度的散点
  docs/ROUTE_C_ALPHA_VERIFY.md           核验报告
  /tmp/routeC_alpha_metrics.json         报告里的数值
"""
import argparse
import json
import os

import numpy as np

ALPHA0, WIN, NEED = 0.12, 5, 3
IN_SAMPLE = {2, 8, 11, 12, 13, 17, 19, 24}      # 当初调 α 用的 8 局


def fire(p, a, win=WIN, need=NEED):
    p = np.asarray(p)
    for i in range(0, len(p) - win + 1):
        if int((p[i:i + win] < a).sum()) >= need:
            return i
    return -1


def ep_stats(EP, T, P, S, official):
    out = []
    for e in sorted(set(EP.tolist())):
        m = EP == e
        idx = np.where(m)[0]
        o = idx[np.argsort(T[idx])]
        p, t = P[o], T[o]
        rec = dict(ep=int(e), n=int(len(p)),
                   p_med=float(np.median(p)), p_min=float(p.min()),
                   flag=float((p < ALPHA0).mean()),
                   score_mean=float(S[o].mean()),
                   official=float(official.get(str(int(e)), np.nan)),
                   progress_pct=float((t.max() - t.min()) / 25.0 / 30.0 * 100))
        for a in (0.05, 0.10, 0.12, 0.16, 0.20):
            f = fire(p, a)
            rec[f"fire_{a:.2f}"] = int(f)
            rec[f"t_{a:.2f}"] = (round(float(t[f]) / 25.0, 2) if f >= 0 else None)
            rec[f"prog_{a:.2f}"] = (round(float(f) / max(1, len(t) - 1) * 100, 1) if f >= 0 else None)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/tmp/eval25_score_frames.npz")
    ap.add_argument("--result", default="data/eval25/result.json")
    ap.add_argument("--out-json", default="/tmp/routeC_alpha_metrics.json")
    a = ap.parse_args()

    d = np.load(a.frames)
    EP, T, P, S = d["EP"], d["T"], d["p"], d["score"]
    official = {}
    if os.path.exists(a.result):
        r = json.load(open(a.result))
        det = r.get("details", r)
        for k, v in det.items():
            if isinstance(v, dict) and "score" in v:
                official[str(int(k))] = float(v["score"])

    recs = ep_stats(EP, T, P, S, official)
    n = len(recs)
    fail = [r for r in recs if not (r["official"] >= 0.999)]
    succ = [r for r in recs if r["official"] >= 0.999]
    print(f"共 {n} 局：失败 {len(fail)} / 成功 {len(succ)}")

    # 阈值扫描
    sweep = []
    for a_ in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.25]:
        tp = 0; fp = 0
        for r in fail:
            m = EP == r["ep"]; idx = np.where(m)[0]; o = idx[np.argsort(T[idx])]
            if fire(P[o], a_) >= 0: tp += 1
        for r in succ:
            m = EP == r["ep"]; idx = np.where(m)[0]; o = idx[np.argsort(T[idx])]
            if fire(P[o], a_) >= 0: fp += 1
        sweep.append(dict(alpha=a_, tp=tp, fp=fp, n_fail=len(fail), n_succ=len(succ)))
        print(f"  α={a_:.2f}  检出 {tp}/{len(fail)}  误报 {fp}/{len(succ)}")

    base = next(s for s in sweep if abs(s["alpha"] - ALPHA0) < 1e-9)
    oos = [r for r in recs if r["ep"] not in IN_SAMPLE]
    oos_fail = [r for r in oos if not (r["official"] >= 0.999)]
    oos_tp = sum(1 for r in oos_fail if r[f"fire_{ALPHA0:.2f}"] >= 0)
    print(f"\nα={ALPHA0}: 全量 检出 {base['tp']}/{base['n_fail']}，误报 {base['fp']}/{base['n_succ']}")
    print(f"        样本外（{len(oos)} 局，其中失败 {len(oos_fail)}）检出 {oos_tp}/{len(oos_fail)}")

    # 秩相关
    xs = np.array([r["score_mean"] for r in recs if not np.isnan(r["official"])])
    ys = np.array([r["official"] for r in recs if not np.isnan(r["official"])])
    rk = lambda v: np.argsort(np.argsort(v))
    rho = float(np.corrcoef(rk(xs), rk(ys))[0, 1])

    # 触发时刻 vs 进度
    print("\n逐局（α=0.12）：")
    for r in recs:
        tag = "✓成功" if r["official"] >= 0.999 else "失败"
        f = r[f"fire_{ALPHA0:.2f}"]
        ft = ("" if f < 0 else f"触发 {r[f't_{ALPHA0:.2f}']}s / {r[f'prog_{ALPHA0:.2f}']}%")
        ins = "in" if r["ep"] in IN_SAMPLE else "OOS"
        print(f"  ep{r['ep']:2d} {tag} 官方 {r['official']:.1f} [{ins}] p_med {r['p_med']:.3f} "
              f"flag {r['flag']*100:4.1f}%  {ft}")

    out = dict(alpha=ALPHA0, n=n, n_fail=len(fail), n_succ=len(succ),
               sweep=sweep, spearman_rho=rho, oos=dict(n=len(oos), n_fail=len(oos_fail), tp=oos_tp),
               episodes=recs)
    json.dump(out, open(a.out_json, "w"), ensure_ascii=False, indent=1)
    print(f"\n写出 {a.out_json}")


if __name__ == "__main__":
    main()
