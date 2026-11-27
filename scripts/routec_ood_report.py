#!/usr/bin/env python3
"""一条命令出 OOD 判据的报告：捕获率 / 误报率（Wilson 区间）/ 触发时机。

用法（在仓库根目录）：
  python3 scripts/routec_ood_report.py                       # 默认 lof，四档维度
  python3 scripts/routec_ood_report.py --stat progress       # 换统计量对照
  python3 scripts/routec_ood_report.py --alpha 0.10 --dims 32

口径见 docs/ROUTE_C_OOD_SPEC.md §5：指标按**集**统计，二项区间用 Wilson。
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "H", os.path.join(HERE, "build_routec_handover_js.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = load_builder()
ALPHA_DEFAULT = 0.12
WIN, NEED = 5, 3


def wilson(k, n, z=1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fires(P, EP, T, a=ALPHA_DEFAULT):
    """逐集首触发（采样点序号）与进度%。"""
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
        out[int(ep)] = (f, None if f < 0 else 100.0 * f / max(1, len(p) - 1),
                        None if f < 0 else float(T[o][f]) / 25.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/routeC_latents")
    ap.add_argument("--stat", default="lof", choices=["lof", "progress", "global"])
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    ap.add_argument("--dims", default="32,16,8,4")
    ap.add_argument("--json", default="", help="把结果另存为 json")
    a = ap.parse_args()

    dims = [int(x) for x in a.dims.split(",") if x.strip()]
    rows = []
    print(f"统计量 {a.stat} · α={a.alpha} · 窗口 {WIN} 帧 ≥{NEED} 帧\n")
    print(f"{'d':>3} {'失败':>4} {'捕获':>9} {'Wilson 95%':>16} {'成功':>4} {'误报':>4} "
          f"{'首触发进度中位':>12} {'早触发<5%':>9}")
    for d in dims:
        files = {k: os.path.join(a.src, f"latent{d}_{k}.npz") for k in ("demo", "ood", "eval")}
        if not all(os.path.exists(v) for v in files.values()):
            print(f"{d:>3}  （缺文件，跳过）")
            continue
        demo = np.load(files["demo"], allow_pickle=True)
        ev = np.load(files["eval"], allow_pickle=True)
        info = dict(latent=d)

        # 只算评测集的 p：借用构建脚本里的同一套实现
        Zd, EPd, Td = demo["Z"].astype(np.float32), demo["EP"], demo["T"]
        Ze, EPe, Te = ev["Z"].astype(np.float32), ev["EP"], ev["T"]
        SUCC, SCORE = ev["SUCCESS"], ev["SCORE"]
        cal = EPd % 4 == 0
        Zf, EPf = Zd[~cal], EPd[~cal]
        Zc, EPc = Zd[cal], EPd[cal]
        if a.stat == "lof":
            _, P = H.lof_p(Zf, EPf, Zc, EPc, Ze, EPe)
        elif a.stat == "progress":
            Pd = H.progress_of(Td, EPd)
            Pe = H.progress_of(Te, EPe)
            _, P = H.progress_knn_p(Zf, Pd[~cal], EPf, Zc, Pd[cal], EPc, Ze, Pe, EPe)
        else:
            S = H.knn_dist(Ze, EPe, Zf, EPf)
            Sc = H.knn_dist(Zc, EPc, Zf, EPf)
            P = (1.0 + (Sc[None, :] >= S[:, None]).sum(1)) / (len(Sc) + 1.0)

        F = fires(P, EPe, Te, a.alpha)
        eps = sorted(set(EPe.tolist()))
        fail = [e for e in eps if not SUCC[EPe == e][0]]
        succ = [e for e in eps if SUCC[EPe == e][0]]
        tp = sum(1 for e in fail if F[e][0] >= 0)
        fp = sum(1 for e in succ if F[e][0] >= 0)
        progs = [F[e][1] for e in eps if F[e][0] >= 0]
        early = sum(1 for x in progs if x < 5)
        lo, hi = wilson(tp, len(fail))
        flo, fhi = wilson(fp, len(succ))
        print(f"{d:>3} {len(fail):>4} {tp:>4}/{len(fail):<4} [{lo*100:4.1f}%, {hi*100:5.1f}%] "
              f"{len(succ):>4} {fp:>4} {np.median(progs):>11.1f}% {early:>4}/{len(progs):<4}")
        rows.append(dict(dim=d, n_fail=len(fail), tp=tp, n_succ=len(succ), fp=fp,
                         cap_rate=tp / max(1, len(fail)), cap_ci=[lo, hi],
                         fp_rate=(fp / len(succ) if succ else None), fp_ci=[flo, fhi],
                         fire_progress_median=float(np.median(progs)) if progs else None,
                         early_ratio=early / max(1, len(progs)),
                         first_fire=[dict(ep=e, progress=F[e][1], t=F[e][2]) for e in eps]))

    if a.json:
        import json
        json.dump(dict(stat=a.stat, alpha=a.alpha, dims=rows), open(a.json, "w"),
                  ensure_ascii=False, indent=1)
        print(f"\n写出 {a.json}")
    print("\n注：误报率的区间只在 1 局成功样本上极宽——上量前不可当作结论（见 PLAN_V2_SCALE §5.2）。")


if __name__ == "__main__":
    sys.exit(main())
