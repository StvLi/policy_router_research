#!/usr/bin/env python3
"""把 routeA_train35.py 的分片输出（/tmp/rA35_w*.jsonl）汇总成表。

用法：
  python3 scripts/routeA_train35_table.py /tmp/rA35_w*.jsonl --out docs/G35_RESULT.md
"""
import argparse
import glob
import json
import os
import statistics as st


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="")
    ap.add_argument("--md", default="")
    a = ap.parse_args()

    paths = []
    for p in a.paths:
        paths.extend(glob.glob(p) if any(ch in p for ch in "*?[") else [p])
    rows = load(sorted(paths))
    rows.sort(key=lambda r: (r.get("status") != "ok", -(r.get("balanced") or 0)))

    ok = [r for r in rows if r.get("status") == "ok"]
    skip = [r for r in rows if r.get("status") == "skip"]
    err = [r for r in rows if r.get("status") == "error"]

    def agg(key, src=None):
        vals = [r[key] for r in (src or ok) if r.get(key) is not None]
        return dict(n=len(vals),
                    mean=float(st.mean(vals)) if vals else None,
                    median=float(st.median(vals)) if vals else None,
                    min=float(min(vals)) if vals else None,
                    max=float(max(vals)) if vals else None)

    summary = dict(
        n_tasks=len(rows), n_ok=len(ok), n_skip=len(skip), n_error=len(err),
        balanced=agg("balanced"), acc=agg("acc"), acc_avail=agg("acc_avail"),
        cover=agg("cover"), n_frames=agg("n_frames"), n_models=agg("n_models"),
        n_ep=agg("n_ep"),
        n_models_lt2=sum(1 for r in ok if (r.get("n_models") or 0) < 2),
        n_models_lt4=sum(1 for r in ok if (r.get("n_models") or 0) < 4),
    )
    # 平衡准确率的分布
    bals = sorted((r["balanced"] for r in ok if r.get("balanced") is not None))
    if bals:
        qs = [0.1, 0.25, 0.5, 0.75, 0.9]
        summary["balanced_quantiles"] = {f"q{int(q*100)}": float(np_quantile(bals, q)) for q in qs}

    out = dict(summary=summary, rows=rows)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("写出", a.out)

    lines = []
    lines.append("| 任务 | 状态 | 平衡准确率 | 整体准确率 | 可用类准确率 | 覆盖率 | 帧数 | 集数 | 训练出的类 | 跳过类 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            (r.get("task") or "")[:58], r.get("status"),
            fmt(r.get("balanced")), fmt(r.get("acc")), fmt(r.get("acc_avail")),
            fmt(r.get("cover")), fmt(r.get("n_frames"), 0), fmt(r.get("n_ep"), 0),
            fmt(r.get("n_models"), 0),
            ",".join(str(x) for x in r.get("skipped", [])) or "—"))
    md = []
    md.append("## 汇总\n")
    md.append(f"- 任务数 {summary['n_tasks']}：成功 {summary['n_ok']}，跳过 {summary['n_skip']}，报错 {summary['n_error']}")
    if ok:
        md.append(f"- 平衡准确率（已建模类）：均值 {fmt(summary['balanced']['mean'])}，"
                  f"中位 {fmt(summary['balanced']['median'])}，范围 {fmt(summary['balanced']['min'])}–{fmt(summary['balanced']['max'])}")
        md.append(f"- 整体准确率（含被跳过的类，argmax 于 k={ok[0].get('k')}）：均值 {fmt(summary['acc']['mean'])}")
        md.append(f"- 覆盖率（留出帧中类别被建模的比例）：均值 {fmt(summary['cover']['mean'])}")
        md.append(f"- 训练出 <2 个类的任务：{summary['n_models_lt2']}；<4 个类的任务：{summary['n_models_lt4']}")
        if "balanced_quantiles" in summary:
            q = summary["balanced_quantiles"]
            md.append(f"- 平衡准确率分位：q10 {fmt(q['q10'])} / q25 {fmt(q['q25'])} / "
                      f"q50 {fmt(q['q50'])} / q75 {fmt(q['q75'])} / q90 {fmt(q['q90'])}")
    md.append("")
    if skip:
        md.append("### 跳过的任务\n")
        for r in skip:
            md.append(f"- `{r.get('task','')[:70]}` — {r.get('reason','')}")
        md.append("")
    if err:
        md.append("### 报错的任务\n")
        for r in err:
            md.append(f"- `{r.get('task','')[:70]}` — {r.get('reason','')}")
        md.append("")
    md.append("### 逐任务明细\n")
    md.extend(lines)
    md.append("")
    text = "\n".join(md)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text)
        print("写出", a.md)
    print("\n".join(md[:14]))


def np_quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


if __name__ == "__main__":
    main()
