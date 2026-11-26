#!/usr/bin/env python3
"""把"同一种评估方法"的数据集中成一组，一次构建出前端的完整清单。

分组（下拉里按组展示，组内按维数从高到低）：

  共形 p · 局部密度归一化（LOF，无进度）★   ← 默认：lof-d32/d16/d8/d4
  共形 p · 进度条件 kNN（需要进度 u）        ← 对照：prog-d32/d16/d8/d4
  共形 p · 全局 kNN（无 u 的朴素对照）        ← 对照：glob-d32（默认只建 32 维，已足够看出崩坏）
  VAE 重建分（v0，20 集 / 25 局）             ← 可选：mse-x32（--with-mse，需 /tmp/latent）

用法：
  python3 scripts/build_routec_all_js.py                  # LOF + 进度条件 + 全局(d32)
  python3 scripts/build_routec_all_js.py --stats lof      # 只留推荐方案
  python3 scripts/build_routec_all_js.py --with-mse       # 再加 v0 对照
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

GROUPS = {
    "lof": "共形 p · 局部密度归一化（LOF，无进度）★",
    "progress": "共形 p · 进度条件 kNN（需要进度 u）",
    "global": "共形 p · 全局 kNN（无 u 的朴素对照）",
}
PREFIX = {"lof": "lof", "progress": "prog", "global": "glob"}
MSE_GROUP = "VAE 重建分（v0 · 20 集 / 25 局 · 需下游能算 MSE）"
ASSIGN = "window.ROUTEC_"


def parse_js(path):
    txt = open(path).read()
    i = txt.index(ASSIGN)
    body = txt[txt.index("=", i) + 1: txt.rindex(";")]
    return json.loads(body.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handover-src", default="data/routeC_latents")
    ap.add_argument("--out", default="docs/assets/routeC_latents")
    ap.add_argument("--stats", default="lof,progress,global")
    ap.add_argument("--global-dims", default="32")
    ap.add_argument("--with-mse", action="store_true")
    ap.add_argument("--mse-src", default="/tmp/latent")
    ap.add_argument("--keep", default="")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    stats = [s.strip() for s in a.stats.split(",") if s.strip()]
    entries = []

    for stat in stats:
        cmd = [PY, os.path.join(HERE, "build_routec_handover_js.py"),
               "--src", a.handover_src, "--out", a.out, "--stat", stat,
               "--prefix", PREFIX[stat], "--group", GROUPS[stat], "--no-manifest"]
        if stat == "global":
            cmd += ["--dims", a.global_dims]
        print(f"\n== {GROUPS[stat]} ==", flush=True)
        subprocess.run(cmd, check=True)
        side = os.path.join(a.out, f"_entries.{PREFIX[stat]}.json")
        if os.path.exists(side):
            entries += json.load(open(side))

    if a.with_mse and os.path.isdir(a.mse_src):
        print(f"\n== {MSE_GROUP} ==", flush=True)
        tmp = "/tmp/_xjs"
        os.makedirs(tmp, exist_ok=True)
        subprocess.run([PY, os.path.join(HERE, "build_routec_latents_js.py"),
                        "--src", a.mse_src, "--out", tmp], check=True)
        for e in parse_js(os.path.join(tmp, "manifest.js"))["dims"]:
            if e["key"] != "d32":
                continue
            ds = parse_js(os.path.join(tmp, "d32.js"))
            ds["meta"]["pSource"] = "VAE 重建分 + 共形 p（v0 对照）"
            key = "mse-x32"
            open(os.path.join(a.out, f"{key}.js"), "w").write(
                "window.ROUTEC_DATA = window.ROUTEC_DATA || {};\n"
                f"window.ROUTEC_DATA[\"{key}\"] = {json.dumps(ds, ensure_ascii=False)};\n")
            entries.append(dict(e, key=key, group=MSE_GROUP,
                                note=f"{ds['meta']['n_ep']} 集训练 · 25 局评测 · 真实重建分 · "
                                     f"α=0.12 检出 {e['tp']}/{e['n_fail']}"))
            print(f"  {key} 已写出")

    order = [GROUPS[s] for s in stats] + ([MSE_GROUP] if a.with_mse else [])
    entries.sort(key=lambda e: (order.index(e["group"]) if e.get("group") in order else 99,
                                -int(e.get("latent", 0))))
    lead = next((s for s in stats if s), "lof")
    default = a.keep or next((e["key"] for e in entries if e["key"].startswith(PREFIX[lead] + "-")),
                             entries[0]["key"])
    out = dict(default=default, dims=entries,
               groups=[dict(label=g, keys=[e["key"] for e in entries if e.get("group") == g])
                       for g in order if any(e.get("group") == g for e in entries)],
               note="同一组 = 同一种评估方法（组内只有潜空间维数在变）；跨组不可比。"
                    "每个维数各自一套 demo 拟合的 PCA 基，3D 只代表其解释率之和。")
    open(os.path.join(a.out, "manifest.js"), "w").write(
        "window.ROUTEC_MANIFEST = " + json.dumps(out, ensure_ascii=False) + ";\n")

    print(f"\n共 {len(entries)} 套：")
    for g in out["groups"]:
        print(f"  {g['label']}\n    {g['keys']}")
    print(f"默认 {default}")


if __name__ == "__main__":
    main()
