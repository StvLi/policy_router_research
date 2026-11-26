#!/usr/bin/env python3
"""一次性把两套数据都挂到页面上：

  d32/d16/d8/d4  交接包（100 集 demo / 8 局评测 / OOD 任务）—— 逐帧 p 为**进度条件 kNN 回退**
  x32/x16/x64    α 核验版（20 集 demo / 25 局评测）—— 逐帧 p 为 **VAE 真实重建分 + 共形 p**

两者坐标系不同（各自的 PCA 基），页面上按 key 分开选；下拉文案写清训练集规模与 p 的来源。
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


ASSIGN = re.compile(r'window\.ROUTEC_(?:DATA\["[^"]+"\]|MANIFEST)\s*=\s*')


def load_js(path):
    """解析 `window.ROUTEC_... = {...};`（base64 里有 '='，不能用 rindex('=')）。"""
    txt = open(path).read().strip()
    m = ASSIGN.search(txt)
    body = txt[m.end(): txt.rindex(";")].strip()
    return json.loads(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handover-src", default="data/routeC_latents")
    ap.add_argument("--alpha-src", default="/tmp/latent")
    ap.add_argument("--out", default="docs/assets/routeC_latents")
    ap.add_argument("--keep", default="d32", help="默认选中的 key")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    entries, data_keys = [], []

    # ---- 1) 交接包维数 ----
    print("== 交接包 ==", flush=True)
    subprocess.run([PY, os.path.join(HERE, "build_routec_handover_js.py"),
                    "--src", a.handover_src, "--out", a.out], check=True)
    man = load_js(os.path.join(a.out, "manifest.js"))
    entries += man["dims"]
    data_keys += [e["key"] for e in man["dims"]]

    # ---- 2) α 核验版（有真实重建分） ----
    if a.alpha_src and os.path.isdir(a.alpha_src):
        print("== α 核验版 ==", flush=True)
        tmp = "/tmp/_xjslatents"
        os.makedirs(tmp, exist_ok=True)
        subprocess.run([PY, os.path.join(HERE, "build_routec_latents_js.py"),
                        "--src", a.alpha_src, "--out", tmp], check=True)
        man2 = load_js(os.path.join(tmp, "manifest.js"))
        for e in man2["dims"]:
            src_key = e["key"]
            new_key = "x" + src_key[1:]
            ds = load_js(os.path.join(tmp, f"{src_key}.js"))
            ds["meta"]["pageTitle"] = "路线 C · α 核验版（真实重建分）"
            ds["meta"]["pSource"] = "VAE 重建分 + 共形 p"
            ds["meta"]["alpha0"] = 0.12
            ds["meta"]["onlyCurOODDefault"] = True
            ds["meta"]["autoplayFirstFire"] = True
            ds["meta"].setdefault("paneATitle",
                                  f"<b>演示</b> · {ds['meta']['n']:,} 帧 / {ds['meta']['n_ep']} 集")
            ds["meta"]["paneBTitle"] = (ds["meta"].get("paneBTitle", "")
                                        .replace("<b>测试轨迹", "<b>α 核验 · 测试轨迹"))
            js = ("window.ROUTEC_DATA = window.ROUTEC_DATA || {};\n"
                  f"window.ROUTEC_DATA[\"{new_key}\"] = {json.dumps(ds, ensure_ascii=False)};\n")
            p = os.path.join(a.out, f"{new_key}.js")
            open(p, "w").write(js)
            e = dict(e, key=new_key,
                     note=f"{ds['meta']['n_ep']} 集训练 · 25 局评测 · 真实重建分 · "
                          f"α=0.12 检出 {e['tp']}/{e['n_fail']}")
            entries.append(e)
            data_keys.append(new_key)
            print(f"  {new_key}: {e['note']} · {os.path.getsize(p)/1024:.0f} KB")

    entries.sort(key=lambda e: (-int(e.get("latent", 0)), e["key"].startswith("x")))
    default = a.keep if any(e["key"] == a.keep for e in entries) else entries[0]["key"]
    out = dict(default=default, dims=entries,
               note="d* = 交接包（100 集 demo，逐帧 p 为进度条件 kNN 回退）；"
                    "x* = α 核验版（20 集 demo，逐帧 p 为 VAE 真实重建分）。两套 PCA 基不同，不可跨套比较。")
    open(os.path.join(a.out, "manifest.js"), "w").write(
        "window.ROUTEC_MANIFEST = " + json.dumps(out, ensure_ascii=False) + ";\n")
    print(f"\nmanifest: {[e['key'] for e in entries]} 默认 {default}")


if __name__ == "__main__":
    main()
