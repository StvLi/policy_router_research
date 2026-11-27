#!/usr/bin/env python3
"""合并 `assets/latents/*/datasets.json` → 唯一的 `latents/manifest.js`，并生成外壳页。

数据布局（加一个任务 = 加一个目录）：
  assets/latents/
    manifest.js                      ← 本脚本产出
    <task_slug>/datasets.json        ← 该任务的数据集清单（各构建脚本产出）
    <task_slug>/<dataset>.js         ← 数据本体（ROUTEC_DATA["<task>@<dataset>"]）

用法：python3 scripts/build_latents_manifest.py
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs/assets")
    ap.add_argument("--default-task", default="")
    ap.add_argument("--default-dataset", default="")
    a = ap.parse_args()

    lat = os.path.join(a.root, "latents")

    def rekey(task_key, path):
        """注册键规范化成 '<task>@<文件名去扩展>'（幂等；文件名就是数据集 key）。"""
        stem = os.path.splitext(os.path.basename(path))[0]
        want = f"{task_key}@{stem}"
        txt = open(path).read()
        m = re.search(r'window\.ROUTEC_DATA\["([^"]+)"\]', txt)
        if not m or m.group(1) == want:
            return
        txt = txt.replace(f'window.ROUTEC_DATA["{m.group(1)}"]',
                          f'window.ROUTEC_DATA["{want}"]', 1)
        open(path, "w").write(txt)

    tasks = []
    for name in sorted(os.listdir(lat)):
        p = os.path.join(lat, name, "datasets.json")
        if not os.path.isfile(p):
            continue
        t = json.load(open(p))
        for d in t.get("datasets", []):
            fp = os.path.join(a.root, d["file"])
            if os.path.exists(fp):
                rekey(t["key"], fp)

        # 只保留文件确实存在的
        t["datasets"] = [d for d in t.get("datasets", [])
                         if os.path.exists(os.path.join(a.root, d["file"]))]
        if t["datasets"]:
            tasks.append(t)

    if not tasks:
        raise SystemExit("没找到任何任务目录")

    # 默认：有评测的任务优先（store_laptop），否则第一个
    evt = next((t for t in tasks if t.get("evVideoDir")), tasks[0])
    dflt = dict(task=a.default_task or evt["key"],
                dataset=a.default_dataset or evt["datasets"][0]["key"])
    if not any(t["key"] == dflt["task"] for t in tasks):
        dflt = dict(task=tasks[0]["key"], dataset=tasks[0]["datasets"][0]["key"])

    man = dict(default=dflt, tasks=tasks,
               note="一个任务一个目录；任务内按评估方法分组，组内只变潜空间维数。跨任务/跨组坐标不可比。")
    out = os.path.join(lat, "manifest.js")
    open(out, "w").write("window.ROUTEC_MANIFEST = " + json.dumps(man, ensure_ascii=False) + ";\n")

    # 外壳页
    shell = dict(pageTitle="Flora · 潜空间浏览器",
                 title="逐个任务：VAE 潜变量 → demo 拟合 PCA 前 3 维（可按数据集/维数切换）",
                 selectorLabel="数据集",
                 taskLabel="任务",
                 dimDir="latents",
                 defaultTask=dflt["task"], defaultDataset=dflt["dataset"],
                 videoDir="", evVideoDir="", eval25VideoDir="",
                 train_cams=["head"], ev_cams=["head", "left_wrist", "right_wrist"])
    tpl = open(os.path.join(HERE, "explorer_template.html"), encoding="utf-8").read()
    html = (tpl.replace("__PRELOAD__", '<script src="latents/manifest.js"></script>')
               .replace("__META__", json.dumps(shell, ensure_ascii=False))
               .replace("__DATA__", "{}")
               .replace("__TITLE__", shell["pageTitle"]))
    page = os.path.join(a.root, "routeC_latent_explorer.html")
    open(page, "w").write(html)

    print(f"{len(tasks)} 个任务 / {sum(len(t['datasets']) for t in tasks)} 套数据集")
    for t in tasks:
        print(f"  {t['key']}: {len(t['datasets'])} 套"
              + (f" · 评测视频 {t['evVideoDir']}" if t.get("evVideoDir") else " · 仅演示"))
    print(f"默认 {dflt['task']} / {dflt['dataset']}")
    print(f"写出 {out} 与 {page}（{os.path.getsize(page)/1024:.0f} KB）")


if __name__ == "__main__":
    main()
