# 前端结构与数据/视频存储（现状 → 统一后）

**日期**：2026-11-27　**状态**：✅ 已合并（单一页面 + 任务/数据集两级下拉）

---

## 1. 现状：三个页面、两套数据目录

### 1.1 页面

| 文件 | 大小 | 内容 | 数据来源 |
|---|---|---|---|
| `assets/latent_explorer.html` | 895 KB | **路线 A**：整任务 VAE 16 维潜空间的 PCA，训练 10,366 帧 + 50 局评测 | 数据内嵌在 html 里（单文件） |
| `assets/routeC_latent_explorer.html` | 64 KB | **路线 C · 单任务**（store_laptop）：4 档维度 × 3 种统计量，含 25 局评测与 OOD 触发 | `assets/routeC_latents/*.js` |
| `assets/flora_task_explorer.html` | 64 KB | **路线 C · 多任务**：35 个任务，每任务独立 VAE，仅演示、无评测 | `assets/flora_tasks/*.js` |

后两个是**同一份模板**（`scripts/explorer_template.html`）生成的外壳页，只是数据目录不同——
本来就该是一个页面，之前为了快速验证拆开了。

### 1.2 数据目录（现在）

```
docs/assets/
  routeC_latents/          ← 单任务（store_laptop）的所有数据集
    lof-d32.js  lof-d16.js  lof-d8.js  lof-d4.js     共形 p · 局部密度归一化（★默认）
    prog-d32.js … prog-d4.js                          共形 p · 进度条件 kNN
    glob-d32.js                                       共形 p · 全局 kNN（对照）
    manifest.js  {default, dims:[{key,latent,group,n_tr,n_ev,n_ep,pc_var,note}], groups:[…]}
  flora_tasks/             ← 多任务，每个任务一个 js
    <task_slug>.js                                    单任务演示潜空间（32 维 → PCA3）
    manifest.js  {default, dims:[{key,label,n_tr,n_ep,pc_var,note}], groups:[…]}
```

问题：两个 manifest 是两套 schema，`key` 在不同任务间会重名（都是 `lof-d32`），只能靠"不同目录"区分。

### 1.3 视频目录（现在）

```
docs/assets/videos/
  head/               100 集 × 头部（路线 A 演示，320×240/12fps）
  head_webm/          同上 VP9（浏览器不支持 H.264 时兜底，未入库）
  eval/               50 个 rollout（路线 A 评测，头部）
  eval25/             25 个 rollout × 3 视角（头/左腕/右腕）
  flora/<task_slug>/  每任务 3 集演示（头部，320×240/12fps，约 100 KB/集）
  ep003129/           单集三视角合成（早期素材）
```

命名约定（播放器靠它拼路径）：`<videoDir>/<name>.mp4`，`name` 由 manifest 里该数据集的
`ep_names`（演示）或 `rollouts[i].file`（评测）给出。

### 1.4 数据文件内部结构（每个 `*.js`）

```js
window.ROUTEC_DATA["<key>"] = {
  meta: {title, pageTitle, paneATitle, paneBTitle, n, n_ep, ep_names, pc_var, t_max,
         cls_counts/ no_cls / cls_names, cmode0,                         // 着色
         ev_n, ev_rollouts[{layout,n,ok,official,file,group}],           // 右栏评测（可空）
         ev2_n, ev2_rollouts, ev_fire, alpha0, pSource,                  // OOD 判定
         videoDir, ev_videoDir, eval25VideoDir, train_cams, ev_cams,     // 视频
         videoEpisodes, videoEpisodes 索引, onePane, selectorLabel},      // 单栏/下拉文案
  pc, cls, ep, off, t, score, p, ang,          // 演示：3D 坐标/类别/集号/偏移/时间/分数/p/相位
  ev:  {n, z, t, ep, off, ok, cfg, p},          // 评测轨迹（可空）
  ev2: {n, …},                                  // 第二批（路线 A 用）
  ood: {n, z, label}                            // 其他任务点（可空）
}
```

页面加载协议：`<latentsDir>/<key>.js` → 注入 `<script>` → 读 `window.ROUTEC_DATA[key]`。

---

## 2. 统一方案：一个页面、两层选择（任务 → 数据集）

### 2.1 数据目录加一层"任务"

```
docs/assets/latents/
  manifest.js                                  ← 唯一清单（tasks → datasets）
  store_laptop_and_headphones/
    datasets.json                              ← 该任务的数据集列表（构建时生成）
    lof-d32.js  lof-d16.js  lof-d8.js  lof-d4.js
    prog-d32.js … prog-d4.js  glob-d32.js
  arrange_the_numbers_from_left_to_right_to_form_t/
    datasets.json
    lof-d32.js
  build_a_tower_using_the_wooden_blocks_and_wooden/
    datasets.json
    lof-d32.js
  …（随编码进度继续加目录）
```

**加一个任务 = 加一个目录**（目录名 = 任务 slug，里面放 `datasets.json` + 若干 `*.js`），
再跑一次 `build_latents_manifest.py` 合并清单。页面不需要改。

### 2.2 清单 schema（v2）

```js
window.ROUTEC_MANIFEST = {
  default: {task: "store_laptop_and_headphones", dataset: "lof-d32"},
  tasks: [
    { key: "store_laptop_and_headphones",
      label: "收纳耳机与笔记本电脑（含 25 局评测）",
      videoDir: "videos/head", evVideoDir: "videos/eval25",   // 任务级默认
      datasets: [
        { key: "lof-d32", group: "共形 p · 局部密度归一化（LOF，无进度）★",
          file: "latents/store_laptop_and_headphones/lof-d32.js",
          latent: 32, n_tr: 10590, n_ev: 3951, n_ep: 100, pc_var: [9.17,8.9,7.54],
          note: "100 集 demo · α=0.12 检出 23/24" },
        … ] },
    { key: "arrange_…", label: "Arrange the numbers…",
      videoDir: "videos/flora/arrange_…",
      datasets: [{ key: "lof-d32", file: "latents/arrange_…/lof-d32.js", … }] },
  ]
}
```

- **任务**下拉：`tasks[].label`；
- **数据集**下拉：当前任务的 `datasets`，按 `group` 分 `<optgroup>`；
- 加载：`<file>`（相对页面路径）→ `window.ROUTEC_DATA[<task>@<key>]`（模板内部把两者拼成唯一键）。

### 2.3 页面

只保留 **`assets/routeC_latent_explorer.html`**（模板生成的外壳），删除 `flora_task_explorer.html`；
顶栏两个下拉：`任务` + `数据集`。单任务（有评测）时右栏与 OOD 面板正常显示；纯演示任务自动单栏。

### 2.4 视频不动

视频目录保持现状，由 manifest 的 `videoDir` / `evVideoDir` 指向；新增任务时把 3 集演示视频
放进 `videos/flora/<task_slug>/` 即可（播放器按 `<videoDir>/<name>.mp4` 拼路径）。

---

## 3. 迁移步骤（一次性）

1. `git mv` 现有数据到 `latents/<task>/…`；
2. `build_routec_all_js.py` / `build_multitask_page.py` 改为写 `latents/<task>/` + 该任务的 `datasets.json`；
3. 新增 `scripts/build_latents_manifest.py`：扫 `latents/*/datasets.json` → 合并 `manifest.js` + 生成外壳页；
4. 模板：加 `任务` 下拉、按 `file` 加载、键名做 `<task>@<dataset>` 命名空间；
5. 删除 `flora_task_explorer.html` 与其预览图，文档索引更新。

---

## 4. 合并结果（已执行）

| 项 | 结果 |
|---|---|
| 页面 | **只剩 `assets/routeC_latent_explorer.html`**（65 KB 外壳）；`flora_task_explorer.html` 已删除 |
| 数据 | `assets/latents/<task_slug>/{datasets.json, *.js}`，共 **3 个任务 / 11 套** |
| 清单 | `assets/latents/manifest.js`：`{default:{task,dataset}, tasks:[{key,label,videoDir,evVideoDir,datasets:[…]}]}` |
| 加载 | `<file>` 指向 `latents/<task>/<dataset>.js`，注册键统一为 **`<task>@<dataset>`**（`build_latents_manifest.py` 每次自动规范化） |
| 顶栏 | **任务**下拉（3 项）+ **数据集**下拉（该任务内按评估方法分组；store_laptop 有 9 套，其它任务暂 1 套） |
| 视频 | 不动：`videoDir` / `evVideoDir` 由清单指向（`videos/head`、`videos/eval25`、`videos/flora/<task>/`） |

**行为验证**：store_laptop（10,590 帧 + 25 局评测，双栏）↔ Arrange（11,708 帧、单栏、3 集视频）↔
Build a tower（14,577 帧）来回切换，评测栏与单栏自动切换、视频路径各自正确。

**新增任务只需**：把 `latents/<task_slug>/{datasets.json, lof-d32.js}` 放进去 → 跑一次
`python3 scripts/build_latents_manifest.py`。**不需要改 HTML，也不需要新页面。**
