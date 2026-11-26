# 交接文档：路线 C 隐空间可视化数据与 encoder

**用途**：把路线 C（冻结 VLA 中间表示 + VAE）在 4 档隐空间维度下的 **latent 状态数据**与**训练好的 encoder** 交给可视化前端 session。
**生成时间**：2026-11-26
**生成脚本**：`scripts/routeC_export_latents.py`（在 ws 上跑，产物已拉回本地）

---

## 1. 一句话背景

冻结 VLA（`Pi_05` 30000）对每帧计算 2048 维中间表示 → 用本任务演示集训练一个 VAE 把它压到 **N 维隐空间**（N = 32/16/8/4）→ 隐空间里的位置与重建残差用来判断"这一帧是否还像正常执行"。
本交接只关心**隐空间长什么样**，不涉及判定阈值。

---

## 2. 数据在哪

全部在仓库内：**`data/routeC_latents/`**（约 40 MB，**不入库**，见 `.gitignore`；如需分享请单独传）。

| 文件 | 内容 | 大小 |
|---|---|---|
| `latent{N}_demo.npz` | **ID**：本任务演示 100 集 / 10,590 帧的 latent | 0.3–1.4 MB |
| `latent{N}_ood.npz` | **OOD**：另一任务（`Arrange the numbers…`）10 集 / 1,228 帧 | 0.03–0.16 MB |
| `latent{N}_eval.npz` | **评测**：本任务 8 局真实 rollout / 1,214 帧 | 0.03–0.16 MB |
| `latent{N}_vae.pt` | 训练好的 VAE（encoder+decoder 全权重、标准化参数、PCA 基、校准分数） | 9.5 MB |
| `export_index.json` | 汇总：每档的帧数、PCA 解释方差、latent 标准差 | 1.4 KB |

> **缺一项（影响判定类功能）**：三个 npz 里都**没有逐帧的 VAE 重建分 `S` 与共形 p 值 `P`**。
> 只有 `Z`/`PC3` 时，可视化方无法复现"哪一帧不像演示"，只能画几何。详见 §9.1。

`N ∈ {32, 16, 8, 4}`，四档用的是**同一份** 2048 维输入、同一划分、同一种子（seed 0），所以可以直接横向比。

---

## 3. 数据字段

### `latent{N}_demo.npz`（ID）

| 字段 | 形状 | 含义 |
|---|---|---|
| `Z` | (10590, N) | VAE 隐空间坐标（encoder 输出均值 mu） |
| `PC3` | (10590, 3) | 在 demo 上拟合的 PCA 前 3 主成分投影（可直接当 3D 坐标画） |
| `EP` | (10590,) | 集号，3100–3199 |
| `T` | (10590,) | 该集内的**原始帧号**（stride 5 采样，25 fps） |
| `CAP` | (10590,) | 阶段标签（定长字符串，见下） |

`CAP` 取值（**按实际文件核对过**）：`pick_headphones`(1,336) / `close_laptop`(2,318) / `pick_laptop`(2,319) /
`place_into_stand`(1,892) / `handover`(1,553) / `hang_headphones`(1,055) / `other`(115) / **`before_start`(2)**。
注意：**没有 `unknown`**，末尾那个 2 帧的小类是 `before_start`（前一份说明写错了）。
由演示集的分段标注（秒）换算到帧号得到，可用于按阶段着色。

### `latent{N}_ood.npz`（OOD）

`Z` (1228, N)、`PC3` (1228, 3)、`EP`、`T`。**没有** `CAP`（另一任务，未做阶段标注）。

### `latent{N}_eval.npz`（真实评测）

| 字段 | 形状 | 含义 |
|---|---|---|
| `Z` / `PC3` | (1214, N) / (1214, 3) | 同上，已用 demo 的 PCA 基投影 |
| `EP` | (1214,) | 布局号 2/8/11/12/13/17/19/24 |
| `T` | (1214,) | 原始帧号 |
| `SUCCESS` | (1214,) bool | 该帧所属 episode 是否成功（只有 ep12 是 true） |
| `SCORE` | (1214,) float | 官方部分得分：ep12=1.0、ep2/ep8=0.8、ep19=0.2、ep11/ep13/ep17/ep24=0.0 |
| `PROGRESS` | (1214,) float | 该集内的归一化进度 0→1 |

### `latent{N}_vae.pt`（torch 权重）

```python
{
  "state": <VAE state_dict>,          # enc / mu / lv / dec
  "latent": N,
  "d_in": 2048,
  "mu": (2048,)  "sd": (2048,),       # 输入标准化（必须在编码前套用）
  "pca_mean": (N,)  "pca_components": (3, N)  "pca_explained": (3,),
  "cal": (n,)                         # 校准集上的分数（做 p 值时用）
}
```

编码新数据（若要自己再抽帧）：

```python
import torch, numpy as np, sys
sys.path.insert(0, "scripts")
from routeC_vae_failure import build_vae

ck = torch.load("data/routeC_latents/latent16_vae.pt", map_location="cpu", weights_only=False)
vae = build_vae(ck["d_in"], ck["latent"]); vae.load_state_dict(ck["state"]); vae.eval()
Zin = (E_2048d - ck["mu"]) / ck["sd"]            # E_2048d: (M, 2048)，冻结 VLA 的表征
with torch.no_grad():
    Z = vae.mu(vae.enc(torch.from_numpy(Zin.astype(np.float32))))   # (M, N)
```

---

## 4. 读图前必须知道的两件事

**① 同一个 3D 散点，在不同 latent 下的"代表性"完全不同。** 因为 PCA 只在 demo 上拟合：

| latent | PC1 / PC2 / PC3 解释方差 | 3D 视图能代表多少 |
|---|---|---|
| 32 | 9.2% / 8.9% / 7.5% | **约 26%** —— 3D 只看到一小部分结构，**别据此下"数据是团状/有洞"的结论** |
| 16 | 11.7% / 10.8% / 9.1% | 约 32% |
| 8 | 18.6% / 15.2% / 13.8% | 约 47% |
| 4 | 31.6% / 28.5% / 22.0% | **约 82%** —— 3D 视图基本代表整体 |

**② 三份数据用的是同一套 PCA 基（在 demo 上拟合）**，所以 OOD / 评测点的偏移方向是有意义的；但反过来，demo 永远"居中"，不要把它当成"demo 更紧凑"的证据。

---

## 5. 值得先画的四张图

1. **同维度下的三色叠加**（demo 蓝 / OOD 红 / 评测绿）：看 OOD 是否被推出去；这是路线 C 的核心卖点。
2. **四档并排的 demo 散点**（2×2 小图）：直观感受"维度压缩后结构是变清晰还是糊掉"。注意结合 §4 的解释方差。
3. **demo 按阶段着色**（用 `CAP`）：验证隐空间里是否出现**阶段顺序结构**（路线 A 曾发现"进度=相位"的闭环，这里可对比）。
4. **评测按官方得分着色**（用 `SCORE`，或按 `EP` 分面）：看失败的轨迹是否落在 demo 之外；`PROGRESS` 可以做成时间轴动画。

---

## 6. 可直接粘贴给可视化 session 的提示词

> 我在做「机器人策略的隐空间异常检测」，现在需要**可视化不同隐空间维度的对比**。数据在仓库 `data/routeC_latents/`，四档 `N ∈ {32,16,8,4}`，每档三个文件：
>
> - `latent{N}_demo.npz`：正常执行（10,590 帧）。字段 `Z`(10590,N) 隐空间坐标、`PC3`(10590,3) 已在 demo 上拟合好的 3D PCA 投影、`EP` 集号、`T` 帧号、`CAP` 阶段标签（`pick_headphones`/`hang_headphones`/`close_laptop`/`pick_laptop`/`handover`/`place_into_stand`/`other`）。
> - `latent{N}_ood.npz`：另一个任务（1,228 帧），字段 `Z`/`PC3`/`EP`/`T`。
> - `latent{N}_eval.npz`：真实评测 8 局（1,214 帧），字段 `Z`/`PC3`/`EP`/`T`/`SUCCESS`/`SCORE`(官方部分得分 0–1)/`PROGRESS`(0→1)。
> - `latent{N}_vae.pt`：对应的 VAE 权重（含 `mu`/`sd` 标准化参数与 PCA 基）。
>
> 请做：
> 1. 四档并排的 3D 散点（可拖拽旋转），同档内 demo 蓝 / OOD 红叠加；
> 2. 同一档里 demo 按 `CAP` 着色的 3D 散点（看阶段结构）；
> 3. 评测轨迹按 `EP` 分面，用 `SUCCESS`/`SCORE` 着色，`PROGRESS` 做时间轴；
> 4. 每档标注 PCA 解释方差（32: 9/9/8%、16: 12/11/9%、8: 19/15/14%、4: 32/28/22%），并在图上注明"3D 只代表该比例的结构"。
>
> 注意：demo 上的 PCA 是拟合基，所以 demo 必然居中；判断 OOD/评测是否"出去"要看同一基下的投影位置，不要只看聚类紧不紧。

---

## 7. 已知局限（请写进图注）

- 只有 **1 个种子**（seed 0）训练；不同种子下隐空间形状会变，但本轮扫描显示**判定指标对 latent 维度不敏感**（AUROC 0.985–0.988，4–32 维无差异）。
- demo 只有 1 个任务、OOD 只有 1 个任务；结构结论不宜外推到其它任务。
- `CAP` 标签来自演示集的分段标注，边界帧可能有一两帧错位。
- 评测只有 8 局、其中 1 局成功；颜色分组很稀疏，不要据此做统计推断。

---

## 8. 相关文档

| 内容 | 路径 |
|---|---|
| 路线 C 预实验完整报告（含 E1/E2 与维度扫描） | `docs/ROUTE_C_PILOT_REPORT.md` |
| 路线 C 设计与工程事实 | `docs/ROUTE_C_VLA_EMBEDDING_VAE.md` |
| 导出脚本 | `scripts/routeC_export_latents.py` |
| 维度扫描脚本 | `scripts/routeC_latent_sweep.py` |

---

## 9. 可视化侧的补充与实测反馈（2026-11-26，接手方补写）

这一节由可视化 session 补写：记录**消费交接包时缺什么、实测到什么**，以及已经做好的接入方式。
四个维数（32/16/8/4）已经全部接进交互页 `docs/assets/routeC_latent_explorer.html`，页眉可切维数。

### 9.1 最关键缺口：逐帧 `S` / `P`（重建分与共形 p）

**现象**：只有 `Z`/`PC3` 时，页面的"判定层"（α 滑块、OOD 星标、检出计数）没有输入。
我先用**潜空间启发式**顶上，效果很差，正好反证了"判定信号在重建残差里、不在潜空间几何里"：

| p 的来源 | α=0.12 时 7 局失败的检出 |
|---|---|
| VAE 重建分 + 共形 p（正式链路，另一份 25 局数据） | **20–24 / 24** |
| 潜空间**马氏距离** + 共形 p（本轮回退方案） | **1 / 7** |
| 潜空间 **kNN 距离**（k=10）+ 共形 p（试过，更差） | **0 / 7**（还会误报成功局） |

（回退方案用"每 4 集留 1 集"做校准，与交接包的 75/25 划分一致。）

**请求**：在导出脚本里给三个 npz 各加两个逐帧字段即可，消费方会自动优先采用（不用改前端代码）：

```python
np.savez_compressed(out, Z=..., PC3=..., EP=..., T=..., CAP=...,
                    S=score_per_frame,            # vae_nll：MSE(dec(mu), x) + beta*KL
                    P=p_per_frame)                # conformal_p(S, ck["cal"])
```

- `S`/`P` 的口径要与 `scripts/routeC_vae_failure.py` 的 `vae_nll` + `conformal_p` 完全一致，
  否则页面上的 α 与报告里的 α 不是同一个东西。
- 有了 `S`/`P`，页面会自动切换成"真实重建分"模式（顶栏那条橙色告警会自动消失）。

### 9.2 浏览器读不了 `.npz`：需要一步转换（已提供）

可视化页是 `file://` 打开的静态页，**读不了 `.npz`（二进制 + zlib）**，`fetch()` 也会被 CORS 拦。
所以交接包不能直接被前端消费，中间必须有一层转换。已经写好：

```bash
python3 scripts/build_routec_handover_js.py        # data/routeC_latents/*.npz → docs/assets/routeC_latents/d{32,16,8,4}.js + manifest.js
python3 scripts/make_routeC_latent_explorer.py     # 重新生成外壳页
```

- 转换后用 `PC3`（**你们拟合的那套基**），不再重算 PCA，保证页面上看到的就是你们图里的坐标。
- 每档产物约 **520 KB**（10,590 demo + 1,228 OOD + 1,214 eval，含逐帧 p），四档共 ~2 MB，可入库。
- 原始 `data/routeC_latents/`（40 MB）仍按 `.gitignore` 不入库。

### 9.3 评测视频路径（页面播放需要）

页面播的是 8 局评测的头/左腕/右腕视频，命名规则固定（已在仓库内，共 75 个文件）：

```
docs/assets/videos/eval25/episode_<layout:07d>_cam_{head|left_wrist|right_wrist}_{success|fail}.mp4
```

`success`/`fail` 由官方 `SCORE >= 1.0` 判定——这与 `latent{N}_eval.npz` 的 `SUCCESS` 一致即可对齐。
若后续评测局数变了（例如另外那份 25 局），请一并说明文件名规则，否则页面只能播头部。

### 9.4 其它建议（不影响渲染，但影响可读性）

| 项 | 建议 |
|---|---|
| **数组顺序** | 现在 demo 是 (EP, T) 升序；请在文档里写明"顺序不作保证/或保证 (EP,T) 升序"，消费方需要按 `EP` 切段算偏移 |
| **时间单位** | 明确 `t_sec = T / 25`（stride 5 采样，25 fps），页面进度条直接用秒，不是采样点序号 |
| **OOD 任务名** | 现在只写 `Arrange the numbers…`；请给**完整指令文本**——VLA 的表征依赖 prompt，换一句指令就不是同一分布 |
| **重复性** | 同一批帧重抽特征**不是逐位可复现**（实测同帧 embedding 最大差 1.25、中位 0.25，但 VAE 打分几乎不变，逐局 p 中位差 <0.005）。建议在 §7 局限里加一句，避免下游误以为能逐位复算 |
| **`CAP` 的 2 帧 `before_start`** | 占比极低，页面图例里会单独列一行；也可以在导出时并入 `other` 以简化图例 |
| **`PROGRESS` 只给了 eval** | 如果想要"demo 按进度着色"，demo 也需要一份同类归一化进度（现在页面用 `T/T_max` 近似） |

### 9.5 已经做好的页面能力（对应 §5 的四张图）

| 交接文档建议的图 | 页面现状 |
|---|---|
| ① 同维度三色叠加（demo / OOD / 评测） | 左栏 demo 按阶段着色 + **OOD 红点叠加**（可关）；右栏 8 条评测轨迹（白点，选中琥珀高亮） |
| ② 四档并排 | 顶栏下拉切换 32/16/8/4，**每档自动换成该档的 PCA 基**，标题与顶栏实时显示 `PC1-3` 解释率 |
| ③ demo 按阶段着色 | 已按 `CAP` 着色，图例用阶段名（`before_start`/`close_laptop`/…），可逐类显隐 |
| ④ 评测按得分着色 / 时间轴 | 右栏按 layout 上色 + 视频（头/左/右三视角）+ 琥珀游标跟随；顶栏显示"失败 7 局" |
