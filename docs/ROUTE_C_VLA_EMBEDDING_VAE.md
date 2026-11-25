# 路线 C：用冻结 VLA 的中间表示做 VAE 拒识

状态：**已记录待验证**（2026-11-25 由用户提出；单任务先验）。本文记录动机、结构事实、实验设计与风险，尚未产出结果。

## 1. 动机：路线 A 的对象信号几乎为零（已实测）

可视化显示潜空间轨迹几乎就是关节角轨迹。归因实验（`scripts/probe_modality_attribution.py`，30 集演示 / 3,951 评测帧）给出定量结论：

| 测量 | 数值 |
|---|---|
| 分数构成：图像 BCE / 状态项 / KL | 0.582 / **1.513** / 0.389 → **状态占 60.9%**（评测帧 63.2%） |
| 只打乱状态 → Δscore 中位 | +0.080 |
| 只打乱整幅图像 → Δscore 中位 | +0.136 |
| **只替换中央物体带（26% 像素）→ Δscore 中位** | **+0.0132**（同尺寸边缘区 +0.0086） |
| └ 只换头部相机物体带 | +0.0033 |
| └ 只换两个手腕相机物体带 | +0.0100 |
| 评测帧上：图像残差超演示 p90 / 状态残差超 p90 | **41.0%** / 10.0% |

结论：**不是阈值问题，是表征问题**。图像链路 640×480 → 48×64 → 自适应池化 **3×4**，整幅场景只剩 12 个空间格；替换 26% 像素只让分数动 0.5%。因此"机械臂照轨迹走、耳机掉了"这种情形下判据不会触发，CAP 也就不会被拉起。评测帧里图像残差 41% 超阈值，但那是**布局/外观**差异，不是对象状态。

## 2. 路线 C 的设计（用户提出）

> 冻结 VLA 的 vision encoder 与本体 tokenizer，取其后输出的时序 embedding，加一个模块做 VAE（输入 embedding → 复原 embedding）。这样可以在更多倚重于图像的同时，和 VLA 做更好的对齐。

要点：

1. **冻结**：SigLIP 视觉塔、PaliGemma 主干、本体（状态）tokenizer 全部不训练。
2. **表征**：每帧取 prefix 过主干后的 hidden states，压成定长向量。
3. **VAE**：仍做"自重构"（输入 embedding → 复原 embedding），每子任务一个 + 整任务一个。
4. **判据**：沿用两级拒识与共形 p 值，但**分数改在该 embedding 空间上算**（重建误差 / 与校准集的共形 p）。
5. **单任务先验**：只在 `store_laptop_and_headphones` 上验证，成立再推广到 35 任务。

### 2.1 结构事实（已在 ws 上核对源码）

- 实现在 PyTorch 侧：`openpi/src/openpi/models_pytorch/pi0_pytorch.py`，类 `PI0Pytorch`。
- 图像：`embed_prefix()` 里逐路 `paligemma_with_expert.embed_image(img)` → SigLIP tokens；224×224 输入对应 **每路 256 token**（16×16 patch），三路共 768。
- 语言：`embed_language_tokens(lang_tokens) × sqrt(dim)`，与图像 token 拼接，全注意力（`att_masks` 置 0）。
- **本体状态在 pi05 里不进 `embed_suffix`**：`embed_suffix` 的 `state_proj` 只在 `not self.pi05` 分支生效；pi05 的 `_preprocess_observation` 返回的是 `tokenized_prompt / tokenized_prompt_mask / state`，即**状态被离散化成 token 混在 prompt 里**。所以"本体 tokenizer 之后"的那段就是 **prompt token 段**，不是单独的 state 向量。
- 环境：ws 上 `/srv/vla_shared/robodojo/RoboDojo/XPolicyLab/policy/Pi_05/openpi/.venv`（torch 2.10.0+cu128、transformers 4.53.2，与代码要求的 4.53.2 一致）；ckpt 在 `~/tmp_ws/floraU_eval/30000`（12 GB）。**6c 上没有 openpi 源码**（只有 1.5 TB 训练 ckpt），所以抽取要在 ws 做。

### 2.2 抽取方式（不猜 obs 格式）

不自己拼 observation（图像归一化、状态归一化、prompt 里状态 token 的编码方式任何一处错了都会静默学垃圾），而是：

1. 用与评测**完全相同**的路径加载策略（`create_trained_policy` + ckpt + norm stats）；
2. monkeypatch `PI0Pytorch.embed_prefix` 抓输入/输出 token，并在 `paligemma_with_expert.paligemma.language_model` 上挂 forward hook 抓 hidden states；
3. 跑正常 `policy.infer(obs)`（抓完即可中止，不必算完扩散），按形状切出：`hidden[:, :256]` 头部、`[256:512]` 左手腕、`[512:768]` 右手腕、`[768:]` prompt+本体 token；
4. 每帧压成定长：三路图像 token 各做均值池化（3×d）+ prompt/本体 token 均值池化（1×d），d≈2048 → 8192 维；fp16 存。

这样"时序"由采样 stride 体现（与路线 A 同一套 t / t−12 / t−50），后续也可把相邻帧 embedding 拼起来做真时序。

### 2.3 成本估算

- 一次前向不跑扩散采样，只取 prefix：批量 64 帧在 6000 级卡上估计数百 ms/批（待实测）。
- 40 集 × ~530 帧 / stride 5 ≈ 4,200 帧 → 约 66 批。**抽 40 集是分钟级**，不是小时级。
- 存储：4,200 × 8,192 × 2 B ≈ 69 MB（fp16）。全 100 集约 172 MB，可接受。

## 3. 判决实验（先做这个，再谈实现）

**E1 · 遮挡敏感度（决定路线 C 成不成立）**

在同一批演示帧上，比较两种表征对"物体被遮挡"的敏感度：

| 表征 | 处理 |
|---|---|
| A（现路线） | 48×64 → 3×4 池化后的图像特征 |
| C（路线 C） | 冻结 VLA 的图像 token 池化（及 prompt 段） |

对每帧把物体区域遮挡（置背景色/替换为他帧同区域），计算 embedding 距离的相对变化，用**同一把尺**比较。判据：**C 的归一化敏感度须显著高于 A**（A 的现测值是 +0.0132 / 总分 2.48 ≈ 0.5%）。若 C 也不敏感，说明瓶颈在"池化"或"VLA 表征本身不编码对象状态"，需要改取 token（保留空间网格）而不是整条路否掉。

**E2 · 子任务可分性**：与路线 A 同协议（k=5、留出 episode、balanced accuracy），确认路线 C 没把可分性丢掉。

**E3 · 跨任务 OOD**：沿用路线 A 的 8 个其它任务，AUROC 对比。

**E4 · 评测轨迹**：25 集评测 + 官方部分得分，看与得分的相关性是否优于路线 A 的 Spearman −0.48。

## 4. 风险

| # | 风险 | 应对 |
|---|---|---|
| C1 | VLA embedding 被 **prompt（含离散化状态）** 主导，图像 token 段信息被注意力稀释 | 分开取：只用图像 token 段 / 只用 prompt 段 / 两者拼接，做消融 |
| C2 | 池化掉对象细节（与路线 A 同病） | 保留 16×16 空间网格做池化（如 4×4），而不是全局均值；E1 一并比较 |
| C3 | 归一化/格式错（静默学垃圾） | 复用生产路径 + 用"同一帧两次前向结果一致""不同帧 embedding 应可区分"做冒烟自检 |
| C4 | ws 单卡与评测任务抢资源 | 抽取批量化、限时；或错峰 |
| C5 | 与 VLA 对齐 = 也继承了 VLA 的盲点 | 这正是 E1 要回答的：E1 不过则路线 C 不成立 |

## 5. 与路线 A 的关系

- 路线 A **冻结不动**（`archive/route-A-vae3tile-state-k0/`），路线 C 是并列的新分支，产物前缀 `routeC_`。
- 复用的部分：子任务聚类、两级拒识的定义、共形 p 值协议、评测数据（`data/eval25/`）。
- 不复用的部分：图像/状态编码器（改为冻结 VLA 中间表示）。

## 6. 复现入口（待实现）

- 抽特征：`scripts/routeC_extract_vla_emb.py`（待写，先在 ws 上对 1 集做冒烟）
- VAE 训练与判定：待写，输入为 `.npz` 的 embedding 表
- 归因尺：`scripts/probe_modality_attribution.py`（已可用，作为 E1 的对照基线）
