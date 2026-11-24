# 存档：路线 A —— 3 格图像 + 关节状态 + 自重构 VAE

> **冻结日期**：2026-11-24
> **状态**：**已冻结，不再改动**。这是本试点确认可用的方案，作为后续所有对比的基线。
> **用途**：之后要开另一条技术路线时，用它作为对照与"可回退点"，避免两套方案混在一起。
> **红线**：本目录内的文件只读。任何改动都必须新建 `route-B-*` 目录，**不得在本目录内修改**。

---

## 1. 一句话定义

把每一条子任务描述编码成句向量并聚成 5 类；为每一类训练一个**很小的变分自编码器**，输入是「3 相机当前帧 + `t/t−12/t−50` 三帧关节状态」，输出是重建**输入本身**；用共形 p 值判断当前观测属于哪一类，并据此给出细粒度指令或退回任务总名；用「集内平均的任务级 p 值」判断这条 episode 是否属于本任务，不属于就升级给外部大模型。

> 注：按设计文档约定的措辞，这里的"子任务类"是**动作语义类**（抓/放/移动/递交/挂），不是"第几步"。

---

## 2. 精确配置（跑出来就是这套数）

| 项 | 取值 |
|---|---|
| 任务 | `Hang the headphones on the headphone stand, close the laptop, then place it into the vertical laptop stand.` |
| episode 数 | 100（`f35` 中该任务全部） |
| 采样步长 | `--stride 5`（0.2 s 一帧）→ 10,366 个样本 |
| 句向量 | `bge-small-en-v1.5`（本地 `/hdfs/models/huggingface/BAAI/bge-small-en-v1.5`），均值池化 + L2 归一化 |
| 聚类 | 球面 k-means，`k=5`（85 条不同句子） |
| 关节状态历史 | **`--hist vlm3`** = 帧偏移 `[50, 12, 0]`（t−2.0 s / t−0.48 s / t），共 42 维 |
| 图像历史 | **`--img-hist single`** = 3 视角当前帧，共 3 格（48×64） |
| 目标函数 | **`--pred-k 0`** = 重构输入本身（非预测式） |
| 关节项权重 | **`--state-weight 4.0`** |
| 池化强度 | `--pool 3x4`（自适应均值池化到 3×4） |
| 隐变量维度 | 16 |
| 每模型参数量 | **49,413** |
| 训练 | `--iters 1200`，Adam lr=2e-3，bs=64，β=1.0 |
| 划分 | 按 episode 不相交：80 训练 / 20 留出；训练内再 70% 校准 / 30% 自检 |
| OOD 来源 | 8 个其它任务 × 6 集 = 2,819 帧（`--ood-stride 10`） |

---

## 3. 结果（这份存档的基准数）

### 3.1 子任务判别（留出 2,059 帧）

| 指标 | 值 |
|---|---|
| 整体准确率 | **89.1–89.4%** |
| 平衡准确率（5 类，含无法建模的 c3） | 70.3% |
| **平衡准确率（4 个有模型的类）** | **87.8–88.7%**（随机 25%） |
| 各类召回 | c0 99% / c1 85% / c2 79% / c3 0% / c4 88% |
| p ≥ 0.05 的覆盖率 | 96.2% |

> **两次独立运行的差异（87.8 vs 88.7）来自 GPU 非确定性**，配置完全相同。**±1 pt 的差异不可解读。**

### 3.2 OOD 判别（ID 2,059 帧 vs OOD 2,819 帧）

| 指标 | 值 |
|---|---|
| 帧级 AUROC（任务级） | **1.000** |
| 帧级 AUROC（子任务级 max p） | 0.999 |
| 集级 AUROC | 1.000 |
| ID 帧 p_task 中位 / OOD 帧 p_task 中位 | 0.501 / 0.000 |
| 原始分数（未过共形）：ID 留出 / OOD | 1.433 / 1.52–2.97 |

**⚠️ 已知脆弱性（必须一起读）**：ID 侧分数分布极窄（校准中位 1.4266、留出 1.4326，相差 0.006）。阈值是刀刃式的——换布局/光照/场景很可能整集被判成"不认识"而全员升级。**此结论只在演示帧上验证过，未经闭环验证。**

### 3.3 这个方案"不好"的地方（一并存档，避免被误读为全能）

| 项 | 事实 |
|---|---|
| 拒识机制 | **失效**。门槛从 0.02 提到 0.3，覆盖率 98%→70%，但该子集准确率几乎不动（89.5%→89.3%）。分数只能回答"更像哪一类"，不能回答"我判得对不对" |
| 类别不平衡 | 最大类 3,520 帧 / 最小类 24 帧（收尾复位动作，无法建模，召回恒 0） |
| 图像贡献 | 仅约 **1 pt**（88.7% vs 仅状态 87.7%，噪声内） |
| 图像重建质量 | 只能学到"类平均模糊块"，BCE 0.5617 对基线 ln2=0.693，仅解释约 19% |
| 单一任务 | 只在 `store_laptop_and_headphones` 上验证 |

---

## 4. 复现（三段式，两个 Python 环境）

`labeler` 环境有 pyarrow/av/cv2/torch 但没 transformers；`vllm` 环境有 transformers 但没 pyarrow，所以句向量单独一段。

```bash
# 0) 同步代码（注意文件名带 routeA 前缀，重命名成远端临时名即可）
scp -P <port> code/routeA_pilot_store_laptop.py dev@<dev-host>:/tmp/pilot_store_laptop.py

# 1) labeler 环境：定位任务、抽标注
L=/hdfs/users/peize/vlm_vla/env/labeler/bin/python
$L /tmp/pilot_store_laptop.py --stage prepare

# 2) vllm 环境：句向量（只需一次）
V=/hdfs/users/peize/vlm_vla/env/vllm/bin/python
CUDA_VISIBLE_DEVICES=4 $V /tmp/pilot_store_laptop.py --stage embed

# 3) labeler 环境：解码 + 训 VAE + 评估（不带任何参数 = 本存档配置）
CUDA_VISIBLE_DEVICES=4 $L /tmp/pilot_store_laptop.py --stage run
CUDA_VISIBLE_DEVICES=4 $L /tmp/pilot_store_laptop.py --stage ood
CUDA_VISIBLE_DEVICES=4 $L /tmp/pilot_store_laptop.py --stage recon --recon-classes 0
```

**默认值已固定为本存档配置**（`hist=vlm3`、`img-hist=single`、`pred-k=0`、`state-weight=4.0`、`pool=3x4`、`k=5`、`stride=5`、`iters=1200`），所以"不带参数"就是复现本路线——这是刻意的，防止后续忘记传参而跑到别的配置上。

**缓存**：`/tmp/pilot_frames_single_s5_k0_nall.npz`（ID，约 300 MB）与 `/tmp/pilot_ood{i}_single_s10_k0.npz`（OOD）。删掉会自动重解码（ID 约 4 分钟、OOD 约 3 分钟）。

---

## 5. 文件清单与校验和

```
code/
  routeA_pilot_store_laptop.py     51582345a445992d8ef4d584a41fadc7dd77c8f110c57295ea6c60cd762f5072
  routeA_pilot_store_laptop.py.sha256
artifacts/
  spec_task_store_laptop.json      d87bf9c4dc2c196bee8257e800c7af048d03a95d147056ec25eaaf7bd0b2d1cc
  caption_emb_bge_small.npz        59234be6353752e6f2af952af9f11bfa899b73df8a2af104492a21e9fa807f0f
results/
  routing.json                     668c2dafc2aab2128c478516fef0a7985d20741977d877925e97cade8b3c6189
  ood.json                         5a393c1922ad5ae52726f4eecb7e8b89095ebaf5c1d29dba475801367a55606d
  ood_scores.npz                   8e0ef0bdc783242abeab6770ea62e7d4e419cac317966ba460337be407f166bd
  vae_recon_mean_c0.png            6c9f67f2aec154a98db2cdef7835163610224ff4782b689efc72073a349b6006
  vae_recon_samples_c0.png         90a043e4a2ab14dfb7470c3002a8a64f261b99c279f30c024bb38768502ad210
```

校验：`cd 到本目录后 sha256sum -c <<'EOF' ... EOF`（值见上表）。

---

## 6. 依赖与环境（换机器时要能重建）

| 项 | 位置 / 版本 |
|---|---|
| 数据 | 6c `/hdfs/users/peize/vlm_vla/datasets/RoboDojo35FloraUnified-v1`（3,500 集 / 1,859,602 帧 / 12,242 条 prompt / 31,968 段） |
| 标注 | 6c `/hdfs/users/peize/vlm_vla/flora/labels/merged_v1/epNNNNNN.json`（段级：`start_s/end_s/caption`） |
| 每集视频 | 6c `/hdfs/users/peize/vlm_vla/flora/clips/epNNNNNN/{main,left_wrist,right_wrist}.mp4` |
| 句向量模型 | `/hdfs/models/huggingface/BAAI/bge-small-en-v1.5`（384 维） |
| 运行环境 A（数据侧） | `/hdfs/users/peize/vlm_vla/env/labeler/bin/python`（py3.10，torch 2.10.0+cu128，pyarrow 25.0.1，av 15.1.0，cv2 4.12.0） |
| 运行环境 B（句向量） | `/hdfs/users/peize/vlm_vla/env/vllm/bin/python`（py3.10，transformers 5.17.0） |

---

## 7. 与其它方案的关系（防混淆）

| 方案 | 状态 | 位置 |
|---|---|---|
| **路线 A（本存档）** | ✅ 可用，**冻结** | `archive/route-A-vae3tile-state-k0/` |
| 预测式目标 `k=12` | ❌ 已否决（路由 82.6% vs 88.7%） | 代码里保留 `--pred-k`，文档见 `docs/PILOT_STORE_LAPTOP.md` §3.7 |
| 图像 9 格对齐 | ➖ 无收益（87.8% vs 88.7%，参数 1.9 倍） | `--img-hist vlm3` 保留 |
| 连续 0.6 s 四帧状态 | ❌ 劣于 vlm3（88.3% vs 90.6%，早期口径） | `--hist window4` 保留 |
| 后续新路线 | 待开 | 命名 `route-B-*`，**不得修改本目录** |

线上可用的脚本仍在 `scripts/pilot_store_laptop.py`（会继续演进）。**本目录下的 `code/routeA_pilot_store_laptop.py` 是它的冻结副本，且刻意改了文件名**——避免两个同名文件被混用或覆盖。

两者当前的差异**只有文件头的 5 行横幅**（`diff` 可验证）；模型与数据代码逐字节相同。

**若两者不一致，以本目录为准**来判断"路线 A 当时到底是什么"。


---

## 8. 已知边界（不要外推）

1. **只在一个任务上验证**（该任务句子高度模板化：85 句 / 100 集）。
2. **所有结论基于演示帧，没有闭环 rollout**。OOD 阈值尤其需要闭环验证。
3. **共形 p 值在分布内服从均匀分布**，所以"接受率"按构造等于 `1−τ`——不能用高阈值去"接受"。
4. **拒识（判对/判错）不可用**，只有"更像哪一类"。
5. 与 VLM 的 `T+Vnow` / `T+Vnow-int10` 基线相比，**本路线尚未进闭环**，SR 差异未知。
