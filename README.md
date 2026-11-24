# Policy Router Research

研究项目：**用"当前观测属于哪个子任务"的可信程度，决定每一步该给视觉-语言-动作模型（VLA）哪一级语言指令**。

具体地，在三类做法之间做决策：

| 决策 | 给 VLA 的指令 | 代价 |
|---|---|---|
| 用子任务指令 | 由小模型把"这一类动作"结合当前画面补全成一句具体指令 | 几乎为零 |
| 退回任务总名 | 整条任务的一句话 | 零 |
| 升级给外部大模型 | 由 system-2 级模型接管 | 每次约 40 秒（≈1.38 倍单集预算） |

---

## 目录

```
docs/
  PLAN_CN.md                 ← 当前计划的完整说明（标准汉语，可独立阅读）
  PRIOR_RESULTS_20261124.md  ← 前序 900 次试验的核对，以及它推翻了哪几条前提
  PILOT_STORE_LAPTOP.md      ← 单任务试点：第 1/2/3 步全部跑通，含消融与 OOD 验证
  ROUTE_B_ATOMIC_SKILLS.md   ← 路线 B：只按原子技能划分（跨任务通用）的首轮结果
  G2_RESULT.md               ← 判决实验：子任务可分性（PASS）
  DECISIONS.md               ← 决策记录（含被推翻的旧说法与更正）
  CAP_LATENCY_NOTE.md        ← 外部大模型的延迟与成本取数
  01…06_*.md                 ← 问题定义 / 方法规格 / 数据与算力 / 可行性评审 / 实验设计 / 待确认清单
  assets/                    ← 图表

scripts/
  pilot_store_laptop.py      ← 路线 A 的实验脚本（线上演进）
  routeB_atomic_skills.py    ← 路线 B 的实验脚本（跨任务原子技能）
  g2_subtask_separability.py ← 子任务可分性探针
  caption_structure.py       ← 标签结构统计

archive/
  route-A-vae3tile-state-k0/ ← **已冻结的可用方案**（配置、结果、复现命令、校验和）
```

---

## 当前状态

- ✅ 单任务试点跑通：子任务判别平衡准确率 **87.8–88.7%**（随机 25%），OOD 判别帧级 AUROC **1.000**
- ✅ 前序实验核对完成：真实基线是 **9.56%**（36 任务 × 25 布局 = 900 局），而非早期小规模试验的 64–84%
- ⚠️ 未经闭环验证；拒识（判对/判错）机制目前失效
- 🔒 路线 A 已冻结在 `archive/`，后续新路线另建目录
- 🧪 路线 B 首轮（跨任务原子技能）：同任务 49.7%（12 类）、**跨任务仅 18.0%**；诊断显示 N=32 会让簇退化成任务专属

**先读**：[`docs/PLAN_CN.md`](docs/PLAN_CN.md) → [`docs/PRIOR_RESULTS_20261124.md`](docs/PRIOR_RESULTS_20261124.md) → [`docs/PILOT_STORE_LAPTOP.md`](docs/PILOT_STORE_LAPTOP.md)

---

## 复现

路线 A（冻结基线）的三段式命令见 [`archive/route-A-vae3tile-state-k0/README.md`](archive/route-A-vae3tile-state-k0/README.md)。

依赖：Python 3.10 + PyTorch；句向量模型 `bge-small-en-v1.5`；数据来自 RoboDojo 基准的 Flora 标注版本（数据集本体不随仓库分发）。

---

## 说明

- 本仓库只含**代码、文档与派生的实验结果**，不含原始数据集与视频。
- `archive/route-A-vae3tile-state-k0/artifacts/` 内含一个任务的**派生标注文本与句向量**（用于复现），若涉及数据再分发问题可直接移除，不影响代码。
