# 35 任务逐一：聚类 + 每类 VAE + 分段有效性（G35）

本文回答一个问题：**路线 A 的"句子聚类 → 每类 VAE → 共形 p 值"在 1 个任务上成立，推广到全部 35 个任务是否普遍成立？**

结论先说：**部分成立**。31/35 个任务能跑通并给出可解释的分段准确率，但跨任务差异极大（平衡准确率 0.261–0.986，中位 0.627），而且**类数不足的任务会把指标虚高**。所以路线 A 不能不加区分地推广成"每个任务都有一套可靠的子任务分布"。

## 1. 配置（与冻结路线 A 完全一致，只有数据规模不同）

| 项 | 取值 |
|---|---|
| 输入 | 3 相机当前帧 48×64（3 张拼 3 格）+ 关节状态 t / t−12 / t−50（3×14） |
| 状态权重 | 4.0；自重构（`--pred-k 0`） |
| 编码器 | 两层卷积 → 自适应池化 3×4 → latent 16；每个类 VAE ≈ 49,413 参数 |
| 聚类 | 句向量（bge-small-en-v1.5，冻结表）球面 k-means，k=5，seed 0 |
| 训练 | 每类 1200 步 Adam，lr 2e-3，batch 64，β=1.0 |
| 划分 | 按 episode 75% / 25% |
| 每任务 episode 数 | **20**（路线 A 是 100） |
| 帧采样 stride | 5（≈5 帧/秒） |
| 最少类帧数 | 150（不足则该类不建模） |
| 判定 | 共形 p 值 `p = (1 + #{cal ≥ s}) / (n+1)`，取 argmax |

脚本：`scripts/routeA_train35.py`（6 卡分片跑完 35 个任务，每个任务一行 jsonl）；
汇总：`scripts/routeA_train35_table.py`。

## 2. 结果

- 任务数 35：**成功 31，跳过 4，报错 0**
- 平衡准确率（只在已建模类上平均）：均值 **0.656**，中位 **0.627**，范围 0.261–0.986
- 整体准确率（argmax 于 5 个类，含被跳过的类）：均值 **0.630**
- 覆盖率（留出帧中类别被建模的比例）：均值 **0.918**
- 平衡准确率分位：q10 0.354 / q25 0.522 / q50 0.627 / q75 0.838 / q90 0.947
- 训练出 <4 个类的任务：**12/31**

### 逐任务明细

| 任务 | 平衡准确率 | 整体准确率 | 覆盖率 | 帧数 | 建模类数 | 跳过类 |
|---|---|---|---|---|---|---|
| Pick up the two T-shaped blocks, swap their positions… | 0.986 | 0.681 | 0.695 | 1114 | 2 | 0,1,3 |
| Insert the three tubes into the rack one by one. | 0.955 | 0.723 | 0.751 | 1296 | 2 | 0,1,2 |
| Pick up the coin from the holder and insert it precisely… | 0.953 | 0.895 | 0.950 | 758 | 2 | 0,1,2 |
| Pick up the mallet and strike all xylophone keys… | 0.947 | 0.923 | 0.978 | 1248 | 3 | 0,4 |
| Hang the headphones…close the laptop…（即路线 A 的任务） | 0.935 | 0.834 | 0.911 | 2035 | 3 | 2,3 |
| Place the four eggs from the basket into the egg holder… | 0.899 | 0.867 | 1.000 | 1653 | 4 | 1 |
| Swap the two blocks using the empty mat… | 0.868 | 0.827 | 0.957 | 1754 | 4 | 0 |
| Pour all the balls from the cup into the vase. | 0.868 | 0.831 | 1.000 | 1419 | 4 | 1 |
| Build a tower using the wooden blocks and wooden boards. | 0.809 | 0.822 | 1.000 | 2908 | 5 | — |
| Pick up the broom, hand it over to the right hand… | 0.793 | 0.679 | 0.830 | 2318 | 3 | 1,3 |
| Cover the blocks from left to right, remember their colors… | 0.748 | 0.768 | 0.971 | 2187 | 4 | 0 |
| Fold the clothes neatly. | 0.743 | 0.639 | 0.885 | 1221 | 3 | 0,4 |
| Press the two red buttons the required number of times… | 0.736 | 0.796 | 0.972 | 1543 | 3 | 0,4 |
| Pour the liquid from the bottle into the cup. | 0.700 | 0.398 | 0.527 | 825 | 2 | 1,2,4 |
| Arrange the letters to spell "RoboDojo" in a row. | 0.656 | 0.656 | 1.000 | 2545 | 5 | — |
| Place the mouse on the mouse pad, push the keyboard into the frame… | 0.627 | 0.694 | 0.940 | 2577 | 4 | 0 |
| Push the T-shaped block to align it precisely… | 0.611 | 0.476 | 0.798 | 1444 | 2 | 0,2,3 |
| Place all stacking toy pieces onto the correct pegs. | 0.593 | 0.656 | 1.000 | 3671 | 5 | — |
| Arrange the five nesting dolls in a row from left to right… | 0.592 | 0.528 | 1.000 | 2288 | 5 | — |
| Pick up two slices of bread, place them into the toaster… | 0.585 | 0.681 | 0.810 | 2047 | 3 | 1,4 |
| Play tic-tac-toe as the first player… | 0.584 | 0.712 | 0.943 | 3400 | 4 | 4 |
| Wait for the opponent to discard a tile, then declare a kong… | 0.573 | 0.516 | 0.779 | 1501 | 4 | 3 |
| Arrange the numbers from left to right to form the largest number… | 0.525 | 0.527 | 1.000 | 2367 | 5 | — |
| Pick up the bottles and throw them into the dustbin… | 0.520 | 0.689 | 0.982 | 1745 | 3 | 2,3 |
| Hold the pen holder with one hand, place all pens into it… | 0.447 | 0.479 | 1.000 | 3069 | 5 | — |
| Hang all the mugs on the mug rack. | 0.409 | 0.514 | 1.000 | 2303 | 5 | — |
| Place all the objects into the box with their front sides facing left. | 0.363 | 0.355 | 1.000 | 3240 | 5 | — |
| Insert and tighten each screw into the nut of the same color. | 0.354 | 0.399 | 1.000 | 5483 | 5 | — |
| Sort the objects by category into the three baskets. | 0.346 | 0.438 | 1.000 | 3183 | 5 | — |
| Observe the object placement order, remember it, then place… | 0.344 | 0.346 | 1.000 | 4697 | 5 | — |
| Remember the first object on the conveyor, then pick the matching object… | 0.261 | 0.195 | 0.791 | 1911 | 4 | 4 |

跳过的 4 个任务（都因"只剩 1 个可训练类"）：
`Plug the charger into the power strip.`、`Stack the three blocks with different textures.`、
`Stack the three bowls together.`、`Pick up the key…insert it into the keyhole.`

## 3. 怎么读这张表

1. **平衡准确率不能跨任务直接比。** 只在 2 个类上算的 0.986（T 形块交换）与在 5 个类上算的 0.809（搭塔）不是同一件事：前者相当于二分类，随机基线 0.50；后者随机基线 0.20。表里前 5 名有 4 个是"建模类数 ≤3"，说明**高分很大程度来自类被跳过**——被跳过的往往是帧数少的"稀有阶段"，把它们剔掉后分类自然变简单。
2. **真正可比的是 11 个"5 类全建模"任务**：0.344 / 0.346 / 0.354 / 0.363 / 0.409 / **0.447** / 0.525 / 0.592 / 0.593 / 0.656 / 0.809（中位 0.447，随机基线 0.20），明显低于单任务路线 A 的 0.887。原因是这些任务每集更长、阶段更多、句子改写更多，5 个类装不下。
3. **覆盖率 0.918 但 12/31 任务类数 <4** —— 覆盖率是按"被建模类的帧"算的，所以它高不代表分段细。真正要看的是"一个任务的 5 个类里能训出几个"。
4. **任务结构与可分性相关**：动词少、阶段重复的（倒球、放鸡蛋、搭塔、摆字母）可分；多步骤装配/排序/记忆类（拧螺丝、分拣、记忆摆放、观察顺序）最差。这些任务恰恰是子任务语言描述本身就很长的类别，句向量聚类的 5 个簇可能对应"同一句话的不同改写"，而不是不同阶段。

## 4. 对两级路由的含义

路线 A 的两级路由在**子任务级**需要"某个子任务分布能接住当前帧"。G35 说明：

- 在可分性高的任务上（top 1/3），子任务级拒识可以给出细粒度信号；
- 在可分性低的任务上（bottom 1/3，平衡准确率 ≈0.26–0.36），argmax 几乎随机，**子任务级判定不可信，应直接退化为整任务级 + CAP**；
- 因此两级路由的**第一级不应是"必选子任务"，而应先判断任务级可分性**：只有 G35 这类离线可分性达到阈值（例如 5 类全建模且平衡准确率 ≥0.5）的任务才启用子任务级，否则默认整任务 prompt。

这正好与本项目"VAE 不做分类器、只做两件事（子任务都不准 → 整任务 prompt；整体超分布 → Astra）"的定位一致：**子任务级只在可信任务上启用**。

## 5. 局限（必须写清楚）

- 每任务只用了 **20 集**（路线 A 是 100 集），留出帧最少 181 帧，单次运行单一种子，**没有方差估计**；与路线 A 的 0.887 不可直接比较。
- `min-frames=150` 使 4 个任务直接跳过、12 个任务少建模 1–3 个类；这不是"分不开"，而是"样本不够"，换 100 集后应重测。
- 阈值 `p≥0.005` 类判据沿用路线 A，未按任务重标定。
- 只测了"留出 episode 上的逐帧 argmax 准确率"，没有测"跨任务/跨布局泛化"，也没测下游 VLA 成功率。

## 6. 复现

```bash
# 6c：6 卡分片（每片 ~6 个任务）
CUDA_VISIBLE_DEVICES=$w python routeA_train35.py --worker $w --workers 6 \
    --episodes-per-task 20 --min-frames 150 --iters 1200 --out /tmp/rA35_w$w.jsonl
# 汇总
python3 scripts/routeA_train35_table.py '/tmp/rA35_w*.jsonl' --out results/g35.json --md docs/G35_RESULT.md
```

原始逐任务输出：`/tmp/rA35_w{0..5}.jsonl`（6c 与本机各一份，35 行）。
