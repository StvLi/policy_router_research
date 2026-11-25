# 25 局评测的原始轨迹与视频（store_laptop_and_headphones / T 臂）

本目录保存 **ws 上跑完的 25 个布局评测**的原始数据，供离线可视化/复查。
生成时间：2026-11-24 22:29–23:02（ws），2026-11-25 拉回本地。

> 体积说明：`videos/`、`states.jsonl`、`scores.json` 合计约 217 MB，**不入库**（见 `.gitignore`）；
> 入库的只有 `manifest.json` 与本文件。

## 数据来源

| 项 | 值 |
|---|---|
| 任务 | `store_laptop_and_headphones` = *Hang the headphones on the headphone stand, close the laptop, then place it into the vertical laptop stand.* |
| 策略 | T 臂（纯 VLA）：`Pi_05` ckpt 30000，`action_type=joint` |
| 布局 | 25 个，`layout_id` 0–24，`num_envs=1` 顺序执行 |
| 结果 | **成功 1/25**；官方部分得分 1.0×1、0.8×2、0.2×6、0.0×16 |
| 视频 | 640×480、25 fps、H.264；每集三路（head / left_wrist / right_wrist） |
| 帧数 | 失败集 801 帧（≈32 s，跑满上限）；成功集 435 帧（≈17 s） |
| 状态 | 逐步落盘的 14 维关节状态，共 19,634 行 |

## 文件清单

```
data/eval25/
├── manifest.json      # 每集：layout_id / success / 官方部分得分 / 帧数 / 三个视频路径（入库）
├── README.md          # 本文件（入库）
├── videos/            # 75 个 mp4 = 25 集 × 3 视角（不入库，178 MB）
│   └── episode_0000012_cam_head_success.mp4  … 命名规则 episode_<7位>_cam_<head|left_wrist|right_wrist>_<success|fail>.mp4
├── states.jsonl       # 19,634 行逐步状态（不入库，41 MB）
├── result.json        # 评测原生 _result.json：details[i] = {layout_id, success, score}
├── scores.json        # 逐帧两級判据结果（见下）
├── idcheck.json       # 演示留出集对照（子任务接受率 0.986 / CAP 0.014）
└── phase.json         # 按进度三段的越界帧占比
```

## `states.jsonl` 结构

一行一个控制步，按实际执行顺序写入（**不是**按 episode 分文件）。字段：

| 字段 | 含义 |
|---|---|
| `ts` | unix 秒（浮点） |
| `env_idx` | 恒为 0（`num_envs=1`） |
| `state` | dict：`left_arm_joint_state`[6]、`left_ee_joint_state`[1]、`right_arm_joint_state`[6]、`right_ee_joint_state`[1] → 拼成 14 维，与数据集 `left_joint_0..6 / right_joint_0..6` **逐维对应，无需置换或符号翻转** |
| `action` | 同结构的策略下发动作 |
| `additional_info` | `{"frequency": 25}` |

第一行是表头（`__keys__` / `__meta__`），不含 `state`，解析时跳过。

### 切分 episode（重要）

文件里没有 episode 边界标记（`reset` 钩子没生效），**必须按“归零姿态”切**：仿真每次 reset 都把两臂归零、夹爪张开。

```python
import json, numpy as np
K = (("left_arm_joint_state", 6), ("left_ee_joint_state", 1),
     ("right_arm_joint_state", 6), ("right_ee_joint_state", 1))
rows = []
for line in open("data/eval25/states.jsonl"):
    r = json.loads(line)
    st = r.get("state")
    if not isinstance(st, dict):
        continue                      # 表头行
    rows.append(np.concatenate([np.asarray(st[k], np.float32).ravel() for k, _ in K]))
S = np.stack(rows)                    # (19634, 14)
is_reset = (np.abs(S[:, :6]).max(1) < 1e-3) & (np.abs(S[:, 7:13]).max(1) < 1e-3)
starts = [0] + [i for i in range(1, len(S)) if is_reset[i] and not is_reset[i-1]]
eps = [S[s:(starts[n+1] if n+1 < len(starts) else len(S))] for n, s in enumerate(starts)]
# 结果：25 段，长度与 25 个 episode 的帧数一一对应
```

切出来是 **25 段**，与 `layout_id` 0–24 顺序对应；已验证每段长度与该集视频帧数一致。

## `scores.json` 结构（两级判据的逐帧输出）

```json
{
  "tau_sub": 0.4145, "tau_task": 0.01, "stride": 5, "n_episodes": 25,
  "episodes":  [ { "ep":0, "layout_id":0, "success":false, "score":0.0,
                   "frames":161, "rate_subtask":1.0, "rate_task":0.0, "rate_cap":0.0,
                   "raw_sub_mean":1.294, "raw_task_mean":1.460, "move_median":0.0092, … } ],
  "per_frame": [ { "ep":0, "f":[0,5,10,…], "p_sub":[…], "p_task":[…],
                   "cls":[…], "raw_sub":[…], "raw_task":[…], "decision":["subtask",…] } ],
  "summary":   { "rate_subtask":1.0, "rate_task":0.0, "rate_cap":0.0, … }
}
```

- `f` 是**原始帧号**（stride=5 采样，所以失败集是 0,5,…,800 共 161 点）。
- `raw_sub` / `raw_task` 是 VAE 重建的原始 NLL（越小越像演示），**未饱和、有分级信息**；`p_sub`/`p_task` 是共形 p 值（冻结口径下几乎全部饱和在高位）。
- `decision` 按冻结规则给出 `subtask` / `task` / `cap`；实测 25 集全部为 `subtask`（见 `docs/EVAL25_OOD.md`）。

## 已知边界

- 成功只有 1 集，不能做成功/失败统计检验。
- 状态与视频的对齐按“段长度 = 帧数”核对过；若某次 reset 姿态非零，切分会错。
- 视频 801 帧 vs 步数 800：多出的是收尾帧，可视化时按 `min(len(video), len(state))` 截断即可。

## 可复现的可视化入口

- 三级联播（head/left_wrist/right_wrist + 状态曲线）：仓库里已有 `docs/assets/videos/ep003129/` 的同类产物可作模板。
- 逐帧判据曲线：`docs/assets/eval25_ood_timeline.png`、`docs/assets/eval25_ood_summary.png`。
