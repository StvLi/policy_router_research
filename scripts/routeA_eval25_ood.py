#!/usr/bin/env python3
"""路线 A 的两级 OOD 判定，作用在**真实评测轨迹**上（不是演示数据）。

判定规则与冻结的路线 A 完全一致（archive/route-A-vae3tile-state-k0）：
  - 输入：3 相机当前帧（48x64，3x4 自适应池化）+ 关节状态 t/t-12/t-50（3 段历史 × 14 维）
  - 每类一个 VAE + 整任务一个 VAE，自重构（--pred-k 0），状态权重 4.0，latent 16
  - 共形 p 值 p = (1 + #{cal >= s}) / (n + 1)
  - 阈值取自 archive/.../results/ood.json：tau_sub=0.4145，tau_task=0.01

逐帧两级判定：
    max_c p_c >= tau_sub   -> "subtask"  子任务分布内（交给 VLM 生成子任务 prompt）
    else p_task >= tau_task -> "task"    子任务都不符，但整体仍在整任务分布内（整任务文本 prompt）
    else                    -> "cap"     超出整任务有效分布（Astra 接管）

两个 stage：
  fit   在 f35 的 store_laptop 标注集上重训路线 A 的 VAE（确定性：VAE seed=类别号，划分 seed=0）
  score 读评测三路视频 + 关节状态落盘，逐帧打分并汇总

评测数据来源：
  - 视频：eval_result/RoboDojo/<task>/Pi_05/arx_x5/.../episode_XXXXXXX_cam_{head,left_wrist,right_wrist}_{success,fail}.mp4
  - 状态：ws 上 RD_DUMP_STATES 落盘的 jsonl（每步一行，state 为 dict）
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_store_laptop as A  # noqa: E402

# 冻结阈值（archive/route-A-vae3tile-state-k0/results/ood.json）
TAU_SUB = 0.4145454545454546
TAU_TASK = 0.01

# 评测视频文件名里的相机名 -> 路线 A 的视角顺序 (main, left_wrist, right_wrist)
EVAL_CAM = {"head": "main", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}
EP_RE = re.compile(r"episode_(\d+)_cam_(head|left_wrist|right_wrist)_(success|fail)\.mp4$")

STATE_KEYS = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)


# --------------------------------------------------------------- 数据：演示集
def task_episodes(dataset, task_key):
    """按 meta 里的 tasks 文本分组 episode（与 routeA_train35 一致）。"""
    import pyarrow.parquet as pq
    out = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(dataset, "meta", "episodes", "chunk-*", "file-*.parquet"))):
        for r in pq.read_table(p, columns=["episode_index", "tasks"]).to_pylist():
            if not r["tasks"]:
                continue
            head = r["tasks"][0].split("Current subtask:")[0].strip()
            if head.lower().startswith("task:"):
                head = head[5:].strip()
            out[head].append(int(r["episode_index"]))
    hits = [k for k in out if task_key.lower() in k.lower()]
    if len(hits) != 1:
        raise SystemExit(f"任务名 {task_key!r} 命中 {len(hits)} 个：{hits[:3]}")
    return hits[0], sorted(out[hits[0]])


def gather_demo(a):
    """返回 (Xi, Xs, Y_ci, caps_all, segs_of_ep)，与路线 A 的取帧逻辑一致。"""
    task_name, eps = task_episodes(a.dataset, a.task)
    eps = eps[: a.episodes]
    print(f"演示集任务: {task_name}  使用 {len(eps)} 集")
    segs_of_ep = {}
    for ep in eps:
        s = A.load_segments(a.labels, ep)
        if s:
            segs_of_ep[ep] = s
    print(f"有标注的 episode: {len(segs_of_ep)}")

    # 句子 -> 下标；只保留冻结句向量表里有的
    bundle = np.load(a.emb, allow_pickle=True)
    emb_caps = list(bundle["caps"])
    emb_vec = bundle["emb"]
    cap_index = {c: i for i, c in enumerate(emb_caps)}
    caps = sorted({x["caption"].strip().lower() for v in segs_of_ep.values() for x in v})
    missing = [c for c in caps if c not in cap_index]
    if missing:
        print(f"[提示] {len(missing)}/{len(caps)} 条句子不在冻结句向量表里，例如 {missing[0][:60]}")
    caps = [c for c in caps if c in cap_index]
    E = np.stack([emb_vec[cap_index[c]] for c in caps]).astype(np.float32)
    cap_lab, _ = A.spherical_kmeans(E, min(a.k, len(caps)))
    cap2clu = {c: int(l) for c, l in zip(caps, cap_lab)}
    print(f"聚类 k={a.k}: {dict(sorted(collections.Counter(cap_lab.tolist()).items()))}")

    st_map = A.load_all_states(a.dataset, list(segs_of_ep))
    Xi, EP, FIDX, Y = [], [], [], []
    for n_done, ep in enumerate(sorted(segs_of_ep), 1):
        st = st_map.get(ep)
        if st is None:
            continue
        views = A.decode_views(a.clips, ep, a.stride, keep_all=True)
        if views is None:
            continue
        T = min((len(views) - 1) // a.stride + 1, (len(st) - 1) // a.stride + 1)
        lab, amb = A.frame_labels(segs_of_ep[ep], len(st))
        for i in range(T):
            f = i * a.stride
            if f >= len(views) or f >= len(st) or amb[f]:
                continue
            j = lab[f]
            if j < 0:
                continue
            cap = segs_of_ep[ep][j]["caption"].strip().lower()
            if cap not in cap2clu:
                continue
            Xi.append(views[[f]])          # (1, 3, H, W, C)
            EP.append(ep)
            FIDX.append(f)
            Y.append(cap2clu[cap])
        if n_done % 20 == 0 or n_done == len(segs_of_ep):
            print(f"  解码 {n_done}/{len(segs_of_ep)} 集，样本 {len(Xi)}", flush=True)
    Xi = np.stack(Xi)
    EP = np.asarray(EP, np.int64)
    FIDX = np.asarray(FIDX, np.int64)
    Y = np.asarray(Y, np.int64)
    Xs = A.state_windows(st_map, EP, FIDX)
    return dict(task=task_name, Xi=Xi, Xs=Xs, Y=Y, EP=EP, FIDX=FIDX,
                n_caps=len(caps), cap_sizes=dict(collections.Counter(cap_lab.tolist())))


def stage_fit(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A.set_hist("vlm3")
    A.set_img_hist("single")
    A.set_pred_k(0)
    A.STATE_W = 4.0
    d = gather_demo(a)

    Xi, Xs, Y = d["Xi"], d["Xs"], d["Y"]
    mu_s = Xs.reshape(-1, A.NS).mean(0)
    sd_s = Xs.reshape(-1, A.NS).std(0) + 1e-6
    Xs = ((Xs - mu_s) / sd_s).astype(np.float32)
    Xst = Xs.reshape(len(Xs), -1)
    Xt = Xi

    eps_sorted = sorted(set(d["EP"].tolist()))
    random.Random(0).shuffle(eps_sorted)
    n_val = max(2, int(round(len(eps_sorted) * 0.25)))
    val = set(eps_sorted[:n_val])
    tr = np.array([e not in val for e in d["EP"]])

    models, cal, info = {}, {}, {}
    for c in range(a.k):
        m = tr & (Y == c)
        info[c] = dict(train_frames=int(m.sum()))
        if m.sum() < a.min_frames:
            info[c]["trained"] = False
            print(f"  [跳过] 类 {c}: 训练帧仅 {m.sum()}")
            continue
        print(f"[fit] 类 {c}: {m.sum()} 帧，训练 VAE（{a.iters} 步）...", flush=True)
        vae, _ = A.train_vae(Xi[m], Xt[m], Xs[m], Xst[m], a.iters, 1.0, dev, seed=c)
        models[c] = vae
        info[c]["trained"] = True
        print(f"  c{c} 训练完成（{m.sum()} 帧）", flush=True)

    # 整任务级 VAE
    m = tr
    print(f"[fit] 整任务 VAE: {m.sum()} 帧，训练 {a.iters} 步...", flush=True)
    tvae, _ = A.train_vae(Xi[m], Xt[m], Xs[m], Xst[m], a.iters, 1.0, dev, seed=99)

    # 校准集：与冻结路线 A **完全一致**——训练帧（所有类混合）的前 70%，
    # 而不是"每类自己的帧"。这一点很关键：混合校准会让 p 值系统性偏高，
    # 冻结的 tau_sub=0.4145 就是在这个 p 标尺上标定出来的。
    idx_tr = np.where(tr)[0]
    cal_idx = idx_tr[: int(len(idx_tr) * 0.7)]
    print(f"[fit] 校准帧 {len(cal_idx)}（全部类混合，路线 A 口径）")
    for c, vae in models.items():
        cal[c] = A.nll_of(vae, Xi[cal_idx], Xt[cal_idx], Xs[cal_idx], Xst[cal_idx], dev)
        info[c]["cal_median"] = float(np.median(cal[c]))
    cal_task = A.nll_of(tvae, Xi[cal_idx], Xt[cal_idx], Xs[cal_idx], Xst[cal_idx], dev)
    torch.save(dict(models={c: m.state_dict() for c, m in models.items()},
                    task=tvae.state_dict(), cal={c: v for c, v in cal.items()},
                    cal_task=cal_task, mu_s=mu_s, sd_s=sd_s, k=a.k,
                    spec=dict(task=d["task"], n_episodes=int(len(set(d["EP"].tolist()))),
                              n_frames=int(len(Y)), cap_sizes=d["cap_sizes"],
                              iters=a.iters, stride=a.stride, tau_sub=TAU_SUB, tau_task=TAU_TASK),
                    info=info), a.models)
    json.dump(dict(task=d["task"], k=a.k, n_frames=int(len(Y)),
                   classes={str(c): v for c, v in info.items()},
                   tau_sub=TAU_SUB, tau_task=TAU_TASK),
              open(a.out_prefix + "_fit.json", "w"), ensure_ascii=False, indent=1)
    print(f"模型写出 {a.models}")


# --------------------------------------------------------------- 数据：评测轨迹
def load_state_dump(path):
    """读 RD_DUMP_STATES 的 jsonl，返回 14 维状态序列（按写入顺序）。"""
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "__reset__" in r:
                continue
            st = r.get("state")
            if not isinstance(st, dict):
                continue
            vec = []
            for key, n in STATE_KEYS:
                v = st.get(key)
                if v is None:
                    vec = None
                    break
                arr = np.asarray(v, np.float32).ravel()
                if arr.size != n:
                    vec = None
                    break
                vec.append(arr)
            if vec is None:
                continue
            rows.append(np.concatenate(vec))
    return np.stack(rows) if rows else np.zeros((0, A.NS), np.float32)


def load_result_json(path):
    """读评测的 _result.json：{ep: {layout_id, success, score}}（权威结果，优先于文件名）。"""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for k, v in (d.get("details") or {}).items():
        try:
            out[int(k)] = dict(layout_id=int(v.get("layout_id", k)),
                               success=bool(v.get("success", False)),
                               score=float(v.get("score", 0.0)))
        except (TypeError, ValueError):
            continue
    return out


def list_eval_episodes(video_dir):
    """返回 {episode_index: {"success": bool, "cams": {cam: path}}}"""
    out = {}
    for p in glob.glob(os.path.join(video_dir, "episode_*_cam_*_*.mp4")):
        m = EP_RE.search(os.path.basename(p))
        if not m:
            continue
        ep, cam, res = int(m.group(1)), m.group(2), m.group(3)
        e = out.setdefault(ep, {"success": res == "success", "cams": {}})
        e["cams"][cam] = p
    return dict(sorted(out.items()))


def _iter_frames(path):
    """按顺序产出 rgb24 帧；优先 PyAV，缺 av 时退回 cv2。"""
    import cv2
    try:
        import av
    except ModuleNotFoundError:
        av = None
    if av is not None:
        with av.open(path) as c:
            for f in c.decode(video=0):
                yield f.to_ndarray(format="rgb24")
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def decode_eval_episode(ep_info, stride):
    """解码一集的三个视角（文件名是 head/left_wrist/right_wrist），按路线 A 视角顺序堆叠。"""
    import cv2
    per_cam = {}
    for cam, path in ep_info["cams"].items():
        out = []
        for i, img in enumerate(_iter_frames(path)):
            if i % stride:
                continue
            out.append(cv2.resize(img, (A.IMG_W, A.IMG_H), interpolation=cv2.INTER_AREA))
        per_cam[EVAL_CAM[cam]] = np.stack(out) if out else None
    if len(per_cam) != 3 or any(v is None for v in per_cam.values()):
        return None
    n = min(len(v) for v in per_cam.values())
    return np.stack([per_cam[v][:n] for v in A.VIEWS], axis=1)   # (T', 3, H, W, C)


def split_states_by_length(states, lengths):
    """按各集视频帧数把状态序列顺序切段；对不上则返回 None（让调用方降级）。"""
    total = sum(lengths)
    if total > len(states):
        return None
    out, pos = {}, 0
    for ep, n in zip(lengths.keys(), lengths.values()):
        out[ep] = states[pos:pos + n]
        pos += n
    return out


def split_states_by_reset(states):
    """按"归零姿态"切集：仿真每次 reset 都把手臂归零、夹爪张开（≈1）。

    比按视频长度切更可靠——视频帧数与落盘行数并不总是严格相等。
    返回 [[rows_of_ep0], [rows_of_ep1], ...]
    """
    if len(states) == 0:
        return []
    is_reset = (np.abs(states[:, :6]).max(1) < 1e-3) & (np.abs(states[:, 7:13]).max(1) < 1e-3)
    starts = [0] + [i for i in range(1, len(states)) if is_reset[i] and not is_reset[i - 1]]
    segs = []
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else len(states)
        segs.append(states[s:e])
    return segs


def stage_score(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A.set_hist("vlm3")
    A.set_img_hist("single")
    A.set_pred_k(0)
    A.STATE_W = 4.0

    z = torch.load(a.models, map_location=dev, weights_only=False)
    k = z["k"]
    mu_s, sd_s, cal, cal_task = z["mu_s"], z["sd_s"], z["cal"], z["cal_task"]
    models, tvae = {}, A.build_model().to(dev)
    for c, sd in z["models"].items():
        m = A.build_model().to(dev)
        m.load_state_dict(sd)
        m.eval()
        models[c] = m
    tvae.load_state_dict(z["task"])
    tvae.eval()

    eps = list_eval_episodes(a.eval_videos)
    if not eps:
        raise SystemExit(f"{a.eval_videos} 下没有 episode_*.mp4")
    official = load_result_json(a.result_json)
    if official:
        n_fix = 0
        for ep, info in eps.items():
            if ep in official:
                if info["success"] != official[ep]["success"]:
                    n_fix += 1
                info["success"] = official[ep]["success"]
                info["layout_id"] = official[ep]["layout_id"]
                info["score"] = official[ep]["score"]
        print(f"按 _result.json 校正 {len(official)} 集的成功标记（与文件名不一致 {n_fix} 处）")
    print(f"评测 episode: {len(eps)} 个（成功 {sum(e['success'] for e in eps.values())}）")

    raw_states = load_state_dump(a.states)
    print(f"状态落盘: {len(raw_states)} 行")

    dec = {}
    for ep, info in eps.items():
        v = decode_eval_episode(info, a.stride)
        if v is None:
            print(f"  [skip] ep{ep} 解码失败")
            continue
        dec[ep] = v
    lengths = {ep: len(v) for ep, v in dec.items()}

    # 状态与 episode 对齐：优先用 reset 归零姿态切段（更可靠），否则退回按视频长度切
    segs = split_states_by_reset(raw_states)
    if len(segs) == len(dec):
        st_by_ep = {ep: segs[i] for i, ep in enumerate(sorted(dec))}
        print(f"状态对齐：按 reset 切成 {len(segs)} 段，与 {len(dec)} 集一一对应")
    else:
        print(f"[warn] reset 切成 {len(segs)} 段但视频有 {len(dec)} 集，退回按视频长度切")
        st_by_ep = split_states_by_length(raw_states, {ep: len(v) * a.stride for ep, v in dec.items()})
        if st_by_ep is None:
            raise SystemExit("状态与视频无法对齐")
    for ep, v in dec.items():
        need = len(v) * a.stride
        st = st_by_ep.get(ep)
        if st is None or len(st) < 2:
            st_by_ep[ep] = np.zeros((max(need, 2), A.NS), np.float32)
        elif len(st) < need:
            pad = np.repeat(st[-1:], need - len(st), axis=0)
            st_by_ep[ep] = np.concatenate([st, pad], axis=0)
        elif len(st) > need:
            st_by_ep[ep] = st[:need]

    rec = dict(episodes=[], per_frame=[], tau_sub=TAU_SUB, tau_task=TAU_TASK,
               models=a.models, states=a.states, eval_videos=a.eval_videos,
               stride=a.stride, n_episodes=len(dec))
    for ep, info in eps.items():
        v = dec.get(ep)
        if v is None:
            continue
        T = len(v)
        v = v[:, None]          # (T, NT=1, 3, H, W, C) —— prep_img 要求时刻维在第二维
        st = st_by_ep[ep]
        f_full = np.arange(T) * a.stride
        Xs = A.state_windows({0: st}, np.zeros(T, np.int64), f_full)
        Xs = ((Xs - mu_s) / sd_s).astype(np.float32)
        Xst = Xs.reshape(T, -1)
        Xt = v
        P = np.zeros((T, k), np.float32)
        Sraw = np.full((T, k), np.inf, np.float32)
        for c, m in models.items():
            s = A.nll_of(m, v, Xt, Xs, Xst, dev)
            Sraw[:, c] = s
            P[:, c] = A.conformal_p(s, cal[c])
        s_task_raw = A.nll_of(tvae, v, Xt, Xs, Xst, dev)
        p_task = A.conformal_p(s_task_raw, cal_task)
        s_sub_raw = Sraw.min(1)          # 最像的那个子任务的原始 NLL（越小越像）
        p_sub = P.max(1)
        cls = P.argmax(1)
        avail = np.array(sorted(models), np.int64)
        has_model = np.isin(cls, avail)
        decision = np.where(p_sub >= TAU_SUB, "subtask",
                            np.where(p_task >= TAU_TASK, "task", "cap"))
        # 运动量（判断"冻结"帧）
        dq = np.abs(np.diff(st, axis=0)).mean(1) if len(st) > 1 else np.zeros(1)
        dq = np.concatenate([dq[:1], dq])[:T]
        # 连续 CAP 段（判断"持续脱轨"还是"偶发"）
        cap_mask = decision == "cap"
        runs, cur = [], 0
        for b in cap_mask:
            cur = cur + 1 if b else 0
            runs.append(cur)
        rec["episodes"].append(dict(
            ep=ep, layout_id=info.get("layout_id", ep), score=info.get("score"),
            success=info["success"], frames=int(T), seconds=round(T * a.stride / 25.0, 2),
            n_subtask=int((decision == "subtask").sum()),
            n_task=int((decision == "task").sum()),
            n_cap=int((decision == "cap").sum()),
            rate_subtask=float((decision == "subtask").mean()),
            rate_task=float((decision == "task").mean()),
            rate_cap=float((decision == "cap").mean()),
            rate_cap_2nd_half=float(cap_mask[T // 2:].mean()) if T > 1 else 0.0,
            first_cap_f=int(f_full[cap_mask][0]) if cap_mask.any() else -1,
            max_cap_run=int(max(runs)) if runs else 0,
            p_sub_median=float(np.median(p_sub)), p_sub_min=float(p_sub.min()),
            p_task_median=float(np.median(p_task)), p_task_min=float(p_task.min()),
            raw_sub_mean=float(np.mean(s_sub_raw)), raw_sub_median=float(np.median(s_sub_raw)),
            raw_task_mean=float(np.mean(s_task_raw)), raw_task_median=float(np.median(s_task_raw)),
            cls_hist={str(int(c)): int((cls == c).sum()) for c in np.unique(cls)},
            cls_hist_modeled={str(int(c)): int(((cls == c) & has_model).sum()) for c in np.unique(cls)},
            move_median=float(np.median(dq)),
            move_frozen_frac=float((dq < 0.005).mean()),
        ))
        rec["per_frame"].append(dict(ep=ep, success=info["success"], f=f_full.tolist(),
                                     p_sub=p_sub.round(6).tolist(),
                                     p_task=p_task.round(6).tolist(),
                                     cls=cls.tolist(), move=dq.round(6).tolist(),
                                     raw_sub=[round(float(x), 6) for x in s_sub_raw],
                                     raw_task=[round(float(x), 6) for x in s_task_raw],
                                     decision=decision.tolist()))
        print(f"  ep{ep:03d} {'OK ' if info['success'] else 'FAIL'} {T:4d} 步 "
              f"sub {decision_eq(decision,'subtask'):5.1%} task {decision_eq(decision,'task'):5.1%} "
              f"cap {decision_eq(decision,'cap'):5.1%}  p_sub中位 {np.median(p_sub):.3f}", flush=True)

    eps_list = rec["episodes"]
    summary = dict(
        n_episodes=len(eps_list),
        n_success=sum(e["success"] for e in eps_list),
        rate_subtask=float(np.mean([e["rate_subtask"] for e in eps_list])) if eps_list else None,
        rate_task=float(np.mean([e["rate_task"] for e in eps_list])) if eps_list else None,
        rate_cap=float(np.mean([e["rate_cap"] for e in eps_list])) if eps_list else None,
        succ_rate_cap=float(np.mean([e["rate_cap"] for e in eps_list if e["success"]])) if any(e["success"] for e in eps_list) else None,
        fail_rate_cap=float(np.mean([e["rate_cap"] for e in eps_list if not e["success"]])) if any(not e["success"] for e in eps_list) else None,
        mean_frames=float(np.mean([e["frames"] for e in eps_list])) if eps_list else None,
    )
    rec["summary"] = summary
    json.dump(summary, open(a.out_prefix + "_summary.json", "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(a.out_prefix + "_frames.npz",
                        **{f"p_sub_{e['ep']}": np.asarray(e["p_sub"], np.float32) for e in rec["per_frame"]},
                        **{f"p_task_{e['ep']}": np.asarray(e["p_task"], np.float32) for e in rec["per_frame"]})
    # 完整逐帧记录可能很大：单独存
    json.dump(rec, open(a.out_prefix + "_scores.json", "w"), ensure_ascii=False)
    print("\n汇总:", json.dumps(summary, ensure_ascii=False))


def decision_eq(decision, value):
    return float(np.mean(decision == value))


def stage_idcheck(a):
    """对照：用同一套 fit 模型给**演示留出集**打分，检验子任务门限在 ID 上的接受率。

    路线 A 的 ood.json 记录 id_subtask_rate=0.9908 / id_cap_rate=0.0083；
    如果这里复现不出来，说明评测轨迹上的低接受率是管线问题而不是分布漂移。
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A.set_hist("vlm3")
    A.set_img_hist("single")
    A.set_pred_k(0)
    A.STATE_W = 4.0
    z = torch.load(a.models, map_location=dev, weights_only=False)
    k = z["k"]
    mu_s, sd_s, cal, cal_task = z["mu_s"], z["sd_s"], z["cal"], z["cal_task"]
    models, tvae = {}, A.build_model().to(dev)
    for c, sd in z["models"].items():
        m = A.build_model().to(dev)
        m.load_state_dict(sd)
        m.eval()
        models[c] = m
    tvae.load_state_dict(z["task"])
    tvae.eval()

    d = gather_demo(a)
    Xi, Xs, Y, EP = d["Xi"], d["Xs"], d["Y"], d["EP"]
    mu2 = Xs.reshape(-1, A.NS).mean(0)
    sd2 = Xs.reshape(-1, A.NS).std(0) + 1e-6
    print(f"演示集逐批 mu/sd 与 fit 时的差异：{np.abs(mu2-mu_s).max():.4f} / {np.abs(sd2-sd_s).max():.4f}")
    Xs = ((Xs - mu_s) / sd_s).astype(np.float32)
    Xst = Xs.reshape(len(Xs), -1)

    eps_sorted = sorted(set(EP.tolist()))
    random.Random(0).shuffle(eps_sorted)
    n_val = max(2, int(round(len(eps_sorted) * 0.25)))
    val = set(eps_sorted[:n_val])
    va = np.array([e in val for e in EP])
    idx = np.where(va)[0]
    print(f"留出集：{len(set(EP[idx].tolist()))} 集 / {len(idx)} 帧")

    Xv, Xt = Xi[idx], Xi[idx]
    Xsv, Xstv = Xs[idx], Xst[idx]
    P = np.zeros((len(idx), k), np.float32)
    Sraw = np.full((len(idx), k), np.inf, np.float32)
    for c, m in models.items():
        sraw = A.nll_of(m, Xv, Xt, Xsv, Xstv, dev)
        Sraw[:, c] = sraw
        P[:, c] = A.conformal_p(sraw, cal[c])
    s_task_raw = A.nll_of(tvae, Xv, Xt, Xsv, Xstv, dev)
    p_task = A.conformal_p(s_task_raw, cal_task)
    s_sub_raw = Sraw.min(1)
    p_sub = P.max(1)
    decision = np.where(p_sub >= TAU_SUB, "subtask", np.where(p_task >= TAU_TASK, "task", "cap"))
    out = dict(n_frames=int(len(idx)), n_episodes=int(len(set(EP[idx].tolist()))),
               rate_subtask=float((decision == "subtask").mean()),
               rate_task=float((decision == "task").mean()),
               rate_cap=float((decision == "cap").mean()),
               acc_against_caption_cluster=float((P.argmax(1) == Y[idx]).mean()),
               raw_sub_quantiles={f"p{q}": float(np.percentile(s_sub_raw, q)) for q in (10, 25, 50, 75, 90)},
               raw_task_quantiles={f"p{q}": float(np.percentile(s_task_raw, q)) for q in (10, 25, 50, 75, 90)},
               raw_sub_mean=float(np.mean(s_sub_raw)), raw_task_mean=float(np.mean(s_task_raw)),
               tau_sub=TAU_SUB, tau_task=TAU_TASK)
    json.dump(out, open(a.out_prefix + "_idcheck.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "score", "idcheck"])
    ap.add_argument("--task", default="headphones",
                    help="任务名子串（数据集里叫 Hang the headphones ..., 不是 store_laptop_and_headphones）")
    ap.add_argument("--dataset", default=A.DEFAULT_DATASET)
    ap.add_argument("--labels", default=A.DEFAULT_LABELS)
    ap.add_argument("--clips", default=A.DEFAULT_CLIPS)
    ap.add_argument("--emb", default="")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--min-frames", type=int, default=200)
    ap.add_argument("--models", default="/tmp/eval25_routeA.pt")
    ap.add_argument("--eval-videos", default="")
    ap.add_argument("--states", default="")
    ap.add_argument("--result-json", default="", help="评测的 _result.json（提供 layout_id 与权威 success）")
    ap.add_argument("--out-prefix", default="/tmp/eval25")
    a = ap.parse_args()
    if not a.emb:
        a.emb = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "archive", "route-A-vae3tile-state-k0",
                             "artifacts", "caption_emb_bge_small.npz")
    if a.stage == "fit":
        stage_fit(a)
    elif a.stage == "idcheck":
        stage_idcheck(a)
    else:
        if not a.eval_videos or not a.states:
            raise SystemExit("score 需要 --eval-videos 与 --states")
        stage_score(a)


if __name__ == "__main__":
    main()
