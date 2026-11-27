#!/usr/bin/env python3
"""流水线 worker（跑在单卡工作站）：从 6 卡机逐个拉"已编码"的任务 → 训 VAE → 测 → 出结果。

分工：
  6 卡机（GPU4-5）只做编码，产出 <remote>/<task>/demo.npz（E 2048 维 / EP / T）
  本站（单卡）消费：训练该任务的 VAE、算 LOF 共形 p、统计误报指标、写 JSON，然后删掉本地 npz

指标只用**该任务自己的 100 集**（不掺其它任务）：
  - fpr_ep   ：留出集（25% 的集）上的误报率 —— 最接近线上情形
  - fpr_tail ：留出集**每集尾部 20%** 上的误报率 —— 检查参考集是否覆盖到后段（无进度条件下的常见坑）
  - p50_id   ：留出集的 p 中位数（越高越好）
  - 以及 α 扫描

用法（工作站上，后台跑）：
  nohup python3 pipeline_worker.py --interval 60 > worker.log 2>&1 &
  python3 pipeline_worker.py --once          # 只消费一个已就绪任务（调试）
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_routec_handover_js import knn_dist, lof_stat  # noqa: E402
from routeC_vae_failure import build_vae  # noqa: E402

ALPHAS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]


def ssh_base(a):
    return ["ssh", "-p", str(a.port), "-o", "BatchMode=yes", a.remote]


def list_local(a):
    """本地收件箱：返回 {task: (size, mtime)}。"""
    res = {}
    for f in sorted(os.listdir(a.src_dir)) if os.path.isdir(a.src_dir) else []:
        if not f.endswith(".npz"):
            continue
        p = os.path.join(a.src_dir, f)
        st = os.stat(p)
        res[f[:-4]] = (st.st_size, int(st.st_mtime))
    return res


def list_remote(a):
    """返回 {task: (size, mtime)}。"""
    cmd = ssh_base(a) + [f"ls -l --time-style=+%s {a.remote_dir}/*/demo.npz 2>/dev/null"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        size, mtime, path = parts[4], parts[5], parts[-1]
        task = os.path.basename(os.path.dirname(path))
        res[task] = (int(size), int(mtime))
    return res


def fetch(a, task, dst):
    src = f"{a.remote}:{a.remote_dir}/{task}/demo.npz"
    cmd = ["scp", "-P", str(a.port), "-q", "-O", src, dst]
    subprocess.run(cmd, check=True)
    return dst


def train_and_eval(E, EP, T, latent, iters, seed, want_p=False):
    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = E.mean(0), E.std(0) + 1e-6
    En = ((E - mu) / sd).astype(np.float32)

    torch.manual_seed(seed)
    vae = build_vae(E.shape[1], latent).to(dev)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    X = torch.from_numpy(En).to(dev)
    bs = min(64, len(X))
    for _ in range(iters):
        idx = torch.randint(0, len(X), (bs,), device=dev)
        x = X[idx]
        mu_, lv, dec, _ = vae(x)
        rec = dec(mu_)
        loss = F.mse_loss(rec, x) + \
            (-0.5 * torch.sum(1 + lv - mu_.pow(2) - lv.exp(), dim=1).mean() / mu_.shape[1])
        opt.zero_grad(); loss.backward(); opt.step()
    vae.eval()
    with torch.no_grad():
        Z = vae.mu(vae.enc(torch.from_numpy(En).to(dev))).cpu().numpy().astype(np.float32)
    return Z


def pca3(X, n=3):
    mu = X.mean(0, keepdims=True)
    _, sv, Vt = np.linalg.svd((X - mu).astype(np.float64), full_matrices=False)
    var = (sv ** 2) / (sv ** 2).sum()
    return (X - mu) @ Vt[:n].T, var[:n]


def eval_task(E, EP, T, latent=32, iters=6000, seed=0, page_dir=""):
    eps = np.unique(EP)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eps))
    n_cal = max(1, int(round(len(eps) * 0.25)))
    cal_eps = set(eps[perm[:n_cal]].tolist())
    is_cal = np.isin(EP, list(cal_eps))

    Z = train_and_eval(E, EP, T, latent, iters, seed)

    Zf, EPf = Z[~is_cal], EP[~is_cal]
    Zc, EPc, Tc = Z[is_cal], EP[is_cal], T[is_cal]
    dk = knn_dist(Zf, EPf, Zf, EPf)
    s_cal = lof_stat(Zc, EPc, Zf, EPf, dk)
    s_fit = lof_stat(Zf[::5], EPf[::5], Zf, EPf, dk)   # 抽样自查（应明显更小）
    p_cal = (1.0 + (s_cal[None, :] >= s_cal[:, None]).sum(1)) / (len(s_cal) + 1.0)

    # 尾部 20%：每集最后 20% 的帧（用 T 判断）
    tail = np.zeros(len(Tc), bool)
    for e in np.unique(EPc):
        m = EPc == e
        t = Tc[m]
        thr = t.min() + 0.8 * (t.max() - t.min())
        idx = np.where(m)[0]
        tail[idx[t >= thr]] = True
    p_tail = p_cal[tail]

    # 页面用：全部帧的 3D 坐标 + 逐帧 p（共形，用同一校准集）
    if page_dir:
        P3, var = pca3(Z)
        p_all = (1.0 + (s_cal[None, :] >= lof_stat(Z, EP, Zf, EPf, dk)[:, None]).sum(1)) / (len(s_cal) + 1.0)
        np.savez_compressed(os.path.join(page_dir, "demo.page.npz"),
                            pc3=P3.astype(np.float32), ep=EP, t=T, p=p_all.astype(np.float32),
                            pc_var=var.astype(np.float32), is_cal=is_cal)
    rec = dict(n_ep=int(len(eps)), n_frames=int(len(E)), n_cal=int(is_cal.sum()),
               latent=latent, iters=iters, seed=seed,
               p50_id=float(np.median(p_cal)),
               p50_tail=float(np.median(p_tail)) if len(p_tail) else None,
               score_fit_median=float(np.median(s_fit)),
               score_cal_median=float(np.median(s_cal)))
    for al in ALPHAS:
        rec[f"fpr_ep@{al}"] = float((p_cal < al).mean())
        rec[f"fpr_tail@{al}"] = float((p_tail < al).mean()) if len(p_tail) else None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="",
                    help="本地收件箱模式：直接消费该目录下的 *.npz（跨网段时由搬运工先拷进来）")
    ap.add_argument("--remote", default="dev@10.30.13.100")
    ap.add_argument("--port", type=int, default=32375)
    ap.add_argument("--remote-dir", default="/root/rtc/latent")
    ap.add_argument("--out", default="data/routeC_multitask/results")
    ap.add_argument("--tmp", default="/tmp/rtc_in")
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--keep", action="store_true", help="保留本地 npz（默认处理后删除）")
    ap.add_argument("--redo", action="store_true", help="忽略 _state.json，重算已完成的（用于补 page 数据）")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    state_path = os.path.join(a.out, "_state.json")
    state = {} if a.redo else (json.load(open(state_path)) if os.path.exists(state_path) else {})
    page_dir = os.path.join(a.out, "page")
    os.makedirs(page_dir, exist_ok=True)
    prev_sizes = {}

    print(f"worker 启动：{a.remote}:{a.remote_dir} → {a.out}（每 {a.interval}s 轮询）", flush=True)
    while True:
        try:
            cur = list_local(a) if a.src_dir else list_remote(a)
        except Exception as ex:
            print("列目录失败:", ex, flush=True)
            cur = {}
        for task, (size, mtime) in sorted(cur.items()):
            if state.get(task, {}).get("done"):
                continue
            # 就绪判据：文件在上一轮已出现且大小未变（避免拷到半个文件）
            if prev_sizes.get(task) != size:
                continue
            dst = os.path.join(a.tmp, f"{task}.npz")
            t0 = time.time()
            try:
                if a.src_dir:
                    src = os.path.join(a.src_dir, f"{task}.npz")
                    os.replace(src, dst)              # 从收件箱移走，避免重复占空间
                else:
                    fetch(a, task, dst)
                d = np.load(dst)
                td = os.path.join(page_dir, task)
                os.makedirs(td, exist_ok=True)
                rec = eval_task(d["E"], d["EP"], d["T"], a.latent, a.iters, a.seed, page_dir=td)
                rec["task"] = task
                rec["fetch_s"] = round(time.time() - t0, 1)
                rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                json.dump(rec, open(os.path.join(a.out, f"{task}.json"), "w"),
                          ensure_ascii=False, indent=1)
                state[task] = dict(done=True, n_frames=rec["n_frames"], ts=rec["ts"])
                json.dump(state, open(state_path, "w"), ensure_ascii=False, indent=1)
                print(f"[✓] {task}: {rec['n_frames']} 帧 · FPR(α=0.12) "
                      f"{rec['fpr_ep@0.12']*100:.1f}% · tail {rec['fpr_tail@0.12']*100:.1f}% · "
                      f"{rec['fetch_s']}s", flush=True)
            except Exception as ex:
                print(f"[!] {task} 处理失败：{type(ex).__name__}: {ex}", flush=True)
            finally:
                if not a.keep and os.path.exists(dst):
                    os.remove(dst)
            if a.once:
                return
        prev_sizes = {k: v[0] for k, v in cur.items()}
        done = sum(1 for v in state.values() if v.get("done"))
        print(f"... 已处理 {done}/{len(cur) if cur else '?'}", flush=True)
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
