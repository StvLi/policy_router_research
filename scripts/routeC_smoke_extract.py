#!/usr/bin/env python3
"""路线 C 冒烟 v2（JAX/nnx）：从冻结 VLA 取逐帧表征。

ws 上部署的是 **JAX 版 Pi0**（flax nnx），所以不走 PyTorch hook，而是直接调它的内部方法：

  1. policy._input_transform(obs) + Observation.from_dict  —— 与线上完全同一条预处理
  2. model.PaliGemma.img(images[cam], train=False)          —— 冻结 SigLIP：每路 256 token
  3. model.PaliGemma.llm(tokenized_prompt, method="embed")  —— 语言/本体 token embedding
  4. model.embed_prefix(obs)                                —— 上面两者拼接后的 prefix 序列

输出各段形状，并验证确定性（同输入两次）与可区分性（图像翻转）。

用法（ws）：
  /srv/.../openpi/.venv/bin/python routeC_smoke_extract.py --ep 3100
"""
import argparse
import os
import sys

import numpy as np

OPENPI_SRC = "/srv/vla_shared/robodojo/RoboDojo/XPolicyLab/policy/Pi_05/openpi/src"
sys.path.insert(0, OPENPI_SRC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/syt/tmp_ws/floraU_eval/30000")
    ap.add_argument("--train-config", default="pi05_base_aloha_full_sim_arx-x5_seed_0")
    ap.add_argument("--repo-id", default="arx_x5_sim")
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--eval-dir", default="", help="评测目录（ws 本地）；给了就用真实帧")
    ap.add_argument("--clips", default="/hdfs/users/peize/vlm_vla/flora/clips")
    a = ap.parse_args()

    from openpi.models import model as _model
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import normalize as _normalize
    from openpi.training import config as _config

    cfg = _config.get_config(a.train_config)
    ns = _normalize.load(os.path.join(a.ckpt, "assets", str(a.repo_id)))
    policy = _policy_config.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    m = policy._model
    print(f"model={type(m).__name__} pi05={m.pi05} is_pytorch={policy._is_pytorch_model}", flush=True)
    print(f"PaliGemma keys: {list(m.PaliGemma.keys())}", flush=True)

    # ---- 组 obs（与线上同格式）----
    rng = np.random.default_rng(0)
    if a.eval_dir and a.ep is not None:
        import cv2
        import glob as _glob

        def load(cam):
            hits = _glob.glob(os.path.join(a.eval_dir, f"episode_{a.ep:07d}_cam_{cam}_*.mp4"))
            assert hits, f"{a.eval_dir} 里没有 ep{a.ep} 的 {cam}"
            p = hits[0]
            cap = cv2.VideoCapture(p)
            ok, fr = cap.read()
            cap.release()
            assert ok, p
            # transform 链要求 CHW（AlohaInputs 会转回 HWC 再 resize）
            return np.ascontiguousarray(np.transpose(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (2, 0, 1)))
        imgs = {"cam_high": load("head"), "cam_left_wrist": load("left_wrist"),
                "cam_right_wrist": load("right_wrist")}
        print(f"真实帧 ep{a.ep}: {[v.shape for v in imgs.values()]}", flush=True)
    else:
        imgs = {k: rng.integers(0, 255, (3, 480, 640), dtype=np.uint8)
                for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
        print("随机图像（仅测形状）", flush=True)

    obs = dict(state=np.zeros(14, np.float32), images=imgs,
               prompt="Hang the headphones on the headphone stand, close the laptop, "
                      "then place it into the vertical laptop stand.")

    import jax
    import jax.numpy as jnp

    def feats(obs_):
        import copy
        obs_ = copy.deepcopy(obs_)          # 变换链会原地改写输入，必须每次深拷贝
        inputs = policy._input_transform(obs_)
        # 单条 obs 要加 batch 维，且必须是 jax 数组（jaxtyping 会检查类型）
        inputs = jax.tree.map(
            lambda x: jnp.asarray(x)[None, ...] if hasattr(x, "__getitem__") and not isinstance(x, str) else x,
            inputs)
        observation = _model.Observation.from_dict(inputs)
        out = {}
        for name in observation.images:
            tok, _ = m.PaliGemma.img(observation.images[name], train=False)
            out[f"img_{name}"] = np.asarray(tok)
        if observation.tokenized_prompt is not None:
            emb = m.PaliGemma.llm(observation.tokenized_prompt, method="embed")
            out["lang_emb"] = np.asarray(emb)
            out["prompt_len"] = int(observation.tokenized_prompt.shape[1])
        toks, mask, ar = m.embed_prefix(observation)
        out["prefix_tokens"] = np.asarray(toks)
        out["prefix_mask_len"] = int(np.asarray(mask).sum(-1)[0])
        return out

    f1 = feats(obs)
    print("\n=== 形状 ===")
    for k, v in f1.items():
        print(f"  {k}: {v.shape if hasattr(v, 'shape') else v}")
    tot_img = sum(v.shape[1] for k, v in f1.items() if k.startswith("img_"))
    print(f"  三路图像 token 合计 {tot_img}；prompt 段 {f1.get('prompt_len')} → "
          f"prefix 总长 {f1['prefix_tokens'].shape[1]}")

    f2 = feats(obs)
    key = "img_base_0_rgb"
    d_same = float(np.abs(f1[key] - f2[key]).max())
    obs3 = dict(obs)
    obs3["images"] = {k: np.ascontiguousarray(v[:, :, ::-1]) for k, v in imgs.items()}
    f3 = feats(obs3)
    d_diff = float(np.abs(f1[key] - f3[key]).mean())
    print(f"\n确定性（同输入两次）max|Δ| = {d_same:.3e}")
    print(f"可区分性（图像左右翻转）mean|Δ| = {d_diff:.3e}")
    a1 = f1["img_base_0_rgb"]
    b1 = f1["img_left_wrist_0_rgb"]
    print(f"两路相机 token 差异 mean|Δ| = {float(np.abs(a1 - b1).mean()):.3e}")
    print("SMOKE DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
