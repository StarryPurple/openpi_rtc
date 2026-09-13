#!/usr/bin/env python3
"""Offline evaluation entry for πR² checkpoints (fast proprio + staircase).

Mirrors ``eval_offline_train_rtc.py`` but wraps the policy with
``wrap_policy_for_pir2``. Separate entry point per method:

  inference-RTC  -> eval_offline_rtc.py --mode rtc
  train-RTC      -> eval_offline_train_rtc.py
  piR2           -> pir2_eval.py

Usage (repo root, GPU machine):
  uv run python pir2_eval.py \
      --checkpoint <exp checkpoint dir> \
      --dataset ${OPENPI05_RAW_TRAIN_DIR:-<hdf5_dir>} \
      --inference-delay 7 --num-steps 10
"""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import sys
import time
from argparse import Namespace

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import tqdm

from openpi_rtc import eval_offline_rtc as ev


DEFAULT_PROMPT = "Transfer the test tube from the right rack to the left rack."


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="pi05-task_00031_entong-xtrainer")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--inference-delay", type=int, default=7,
                    help="deployment d; must be <= training max_delay")
    ap.add_argument("--num-steps", type=int, default=10,
                    help="denoising steps (try 1 after training for the "
                         "paper's fast mode; 10 is the safe default)")
    ap.add_argument("--slow-channel", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="异步慢通道 + 单步流评估（每次调用一步 DiT；"
                         "与真机 --mode pir2 --slow-channel 对齐）")
    ap.add_argument("--slow-refresh-every", type=int, default=5,
                    help="慢通道前缀刷新间隔（tick）")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--log-dir", default="eval_logs")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # Fail fast on missing/misconfigured data paths before model load.
    from openpi_rtc.paths import require_checkpoint, require_dataset

    require_checkpoint(args.checkpoint)
    require_dataset(args.dataset, "dataset")

    from openpi.policies import policy_config
    from openpi.training import config
    from openpi_rtc import load_norm_stats, wrap_policy_for_pir2

    print(f"Loading πR² checkpoint {args.checkpoint} "
          f"(inference_delay={args.inference_delay}, num_steps={args.num_steps}) ...")
    cfg = config.get_config(args.config)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    norm_stats = load_norm_stats(args.checkpoint, cfg)
    policy = wrap_policy_for_pir2(
        policy,
        args.inference_delay,
        norm_stats=norm_stats,
        slow_channel=args.slow_channel,
        image_delay_max=max(0, args.slow_refresh_every),
        slow_refresh_every=args.slow_refresh_every,
        num_steps=args.num_steps,
    )

    files = []
    if os.path.isdir(args.dataset):
        for root, _, filenames in os.walk(args.dataset):
            files += [os.path.join(root, f) for f in filenames if f.endswith(".hdf5")]
    elif os.path.isfile(args.dataset):
        files.append(args.dataset)
    else:
        print(f"Dataset path not found: {args.dataset}")
        return 1
    files.sort()

    eval_args = Namespace(
        mode="pir2",
        config=args.config,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        prompt=args.prompt,
        inference_delay=args.inference_delay,
        stride=args.inference_delay,
        execution_horizon=10,
        max_guidance_weight=10.0,
        schedule="exp",
        max_steps=args.max_steps,
        anchor_correction=True,
        log_dir=args.log_dir,
    )

    total = {"mse": [], "l1": [], "steps_list": [], "mse_list": [], "l1_list": [],
             "boundary_mse": [], "boundary_l1": [], "boundary_mse_list": [],
             "boundary_l1_list": [], "infer_ms": []}
    for f in tqdm.tqdm(files, desc="Evaluating files"):
        stats = (
            ev.evaluate_file(f, policy, eval_args)
            if not args.slow_channel
            else evaluate_file_stream(f, policy, eval_args)
        )
        if not stats:
            continue
        total["mse"].append(stats["mse"])
        total["l1"].append(stats["l1"])
        total["steps_list"].append(stats["steps"])
        total["mse_list"].extend(stats["mse_list"])
        total["l1_list"].extend(stats["l1_list"])
        total["infer_ms"].append(stats["mean_infer_ms"])
        if "boundary_mse" in stats:
            total["boundary_mse"].append(stats["boundary_mse"])
            total["boundary_l1"].append(stats["boundary_l1"])
            total["boundary_mse_list"].extend(stats["boundary_mse_list"])
            total["boundary_l1_list"].extend(stats["boundary_l1_list"])

    if not total["mse"]:
        print("No valid results collected.")
        return 1

    final = {
        "mode": "pir2",
        "inference_delay": args.inference_delay,
        "num_steps": args.num_steps,
        "mse": float(np.mean(total["mse"])),
        "l1": float(np.mean(total["l1"])),
        "boundary_mse": float(np.mean(total["boundary_mse"]))
        if total["boundary_mse"] else None,
        "boundary_l1": float(np.mean(total["boundary_l1"]))
        if total["boundary_l1"] else None,
        "mean_infer_ms": float(np.mean(total["infer_ms"])),
        "files": len(total["mse"]),
    }
    print("\n===== πR² EVALUATION =====")
    for k, v in final.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.join(args.log_dir, "results"), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        args.log_dir, "results",
        f"{args.config}_{pathlib.Path(args.checkpoint).name}_pir2_{stamp}.txt",
    )
    with open(results_file, "w") as fh:
        for k, v in final.items():
            fh.write(f"{k}: {v}\n")
    print(f"Results written to {results_file}")
    return 0


def evaluate_file_stream(file_path, policy, args):
    """单步流离线评估：每 d 帧调用一次 ``infer_stream``（warm 首帧），
    只把本次新发射的前 d 个动作与 GT 对比（对齐真机语义）。"""
    import h5py

    from openpi_rtc import eval_offline_rtc as _ev

    mse_list, l1_list, infer_ms_list = [], [], []
    d = int(args.inference_delay)
    steps = 0
    warm = True
    try:
        with h5py.File(file_path, "r", rdcc_nbytes=1024 ** 2 * 2) as root:
            data_len = (
                len(root["/observations/qpos"])
                if "/observations/qpos" in root
                else len(root["action"])
            )
            if args.max_steps:
                data_len = min(data_len, args.max_steps)
            for i in range(0, data_len, d):
                try:
                    observation = _ev.build_observation(root, i, args.prompt)
                except KeyError as e:
                    continue
                t0 = time.perf_counter()
                result = policy.infer_stream(
                    observation, inference_delay=d, warm=warm
                )
                warm = False
                infer_ms = (time.perf_counter() - t0) * 1000.0
                if not result.get("refreshed"):
                    infer_ms_list.append(infer_ms)  # 排除前缀刷新调用
                pred = np.asarray(result["actions"], dtype=np.float32)
                gt = np.asarray(root["action"][i : i + d], dtype=np.float32)
                n = min(len(pred), len(gt), d)
                if n <= 0:
                    continue
                pred_c, gt_c = pred[:n, : gt.shape[1]], gt[:n]
                mse_list.append(float(np.mean((pred_c - gt_c) ** 2)))
                l1_list.append(float(np.mean(np.abs(pred_c - gt_c))))
                steps += 1
    except Exception:
        import traceback

        traceback.print_exc()
        return None
    if not steps:
        return None
    return {
        "mse": float(np.mean(mse_list)),
        "l1": float(np.mean(l1_list)),
        "steps": steps,
        "mse_list": mse_list,
        "l1_list": l1_list,
        "mean_infer_ms": float(np.mean(infer_ms_list)),
    }


if __name__ == "__main__":
    sys.exit(main())
