#!/usr/bin/env python3
"""Offline evaluation entry for train-RTC checkpoints (hard-clamp sampler).

Mirrors ``eval_offline_rtc.py`` but wraps the policy with
``wrap_policy_for_train_rtc`` instead of the inference-time guidance, so the
three methods keep separate entry points:

  inference-RTC  -> eval_offline_rtc.py --mode rtc
  train-RTC      -> eval_offline_train_rtc.py
  piR2           -> pir2_eval.py

Usage (repo root, GPU machine):
  uv run python eval_offline_train_rtc.py \
      --checkpoint <exp checkpoint dir> \
      --dataset ${OPENPI05_RAW_TRAIN_DIR:-<hdf5_dir>} \
      --inference-delay 5
"""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import sys
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
                    help="deployment d; must be <= training simulated_delay - 1")
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
    from openpi_rtc import load_norm_stats, wrap_policy_for_train_rtc

    print(f"Loading train-RTC checkpoint {args.checkpoint} "
          f"(inference_delay={args.inference_delay}) ...")
    cfg = config.get_config(args.config)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    norm_stats = load_norm_stats(args.checkpoint, cfg)
    policy = wrap_policy_for_train_rtc(policy, args.inference_delay, norm_stats=norm_stats)

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
        mode="train_rtc",
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
        stats = ev.evaluate_file(f, policy, eval_args)
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
        "mode": "train_rtc",
        "inference_delay": args.inference_delay,
        "mse": float(np.mean(total["mse"])),
        "l1": float(np.mean(total["l1"])),
        "boundary_mse": float(np.mean(total["boundary_mse"]))
        if total["boundary_mse"] else None,
        "boundary_l1": float(np.mean(total["boundary_l1"]))
        if total["boundary_l1"] else None,
        "mean_infer_ms": float(np.mean(total["infer_ms"])),
        "files": len(total["mse"]),
    }
    print("\n===== TRAIN-RTC EVALUATION =====")
    for k, v in final.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.join(args.log_dir, "results"), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        args.log_dir, "results",
        f"{args.config}_{pathlib.Path(args.checkpoint).name}_train_rtc_{stamp}.txt",
    )
    with open(results_file, "w") as fh:
        for k, v in final.items():
            fh.write(f"{k}: {v}\n")
    print(f"Results written to {results_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
