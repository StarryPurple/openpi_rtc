#!/usr/bin/env python3
"""Offline sweep over the RTC inference delay ``d`` (GPU machine).

Loads the policy once, wraps it with RTC guidance, and replays the HDF5
episodes for every ``d`` in [--d-min, --d-max]. Reports per-d boundary
continuity / action accuracy / mean infer_ms. This does NOT measure the real
deployment latency — it validates which ``d`` gives the cleanest chunk
continuity given the data, and is cross-checked against the measured latency
(see ``measure_latency.py``).

Usage (repo root, GPU machine):
  uv run python openpi_rtc/sweep_d.py \
      --checkpoint <49999 dir> \
      --dataset ${OPENPI05_RAW_TRAIN_DIR:-<hdf5_dir>} \
      --d-min 2 --d-max 6 --max-steps 200
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from argparse import Namespace

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import tqdm

from openpi_rtc import eval_offline_rtc as ev


DEFAULT_PROMPT = "Transfer the test tube from the right rack to the left rack."


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="pi05-task_00031_yulong-xtrainer")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--d-min", type=int, default=2)
    ap.add_argument("--d-max", type=int, default=8)
    ap.add_argument("--execution-horizon", type=int, default=10)
    ap.add_argument("--max-guidance-weight", type=float, default=10.0)
    ap.add_argument("--schedule", default="exp",
                    choices=["exp", "linear", "ones", "zeros"])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--log-dir", default="eval_logs")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # Fail fast on missing/misconfigured data paths before model load.
    from openpi_rtc.paths import require_checkpoint, require_dataset

    require_checkpoint(args.checkpoint)
    require_dataset(args.dataset, "dataset")

    if args.d_min < 1:
        raise ValueError("--d-min must be >= 1")
    if args.d_min > args.d_max:
        raise ValueError("--d-min must be <= --d-max")

    from openpi.policies import policy_config
    from openpi.training import config
    from openpi_rtc import load_norm_stats, wrap_policy_for_rtc
    from openpi_rtc.rtc_config import RTCConfig

    print(f"Loading checkpoint {args.checkpoint} ...")
    cfg = config.get_config(args.config)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    norm_stats = load_norm_stats(args.checkpoint, cfg)
    policy = wrap_policy_for_rtc(
        policy,
        RTCConfig(enabled=True,
                  execution_horizon=args.execution_horizon,
                  max_guidance_weight=args.max_guidance_weight,
                  prefix_attention_schedule=args.schedule,
                  anchor_correction=True),
        norm_stats=norm_stats,
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
    print(f"Found {len(files)} HDF5 files; sweeping d in "
          f"[{args.d_min}, {args.d_max}] ...")

    rows = []
    for d in range(args.d_min, args.d_max + 1):
        eval_args = Namespace(
            mode="rtc",
            config=args.config,
            checkpoint=args.checkpoint,
            dataset=args.dataset,
            prompt=args.prompt,
            inference_delay=d,
            stride=d,
            execution_horizon=args.execution_horizon,
            max_guidance_weight=args.max_guidance_weight,
            schedule=args.schedule,
            max_steps=args.max_steps,
            anchor_correction=True,
            log_dir=args.log_dir,
        )
        per_file = []
        for f in tqdm.tqdm(files, desc=f"d={d}"):
            stats = ev.evaluate_file(f, policy, eval_args)
            if stats:
                per_file.append(stats)
        if not per_file:
            print(f"d={d}: no valid results")
            continue
        row = {
            "d": d,
            "boundary_mse": float(np.mean([s["boundary_mse"] for s in per_file
                                           if "boundary_mse" in s])),
            "mse": float(np.mean([s["mse"] for s in per_file])),
            "mean_infer_ms": float(np.mean([s["mean_infer_ms"] for s in per_file])),
            "steps": int(np.sum([s["steps"] for s in per_file])),
        }
        rows.append(row)
        print(f"  d={d}: boundary_mse={row['boundary_mse']:.6f} "
              f"mse={row['mse']:.6f} infer_ms={row['mean_infer_ms']:.1f}")

    if not rows:
        print("No rows produced.")
        return 1

    print("\n===== d SWEEP SUMMARY =====")
    print(f"{'d':>2}  {'boundary_mse':>13}  {'action_mse':>11}  {'infer_ms':>9}  {'steps':>7}")
    for r in rows:
        print(f"{r['d']:>2}  {r['boundary_mse']:>13.6f}  {r['mse']:>11.6f}  "
              f"{r['mean_infer_ms']:>9.1f}  {r['steps']:>7d}")
    best = min(rows, key=lambda r: r["boundary_mse"])
    print(f"\nLowest boundary_mse at d={best['d']} "
          f"({best['boundary_mse']:.6f}). Pair with the measured latency "
          f"(d_measured = ceil(end_to_end_ms / 40ms)) — prefer the larger of "
          f"d_measured+1 and the sweep's plateau.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
