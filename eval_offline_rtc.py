#!/usr/bin/env python3
"""Offline evaluation of openpi JAX pi0/pi0.5 policies with RTC comparison.

Replays the XTrainer HDF5 episodes (same observation builder as
``scripts/eval_online.py``) and reports, per mode:

- action accuracy (MSE/L1 of the predicted chunk vs ground truth, robot units);
- chunk-boundary continuity (how continuous consecutive chunks are), measured
  in *absolute* action space: ``|new_chunk[:d] - prev_chunk[s:s+d]|``;
- per-inference latency (``infer_ms``), the basis for the real-robot delay ``d``.

RTC protocol: inference every ``stride == d`` steps (the paper's async
protocol), feeding the previous raw chunk (re-anchored to the current
observation state) as ``prev_chunk_left_over``.

Usage:
  uv run python openpi_rtc/eval_offline_rtc.py --mode baseline \
      --checkpoint <49999 dir> --dataset <hdf5_dir> \
      --prompt "Transfer the test tube from the right rack to the left rack."
  uv run python openpi_rtc/eval_offline_rtc.py --mode rtc --inference_delay 4 ...
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openpi_rtc import load_norm_stats, wrap_policy_for_rtc  # noqa: E402
from openpi_rtc.rtc_config import RTCConfig  # noqa: E402


DEFAULT_PROMPT = "Transfer the test tube from the right rack to the left rack."


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="pi05-task_00031_yulong-xtrainer")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="eval_logs")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "rtc"])
    # RTC knobs.
    parser.add_argument("--inference_delay", type=int, default=7,
                        help="d: in-flight prefix steps (25Hz ticks; 7 = 280ms)")
    parser.add_argument("--execution_horizon", type=int, default=10)
    parser.add_argument("--max_guidance_weight", type=float, default=10.0)
    parser.add_argument("--schedule", type=str, default="exp",
                        choices=["exp", "linear", "ones", "zeros"])
    parser.add_argument("--stride", type=int, default=None,
                        help="steps between inference calls; default: d (rtc) / 1 (baseline)")
    parser.add_argument("--anchor_correction", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="re-anchor prev raw chunk to the current state (delta policies)")
    return parser.parse_args()


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"eval_rtc_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_file


def build_observation(root, i, prompt):
    qpos = root["/observations/qpos"][i]
    img_top = cv2.imdecode(np.asarray(root["/observations/images/top"][i], dtype="uint8"),
                           cv2.IMREAD_COLOR)
    img_left = cv2.imdecode(np.asarray(root["/observations/images/left_wrist"][i],
                                       dtype="uint8"), cv2.IMREAD_COLOR)
    img_right = cv2.imdecode(np.asarray(root["/observations/images/right_wrist"][i],
                                        dtype="uint8"), cv2.IMREAD_COLOR)
    return {
        "state": qpos,
        "images": {
            "cam_high": img_top.swapaxes(0, 2).swapaxes(1, 2),
            "cam_left_wrist": img_left.swapaxes(0, 2).swapaxes(1, 2),
            "cam_right_wrist": img_right.swapaxes(0, 2).swapaxes(1, 2),
        },
        "prompt": prompt,
    }


def evaluate_file(file_path, policy, args):
    logging.info(f"Evaluating file: {file_path} (mode={args.mode})")
    mse_list, l1_list = [], []
    boundary_mse_list, boundary_l1_list = [], []
    infer_ms_list = []
    steps = 0
    d = args.inference_delay
    s = args.stride
    prev_out = None       # previous chunk, robot units (H, A)
    prev_raw = None       # previous chunk, model space (H, A)
    prev_state = None     # observation state that generated prev_raw

    try:
        with h5py.File(file_path, "r", rdcc_nbytes=1024**2 * 2) as root:
            data_len = len(root["/observations/qpos"]) if "/observations/qpos" in root \
                else len(root["action"])
            if args.max_steps:
                data_len = min(data_len, args.max_steps)

            for i in range(0, data_len, s):
                try:
                    observation = build_observation(root, i, args.prompt)
                except KeyError as e:
                    logging.warning(f"  Error reading obs at step {i}: {e}")
                    continue
                cur_state = np.asarray(observation["state"], dtype=np.float32)

                t0 = time.perf_counter()
                if args.mode == "rtc" and prev_raw is not None:
                    prepared = policy.prepare_prev_chunk(prev_raw, prev_state, cur_state)
                    result = policy.infer(
                        observation,
                        prev_chunk_left_over=prepared[s:],
                        inference_delay=d,
                        execution_horizon=args.execution_horizon,
                    )
                else:
                    result = policy.infer(observation)
                infer_ms = (time.perf_counter() - t0) * 1000.0
                infer_ms_list.append(infer_ms)

                pred_actions = result.get("actions")
                if pred_actions is None:
                    logging.warning(f"  Unexpected result at step {i}: {result.keys()}")
                    continue
                pred_actions = np.asarray(pred_actions)
                if pred_actions.ndim == 1:
                    pred_actions = pred_actions[np.newaxis, :]

                chunk_size = pred_actions.shape[0]
                total_actions = root["action"].shape[0]
                end_idx = min(i + chunk_size, total_actions)
                gt_chunk = root["action"][i:end_idx]
                valid_len = gt_chunk.shape[0]
                pred_chunk = pred_actions[:valid_len]
                if pred_chunk.shape[1] != gt_chunk.shape[1]:
                    min_dim = min(pred_chunk.shape[1], gt_chunk.shape[1])
                    pred_chunk = pred_chunk[:, :min_dim]
                    gt_chunk = gt_chunk[:, :min_dim]

                mse_list.append(np.mean((pred_chunk - gt_chunk) ** 2))
                l1_list.append(np.mean(np.abs(pred_chunk - gt_chunk)))

                # Boundary continuity in absolute action space.
                if prev_out is not None and s + d <= len(prev_out):
                    new_front = pred_actions[:d]
                    prev_window = prev_out[s : s + d]
                    if new_front.shape == prev_window.shape:
                        boundary_mse_list.append(np.mean((new_front - prev_window) ** 2))
                        boundary_l1_list.append(np.mean(np.abs(new_front - prev_window)))

                prev_out = pred_actions
                if args.mode == "rtc":
                    raw = policy.last_raw_chunk
                    if raw is not None:
                        prev_raw = np.asarray(raw)
                        prev_state = cur_state
                steps += 1

                if i % 50 == 0:
                    logging.info(f"  Step {i}/{data_len}: MSE={mse_list[-1]:.6f} "
                                 f"L1={l1_list[-1]:.6f} infer_ms={infer_ms:.1f}")
    except Exception as e:  # noqa: BLE001
        logging.error(f"Failed to process {file_path}: {e}")
        return None

    if steps == 0:
        logging.warning(f"No steps evaluated for {file_path}")
        return None

    stats = {
        "mse": float(np.mean(mse_list)),
        "l1": float(np.mean(l1_list)),
        "steps": steps,
        "mse_list": mse_list,
        "l1_list": l1_list,
        "mean_infer_ms": float(np.mean(infer_ms_list)),
    }
    if boundary_mse_list:
        stats["boundary_mse"] = float(np.mean(boundary_mse_list))
        stats["boundary_l1"] = float(np.mean(boundary_l1_list))
        stats["boundary_mse_list"] = boundary_mse_list
        stats["boundary_l1_list"] = boundary_l1_list
    logging.info(f"  File complete. MSE={stats['mse']:.6f} L1={stats['l1']:.6f} "
                 f"boundary_mse={stats.get('boundary_mse')} "
                 f"mean_infer_ms={stats['mean_infer_ms']:.1f}")
    return stats


def main():
    args = parse_args()

    # Fail fast on missing/misconfigured data paths before model load.
    from openpi_rtc.paths import require_checkpoint, require_dataset

    require_checkpoint(args.checkpoint)
    require_dataset(args.dataset, "dataset")

    if args.mode == "rtc":
        if args.stride is None:
            args.stride = args.inference_delay
        if args.stride != args.inference_delay:
            raise SystemExit(
                "RTC protocol requires --stride == --inference_delay "
                f"(got stride={args.stride}, d={args.inference_delay})."
            )
    else:
        args.stride = 1

    from openpi.policies import policy_config
    from openpi.training import config

    tag = f"{args.config}_{Path(args.checkpoint).name}_{args.mode}"
    results_dir = os.path.join(args.log_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    log_file = setup_logging(args.log_dir)
    logging.info(f"mode={args.mode} config={args.config} checkpoint={args.checkpoint}")
    logging.info(f"dataset={args.dataset} prompt={args.prompt!r} stride={args.stride} "
                 f"inference_delay={args.inference_delay} horizon={args.execution_horizon} "
                 f"kappa={args.max_guidance_weight} schedule={args.schedule}")

    print("Loading policy...")
    cfg = config.get_config(args.config)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    if args.mode == "rtc":
        norm_stats = load_norm_stats(args.checkpoint, cfg)
        policy = wrap_policy_for_rtc(
            policy,
            RTCConfig(enabled=True,
                      execution_horizon=args.execution_horizon,
                      max_guidance_weight=args.max_guidance_weight,
                      prefix_attention_schedule=args.schedule,
                      anchor_correction=args.anchor_correction),
            norm_stats=norm_stats,
        )

    files = []
    if os.path.isdir(args.dataset):
        for root, _, filenames in os.walk(args.dataset):
            files += [os.path.join(root, f) for f in filenames if f.endswith(".hdf5")]
    elif os.path.isfile(args.dataset):
        files.append(args.dataset)
    else:
        logging.error(f"Dataset path not found: {args.dataset}")
        return
    files.sort()
    logging.info(f"Found {len(files)} files.")

    total = {"mse": [], "l1": [], "steps_list": [], "mse_list": [], "l1_list": [],
             "boundary_mse": [], "boundary_l1": [], "boundary_mse_list": [],
             "boundary_l1_list": [], "infer_ms": []}
    for file_path in tqdm.tqdm(files, desc="Evaluating files"):
        stats = evaluate_file(file_path, policy, args)
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
        logging.warning("No valid results collected.")
        return

    final = {
        "mse": float(np.mean(total["mse"])),
        "l1": float(np.mean(total["l1"])),
        "boundary_mse": float(np.mean(total["boundary_mse"])) if total["boundary_mse"] else None,
        "boundary_l1": float(np.mean(total["boundary_l1"])) if total["boundary_l1"] else None,
        "mean_infer_ms": float(np.mean(total["infer_ms"])),
        "files": len(total["mse"]),
    }
    logging.info("=" * 50)
    logging.info("FINAL EVALUATION RESULTS")
    for k, v in final.items():
        logging.info(f"  {k}: {v}")
    logging.info("=" * 50)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"{tag}_{stamp}_results.txt")
    with open(results_file, "w") as f:
        f.write("FINAL EVALUATION RESULTS\n")
        f.write(f"Eval Config: {args.config}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Stride: {args.stride}, Inference Delay: {args.inference_delay}\n")
        for k, v in final.items():
            f.write(f"{k}: {v}\n")
    print(f"Final results written to {results_file}")

    intermediate_dir = os.path.join(args.log_dir, "intermediate_results")
    os.makedirs(intermediate_dir, exist_ok=True)
    npz_file = os.path.join(intermediate_dir, f"{tag}_{stamp}_intermediate_results.npz")
    np.savez_compressed(
        npz_file,
        mse_list=np.array(total["mse_list"]),
        l1_list=np.array(total["l1_list"]),
        steps_list=np.array(total["steps_list"]),
        boundary_mse_list=np.array(total["boundary_mse_list"]),
        boundary_l1_list=np.array(total["boundary_l1_list"]),
        infer_ms_list=np.array(total["infer_ms"]),
    )
    print(f"Intermediate results saved to {npz_file}")


if __name__ == "__main__":
    main()
