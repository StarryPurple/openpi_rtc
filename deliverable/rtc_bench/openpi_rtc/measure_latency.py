#!/usr/bin/env python3
"""End-to-end inference latency + d recommendation (target machine).

Runs the real observation path (or an HDF5 stand-in) on the machine that will
drive the robot and reports steady-state latency, then recommends the
train-RTC / inference-RTC delay ``d`` for a given control frequency.

Usage (repo root):
  # model-only + HDF5 decode path (any machine with the checkpoint):
  uv run python measure_latency.py \
      --checkpoint <49999 dir> \
      --hdf5 ${OPENPI05_RAW_TRAIN_DIR:-<hdf5_dir>}

  # real robot path: supply a live observation source and run without --hdf5
  uv run python measure_latency.py --checkpoint <dir>
"""

from __future__ import annotations

import argparse
import math
import pathlib
import statistics
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from openpi_rtc import load_norm_stats, wrap_policy_for_rtc
from openpi_rtc.eval_offline_rtc import build_observation
from openpi_rtc.rtc_config import RTCConfig


DEFAULT_PROMPT = "Transfer the test tube from the right rack to the left rack."


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="pi05-task_00031_entong-xtrainer")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--hdf5", default=None,
                    help="HDF5 file or dir; if omitted the real-robot "
                         "get_observation() TODO must be wired")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--control-hz", type=float, default=25.0)
    ap.add_argument("--mode", default="baseline", choices=["baseline", "rtc"])
    ap.add_argument("--inference-delay", type=int, default=4)
    return ap.parse_args()


class Hdf5ObservationSource:
    """Reads observations from the raw HDF5 (decode cost included)."""

    def __init__(self, dataset: str):
        import h5py

        p = pathlib.Path(dataset)
        files = sorted(p.rglob("*.hdf5")) if p.is_dir() else [p]
        if not files:
            raise FileNotFoundError(f"no hdf5 under {dataset}")
        self._files = files
        self._idx = 0
        self._file = h5py.File(files[0], "r")
        self._len = len(self._file["/observations/qpos"])

    def __call__(self) -> dict:
        i = self._idx % self._len
        self._idx += 1
        if self._idx % self._len == 0 and self._idx > 0:
            self._file.close()
            fi = (self._idx // self._len) % len(self._files)
            self._file = h5py.File(self._files[fi], "r")
            self._len = len(self._file["/observations/qpos"])
        return build_observation(self._file, i, DEFAULT_PROMPT)


class RealObservationSource:
    """Wire to the robot/inference machine's real observation API."""

    def __call__(self) -> dict:
        # TODO: return the openpi observation dict: state (14,), images
        # {cam_high, cam_left_wrist, cam_right_wrist} as (3,480,640) uint8,
        # and the fixed prompt. E.g. from the XTrainer/Dobot SDK.
        raise NotImplementedError(
            "wire RealObservationSource.__call__ to the robot API, or use --hdf5"
        )


def main() -> int:
    args = parse_args()

    # Fail fast on missing/misconfigured data paths before model load.
    from openpi_rtc.paths import require_checkpoint, require_dataset

    require_checkpoint(args.checkpoint)
    if args.hdf5:
        require_dataset(args.hdf5, "hdf5")

    from openpi.policies import policy_config
    from openpi.training import config

    print(f"Loading checkpoint {args.checkpoint} ...")
    cfg = config.get_config(args.config)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    if args.mode == "rtc":
        norm_stats = load_norm_stats(args.checkpoint, cfg)
        policy = wrap_policy_for_rtc(
            policy,
            RTCConfig(enabled=True, execution_horizon=10,
                      max_guidance_weight=10.0, anchor_correction=True),
            norm_stats=norm_stats,
        )

    source = (
        Hdf5ObservationSource(args.hdf5)
        if args.hdf5
        else RealObservationSource()
    )

    print(f"Warming up ({args.warmup}) ...")
    for _ in range(args.warmup):
        obs = source()
        policy.infer(obs)

    latencies = []
    print(f"Measuring {args.repeats} steady-state infer calls ...")
    for _ in range(args.repeats):
        obs = source()
        t0 = time.perf_counter()
        policy.infer(obs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    ms = sorted(latencies)
    p95 = ms[min(len(ms) - 1, max(0, math.ceil(0.95 * len(ms)) - 1))]
    summary = {
        "mean_ms": statistics.fmean(ms),
        "median_ms": statistics.median(ms),
        "p95_ms": p95,
        "min_ms": ms[0],
        "max_ms": ms[-1],
    }
    print("\n===== LATENCY SUMMARY =====")
    for k, v in summary.items():
        print(f"  {k}: {v:.1f} ms")

    period_ms = 1000.0 / args.control_hz
    print(f"\nControl period @ {args.control_hz:.0f} Hz = {period_ms:.1f} ms")
    print("d = ceil(latency / period) with optional safety margin:")
    for margin in (0, 1, 2):
        d_med = math.ceil(summary["median_ms"] / period_ms) + margin
        d_p95 = math.ceil(p95 / period_ms) + margin
        print(f"  margin +{margin}: median -> d={d_med}, p95 -> d={d_p95}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
