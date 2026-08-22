#!/usr/bin/env python3
"""One-shot probe: is this JAX checkpoint ready for RTC offline eval?

Run it on the GPU machine inside the openpi fork repo root:

  uv run python openpi_rtc/probe_checkpoint.py \
      --checkpoint /abs/path/to/49999 \
      --dataset /abs/path/to/hdf5_dir \
      --prompt "Transfer the test tube from the right rack to the left rack."

Checks, in order:
  1. python/jax versions and accelerator availability;
  2. config registration;
  3. checkpoint loads via ``policy_config.create_trained_policy`` and the
     model backend is JAX (as required by ``openpi_rtc``);
  4. a real first-frame infer from the HDF5 dataset succeeds (baseline);
  5. an RTC-guided infer succeeds (wrap + guidance path), with latency.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="pi05-task_00031_yulong-xtrainer")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--inference_delay", type=int, default=4)
    ap.add_argument("--execution_horizon", type=int, default=10)
    args = ap.parse_args()

    # Fail fast on missing/misconfigured data paths before any heavy work.
    from openpi_rtc.paths import require_checkpoint, require_dataset

    require_checkpoint(args.checkpoint)
    if args.dataset:
        require_dataset(args.dataset, "dataset")

    ok = True

    def report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        status = "OK  " if passed else "FAIL"
        print(f"[{status}] {name} {detail}")
        if not passed:
            ok = False

    # ---------------- 0. environment ----------------
    import platform

    print(f"host={platform.node()} python={platform.python_version()}")
    try:
        import jax

        print(f"jax={jax.__version__} devices={jax.devices()}")
        if jax.devices()[0].platform == "cpu":
            print("  WARNING: running on CPU; RTC latency numbers are meaningless here.")
    except Exception as e:  # noqa: BLE001
        report("import jax", False, repr(e))

    # ---------------- 1. config registration ----------------
    try:
        from openpi.training import config

        cfg = config.get_config(args.config)
        report("config registered", True, args.config)
        print(f"  action_dim={cfg.model.action_dim} action_horizon={cfg.model.action_horizon} "
              f"pi05={cfg.model.pi05}")
    except Exception as e:  # noqa: BLE001
        report("config registered", False, repr(e))
        return 1

    # ---------------- 2. checkpoint loading + backend ----------------
    try:
        from openpi.policies import policy_config

        t0 = time.time()
        policy = policy_config.create_trained_policy(cfg, args.checkpoint)
        print(f"  policy loaded in {time.time() - t0:.1f}s")
        model = getattr(policy, "_model", None)
        if model is None:
            report("policy exposes _model", False, "no _model attribute")
        else:
            mod = type(model).__module__
            print(f"  model type: {mod}.{type(model).__name__}")
            is_jax = "models" in mod and "pytorch" not in mod
            report("jax backend (required for openpi_rtc)", is_jax)
            if is_jax:
                try:
                    leaves = jax.tree.leaves(
                        {k: v for k, v in model.__dict__.items() if k in ("params",)}
                    )
                except Exception:  # noqa: BLE001
                    leaves = []
                if leaves:
                    n = sum(int(l.size) for l in leaves if hasattr(l, "size"))
                    print(f"  approximate param count: {n / 1e9:.2f}B")
    except Exception as e:  # noqa: BLE001
        report("load policy", False, repr(e))
        return 1

    # ---------------- 3. real-observation infer smoke ----------------
    if args.dataset:
        try:
            import cv2
            import h5py
            import numpy as np

            from openpi_rtc import load_norm_stats, wrap_policy_for_rtc
            from openpi_rtc.rtc_config import RTCConfig

            files = sorted(Path(args.dataset).rglob("*.hdf5"))
            if not files:
                report("dataset hdf5 found", False, f"no .hdf5 under {args.dataset}")
                return 1
            report("dataset hdf5 found", True, str(files[0]))
            with h5py.File(files[0], "r") as f:
                n = len(f["/observations/qpos"])
                i = min(args.step, n - 1)
                qpos = f["/observations/qpos"][i]
                imgs = {}
                for hk, pk in [("top", "cam_high"),
                               ("left_wrist", "cam_left_wrist"),
                               ("right_wrist", "cam_right_wrist")]:
                    if f"/observations/images/{hk}" not in f:
                        report(f"hdf5 key /observations/images/{hk}", False, "missing")
                        return 1
                    data = np.asarray(f[f"/observations/images/{hk}"][i], dtype="uint8")
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if img is None:
                        report(f"imdecode {hk}", False, "decoded to None")
                        return 1
                    imgs[pk] = img.swapaxes(0, 2).swapaxes(1, 2)
            obs = {
                "state": qpos,
                "images": imgs,
                "prompt": args.prompt or "Transfer the test tube from the right rack to the left rack.",
            }

            t0 = time.time()
            res = policy.infer(obs)
            dt0 = (time.time() - t0) * 1000.0
            act = np.asarray(res["actions"])
            print(f"  baseline infer: {dt0:.0f} ms, actions shape={act.shape}, "
                  f"abs mean={np.abs(act).mean():.4f}")
            report("baseline infer", True)

            # ---------------- 4. RTC-guided infer ----------------
            try:
                norm_stats = load_norm_stats(args.checkpoint, cfg)
                pol_rtc = wrap_policy_for_rtc(
                    policy,
                    RTCConfig(enabled=True,
                              execution_horizon=args.execution_horizon,
                              max_guidance_weight=10.0,
                              prefix_attention_schedule="exp"),
                    norm_stats=norm_stats,
                )
                t0 = time.time()
                _ = pol_rtc.infer(obs)
                dt1 = (time.time() - t0) * 1000.0
                prev = pol_rtc.last_raw_chunk
                prev_state = np.asarray(obs["state"], dtype=np.float32)
                prepared = pol_rtc.prepare_prev_chunk(prev, prev_state, prev_state)
                t0 = time.time()
                _ = pol_rtc.infer(obs, prev_chunk_left_over=prepared[args.inference_delay:],
                                  inference_delay=args.inference_delay,
                                  execution_horizon=args.execution_horizon)
                dt2 = (time.time() - t0) * 1000.0
                print(f"  rtc infer (no prev): {dt1:.0f} ms | "
                      f"rtc infer (guidance): {dt2:.0f} ms")
                report("rtc guidance infer", True)
            except Exception as e:  # noqa: BLE001
                report("rtc guidance infer", False, repr(e))
        except Exception as e:  # noqa: BLE001
            report("smoke infer", False, repr(e))
            return 1
    else:
        print("(no --dataset given; skipped infer smoke test)")

    print("\n===== PROBE SUMMARY =====")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
