#!/usr/bin/env python3
"""Async real-robot loop for the JAX pi0/pi0.5 policy (baseline / inference-RTC).

Run the same episodes with ``--mode baseline`` and ``--mode rtc`` on the same
checkpoint. The robot is provided by an adapter implementing
``get_observation`` / ``execute_action`` / ``episode_done`` /
``reset_episode`` — see ``openpi_rtc.robot_xtrainer.XtrainerRobot``.

Usage:
  uv run python run_robot.py --mode rtc \
      --checkpoint <model dir> --robot openpi_rtc.robot_xtrainer:XtrainerRobot \
      --episodes 10
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openpi_rtc import ActionQueue, load_norm_stats, wrap_policy_for_rtc  # noqa: E402
from openpi_rtc.rtc_config import RTCConfig  # noqa: E402


DEFAULT_PROMPT = "Transfer the test tube from the right rack to the left rack."


def build_policy(args):
    from openpi.policies import policy_config
    from openpi.training import config

    train_cfg = config.get_config(args.config)
    policy = policy_config.create_trained_policy(train_cfg, args.checkpoint)
    if args.mode == "rtc":
        norm_stats = load_norm_stats(args.checkpoint, train_cfg)
        policy = wrap_policy_for_rtc(
            policy,
            RTCConfig(enabled=True,
                      execution_horizon=args.execution_horizon,
                      max_guidance_weight=args.max_guidance_weight,
                      prefix_attention_schedule=args.schedule,
                      anchor_correction=args.anchor_correction),
            norm_stats=norm_stats,
        )
    return policy


class EpisodeRunner:
    """Minimal async executor. Pass a ``robot`` object with the four callbacks
    (get_observation / execute_action / episode_done / reset_episode), e.g.
    ``openpi_rtc.robot_xtrainer.XtrainerRobot``."""

    def __init__(self, policy, *, control_period_s: float = 1 / 25.0,
                 rtc_enabled: bool, delay_est_window: int = 20,
                 robot=None):
        self.policy = policy
        self.period = control_period_s
        self.rtc_enabled = rtc_enabled
        self.rtc_cfg = RTCConfig(enabled=rtc_enabled)
        self.robot = robot
        self.queue = ActionQueue(self.rtc_cfg)
        self._latency_s = collections.deque(maxlen=delay_est_window)
        self._stop = threading.Event()
        self._prev_state = None

    # ---------------- platform callbacks (delegate to robot if given) ------
    def get_observation(self) -> dict:
        """Return the openpi observation dict (state, images, prompt)."""
        if self.robot is not None:
            return self.robot.get_observation()
        raise NotImplementedError("wire to your env/robot observation source")

    def execute_action(self, action_np: np.ndarray) -> None:
        """Send one action (action_dim,) to the robot."""
        if self.robot is not None:
            self.robot.execute_action(action_np)
            return
        raise NotImplementedError("wire to your robot action API")

    def episode_done(self) -> bool:
        if self.robot is not None:
            return self.robot.episode_done()
        raise NotImplementedError("return True when the episode ends")

    def reset_episode(self) -> None:
        if self.robot is not None:
            self.robot.reset_episode()
            return
        raise NotImplementedError("reset the env / home the robot")

    # ----------------------------------------------------------

    def _estimate_delay_ticks(self) -> int:
        if not self._latency_s:
            return 7  # conservative initial guess (280ms @25Hz)
        return max(1, math.ceil(float(np.mean(self._latency_s)) / self.period))

    def _inference_worker(self):
        while not self._stop.is_set():
            if self.queue.qsize() > 10:  # keep the buffer bounded
                time.sleep(self.period)
                continue
            obs = self.get_observation()
            cur_state = np.asarray(obs["state"], dtype=np.float32)
            t0 = time.perf_counter()
            if self.rtc_enabled:
                prev_raw = self.policy.last_raw_chunk
                prev = None
                if prev_raw is not None and self._prev_state is not None:
                    prev = self.policy.prepare_prev_chunk(
                        prev_raw, self._prev_state, cur_state
                    )
                d = self._estimate_delay_ticks()
                out = self.policy.infer(
                    obs,
                    prev_chunk_left_over=prev[d:] if prev is not None else None,
                    inference_delay=d,
                    execution_horizon=self.rtc_cfg.execution_horizon,
                )
            else:
                out = self.policy.infer(obs)
            elapsed = time.perf_counter() - t0
            self._latency_s.append(elapsed)

            actions = np.asarray(out["actions"], dtype=np.float32)
            processed = actions
            raw = (
                np.asarray(self.policy.last_raw_chunk, dtype=np.float32)
                if self.rtc_enabled
                else actions
            )
            d = self._estimate_delay_ticks()
            self.queue.merge(raw, processed, d)
            if self.rtc_enabled:
                self._prev_state = cur_state

    def run_episode(self) -> dict:
        self.reset_episode()
        self.queue.clear()
        self._latency_s.clear()
        self._prev_state = None
        worker = threading.Thread(target=self._inference_worker, daemon=True)
        worker.start()
        actions_sent = 0
        infer_latencies = []
        try:
            while not self.episode_done():
                t0 = time.perf_counter()
                action = self.queue.get()
                if action is not None:
                    self.execute_action(action)
                    actions_sent += 1
                else:
                    time.sleep(self.period / 4)  # inference still warming up
                infer_latencies.extend(list(self._latency_s))
                rem = self.period - (time.perf_counter() - t0)
                if rem > 0:
                    time.sleep(rem)
        finally:
            self._stop.set()
            worker.join(timeout=5)
        return {
            "actions_sent": actions_sent,
            "mean_infer_ms": 1000 * float(np.mean(infer_latencies)) if infer_latencies else None,
            "est_delay_ticks": self._estimate_delay_ticks(),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["baseline", "rtc"], default="rtc")
    ap.add_argument("--config", default="pi05-task_00031_entong-xtrainer")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--execution_horizon", type=int, default=10)
    ap.add_argument("--max_guidance_weight", type=float, default=10.0)
    ap.add_argument("--schedule", type=str, default="exp",
                    choices=["exp", "linear", "ones", "zeros"])
    ap.add_argument("--anchor_correction", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--robot",
        default=None,
        help="import path of a robot adapter, e.g. "
             "openpi_rtc.robot_xtrainer:XtrainerRobot",
    )
    ap.add_argument(
        "--robot-type",
        default="Nova 2",
        choices=["Nova 2", "Nova 5"],
        help="robot model for FK pose protection (default: Nova 2)",
    )
    args = ap.parse_args()

    # Fail fast on a missing/misconfigured checkpoint before loading jax/openpi.
    from openpi_rtc.paths import require_checkpoint

    require_checkpoint(args.checkpoint)

    policy = build_policy(args)
    robot = None
    if args.robot:
        module_path, _, class_name = args.robot.partition(":")
        import importlib

        robot = getattr(importlib.import_module(module_path), class_name)(
            robot_type=args.robot_type
        )
    runner = EpisodeRunner(policy, rtc_enabled=(args.mode == "rtc"), robot=robot)
    print(f"mode={args.mode} checkpoint={args.checkpoint}")
    for ep in range(args.episodes):
        stats = runner.run_episode()
        print(f"episode {ep}: {stats}")


if __name__ == "__main__":
    main()
