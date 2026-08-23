#!/usr/bin/env python3
"""test_dobot_rtc_bench.py — 工控机 RTC 对比测试单文件入口。

放在 openpi-main/rtc_bench/ 下（openpi_rtc 模块同级），用 openpi-main 的
venv python（3.11）运行。脚本**复用 openpi-main 的 src/openpi**，不携带
openpi 副本；因为 openpi-main 的 config 里没有 yulong/light/entong 注册，
脚本会在运行时把这三个 TrainConfig 注入 openpi 的 config 注册表
（不改 openpi-main 任何文件）。

用法（在 openpi-main 根目录执行）:
    python rtc_bench/test_dobot_rtc_bench.py --mode probe
    python rtc_bench/test_dobot_rtc_bench.py --mode baseline --episodes 5
    python rtc_bench/test_dobot_rtc_bench.py --mode rtc --episodes 5
    python rtc_bench/test_dobot_rtc_bench.py --mode train_rtc --episodes 5   # 微调后
    python rtc_bench/test_dobot_rtc_bench.py --mode pir2 --episodes 5       # 微调后

输出视频: openpi-main/records/<模型名>/<mode>/episode_N/{cam_high,
cam_left_wrist,cam_right_wrist}.avi + episode_N.json
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import os
import pathlib
import statistics
import sys
import threading
import time

import numpy as np

# ---------------------------------------------------------------------------
# 0) 路径引导：本文件在 <openpi-main>/rtc_bench/ 内
#    - openpi / openpi_client 来自 openpi-main（editable 安装或 src 目录）
#    - openpi_rtc 模块来自本目录
# ---------------------------------------------------------------------------
BENCH_DIR = pathlib.Path(__file__).resolve().parent
OPENPI_MAIN = BENCH_DIR.parent
for _p in (OPENPI_MAIN, OPENPI_MAIN / "src", BENCH_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import openpi  # noqa: E402

if not pathlib.Path(openpi.__file__).resolve().is_relative_to(OPENPI_MAIN.resolve()):
    sys.exit(
        "ERROR: import openpi 命中 %s（应为 openpi-main/src/openpi/__init__.py）。"
        "请确认 rtc_bench 下没有残留的 openpi/ 目录（rm -rf rtc_bench/openpi）。"
        % openpi.__file__
    )

import cv2  # noqa: E402

from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as openpi_config  # noqa: E402

from openpi_rtc import (  # noqa: E402
    ActionQueue,
    load_norm_stats,
    wrap_policy_for_pir2,
    wrap_policy_for_rtc,
    wrap_policy_for_train_rtc,
)
from openpi_rtc.rtc_config import RTCConfig  # noqa: E402
from openpi_rtc.safety import SafetyConfig, check_action  # noqa: E402

PROMPT = "Transfer the test tube from the right rack to the left rack."
CONTROL_HZ = 25.0
PERIOD = 1.0 / CONTROL_HZ
TICK_MS = PERIOD * 1000.0

# 默认参考复位位姿（平台 robot_pose_init 同源；夹爪置开=1.0）。
# 每集（含第一集）开始前都回到这里，保证起点一致。
DEFAULT_RESET_POSE = np.concatenate([
    np.deg2rad([-90, 30, -110, 20, 90, 90]),
    [1.0],
    np.deg2rad([90, -30, 110, -20, -90, -90]),
    [1.0],
]).astype(np.float32)

# 模式 -> 模型映射（微调产物暂时占位；也可用 --checkpoint 覆盖）
MODELS: dict[str, dict] = {
    "baseline": {
        "checkpoint": OPENPI_MAIN
        / "checkpoints/dobot/pi05-task_00031_yulong-xtrainer/49999",
        "config": "pi05-task_00031_yulong-xtrainer",
        "wrapper": None,
    },
    "rtc": {
        "checkpoint": OPENPI_MAIN
        / "checkpoints/dobot/pi05-task_00031_yulong-xtrainer/49999",
        "config": "pi05-task_00031_yulong-xtrainer",
        "wrapper": "rtc",
    },
    "train_rtc": {
        "checkpoint": OPENPI_MAIN
        / "checkpoints/dobot/pi05-task_00031_entong-xtrainer/rtc_train_d7/49999",
        "config": "pi05-task_00031_entong-xtrainer",
        "wrapper": "train_rtc",
    },
    "pir2": {
        "checkpoint": OPENPI_MAIN
        / "checkpoints/dobot/pi05-task_00031_entong-xtrainer/pir2_v1/49999",
        "config": "pi05-task_00031_entong-xtrainer",
        "wrapper": "pir2",
    },
}


# openpi-main 的 config.py 缺少的 task 注册（运行时注入，不改其文件）
_EXTRA_TASKS = {
    "task_00031_yulong": ("task_00031_yulong_train", "task_00031_yulong_eval"),
    "task_00031_light": ("task_00031_light_train", "task_00031_light_eval"),
    "task_00031_entong": ("task_00031_entong_train", "task_00031_entong_eval"),
}


def ensure_task_configs() -> None:
    """把 yulong/light/entong 的 TrainConfig 注入 openpi 的 config 注册表。"""
    from openpi import transforms as _transforms
    from openpi.models import pi0_config as _pi0_config
    from openpi.training import config as _cfg
    from openpi.training import weight_loaders as _wl

    repack = _transforms.Group(inputs=[
        _transforms.RepackTransform({
            "images": {
                "cam_high": "observation.images.cam_high",
                "cam_left_wrist": "observation.images.cam_left_wrist",
                "cam_right_wrist": "observation.images.cam_right_wrist",
            },
            "state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        })
    ])
    for task, (train_repo, _eval_repo) in _EXTRA_TASKS.items():
        name = f"pi05-{task}-xtrainer"
        if name in _cfg._CONFIGS_DICT:
            continue
        cfg = _cfg.TrainConfig(
            name=name,
            model=_pi0_config.Pi0Config(pi05=True),
            data=_cfg.LeRobotAlohaDataConfig(
                repo_id=train_repo,
                adapt_to_pi=False,
                repack_transforms=repack,
                base_config=_cfg.DataConfig(prompt_from_task=True),
                assets=_cfg.AssetsConfig(assets_dir="./assets", asset_id=train_repo),
            ),
            batch_size=32,
            weight_loader=_wl.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=50000,
            save_interval=10000,
            keep_period=30000,
            fsdp_devices=1,
            num_workers=12,
        )
        _cfg._CONFIGS_DICT[name] = cfg
        _cfg._CONFIGS.append(cfg)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["probe", "baseline", "rtc", "train_rtc", "pir2"], required=True)
    ap.add_argument("--probe-target", choices=["baseline", "rtc", "train_rtc", "pir2"], default="rtc",
                    help="probe 模式要测的模型类型（默认 rtc）")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--checkpoint", default=None, help="覆盖 MODELS 里的 checkpoint 路径")
    ap.add_argument("--config", default=None, help="覆盖 MODELS 里的 config 名")
    ap.add_argument("--inference-delay", type=int, default=None,
                    help="d；默认 None=实机按实测延迟自动估计")
    ap.add_argument("--execution-horizon", type=int, default=10)
    ap.add_argument("--max-guidance-weight", type=float, default=10.0)
    ap.add_argument("--schedule", default="exp", choices=["exp", "linear", "ones", "zeros"])
    ap.add_argument("--num-steps", type=int, default=10, help="πR² 去噪步数（10 或 1 快模式）")
    ap.add_argument("--arms", default="right", choices=["left", "right", "both"])
    ap.add_argument("--robot-type", default="Nova 2", choices=["Nova 2", "Nova 5"])
    ap.add_argument("--episode-timeout-s", type=float, default=30.0)
    ap.add_argument("--min-episode-s", type=float, default=10.0,
                    help="自动结束的最短运行秒数（防一开始回位误判）")
    ap.add_argument("--home-threshold-rad", type=float, default=0.1,
                    help="qpos 距起始位姿的最大偏差（rad）；设 0 禁用自动结束")
    ap.add_argument("--home-hold-s", type=float, default=1.0,
                    help="回到起始位姿需持续秒数才算完成")
    ap.add_argument("--pose-check-rad", type=float, default=0.52,
                    help="首动作与当前位姿的最大偏差(rad)，默认 0.52≈30°（对齐旧 harness "
                         "pose_check）；超限中止；0 禁用")
    ap.add_argument("--interp-first", action=argparse.BooleanOptionalAction, default=True,
                    help="首动作在允许范围内时插值逐步逼近（默认开），避免单步大动")
    ap.add_argument("--joint-limit-rad", type=float, default=2.9,
                    help="关节角绝对值上限(rad)；超限中止（防 rad/deg 混用等垃圾指令）；0 禁用")
    ap.add_argument("--max-step-rad", type=float, default=0.35,
                    help="相邻动作单步最大偏差(rad)；超限中止（比 safety 的 0.9 更严）；0 禁用")
    ap.add_argument("--record-dir", default=None,
                    help="录像根目录（默认 <openpi-main>/records）")
    ap.add_argument("--hdf5", default=None, help="probe 模式用 hdf5 帧代替真机观测")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--safety-off", action="store_true")
    ap.add_argument("--auto-reset", action="store_true",
                    help="episode 之间自动回到 reset_position（默认手动复位）")
    ap.add_argument("--reset-pose", default="",
                    help="14 个逗号分隔的关节角(弧度, 夹爪 0~1)，覆盖默认参考位姿；"
                         "空=用平台复位位姿")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# 1) 策略构建
# ---------------------------------------------------------------------------
def build_policy(spec: dict, args: argparse.Namespace):
    ensure_task_configs()
    ckpt = pathlib.Path(args.checkpoint or spec["checkpoint"])
    if not (ckpt / "params").is_dir() or not (ckpt / "assets").is_dir():
        sys.exit(
            f"ERROR: checkpoint 不存在或缺少 params/assets: {ckpt}\n"
            f"  模式 {args.mode} 需要 {spec['config']} 的模型。"
            f"  baseline/rtc 用 yulong 49999；train_rtc/pir2 需微调产物，"
            f"  --checkpoint 可覆盖路径。"
        )
    cfg = openpi_config.get_config(args.config or spec["config"])
    policy = policy_config.create_trained_policy(cfg, str(ckpt))
    norm_stats = load_norm_stats(str(ckpt), cfg)
    wrapper = spec["wrapper"]
    if wrapper == "rtc":
        policy = wrap_policy_for_rtc(
            policy,
            RTCConfig(enabled=True,
                      execution_horizon=args.execution_horizon,
                      max_guidance_weight=args.max_guidance_weight,
                      prefix_attention_schedule=args.schedule,
                      anchor_correction=True),
            norm_stats=norm_stats,
        )
    elif wrapper == "train_rtc":
        policy = wrap_policy_for_train_rtc(policy, args.inference_delay or 7, norm_stats=norm_stats)
    elif wrapper == "pir2":
        policy = wrap_policy_for_pir2(policy, args.inference_delay or 7, norm_stats=norm_stats)
    return policy, cfg, ckpt


# ---------------------------------------------------------------------------
# 2) 机器人环境（复用 openpi-main 的 examples.xtrainer_real.RealEnv）
# ---------------------------------------------------------------------------
def make_env(args: argparse.Namespace):
    from examples.xtrainer_real.real_env import RealEnv

    reset_position = [-1.5707964, 0.5235988, -1.9198622, 0.34906584, 1.5707964, 1.5707964]
    return RealEnv(False, arms=args.arms, reset_position=reset_position)


def get_observation(env) -> dict:
    obs = env.get_observation()
    state = np.asarray(obs["qpos"], dtype=np.float32)
    images = {}
    for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        images[cam] = obs["images"][cam].swapaxes(0, 2).swapaxes(1, 2)
    return {"state": state, "images": images, "prompt": PROMPT}


def auto_reset(env, target: np.ndarray) -> None:
    """插值移动到目标位姿（默认用第一集开始时的位姿，保证与策略起点一致）。"""
    target = np.asarray(target, dtype=np.float32)
    curr = np.asarray(env.get_observation()["qpos"], dtype=np.float32)
    steps = min(int(np.abs(curr - target).max() / 0.01), 100)
    for jnt in np.linspace(curr, target, steps):
        env.step(jnt)  # RealEnv.step(action, single_arm=True)，不要传数组
        time.sleep(PERIOD)
    time.sleep(1.0)  # 等机械臂稳定，避免下一集首动作观测未稳定


# ---------------------------------------------------------------------------
# 3) 录像
# ---------------------------------------------------------------------------
class Recorder:
    def __init__(self, root: pathlib.Path, model_name: str, mode: str):
        self.ep_dir = root / model_name / mode
        self.writers: dict[str, cv2.VideoWriter] = {}
        self.shapes: dict[str, tuple[int, int]] = {}

    def ensure(self, cam: str, frame: np.ndarray) -> None:
        if cam in self.writers:
            return
        h, w = frame.shape[:2]
        self.ep_dir.mkdir(parents=True, exist_ok=True)
        self.writers[cam] = cv2.VideoWriter(
            str(self.ep_dir / f"{cam}.avi"), cv2.VideoWriter_fourcc(*"XVID"), CONTROL_HZ, (w, h)
        )

    def write(self, obs_images: dict) -> None:
        for cam, img in obs_images.items():
            self.ensure(cam, img)
            self.writers[cam].write(img)

    def close(self) -> None:
        for w in self.writers.values():
            w.release()
        self.writers = {}


# ---------------------------------------------------------------------------
# 4) 异步 episode 执行（RTC 协议：infer 与执行并行）
# ---------------------------------------------------------------------------
class BenchRunner:
    def __init__(self, policy, env, args: argparse.Namespace, model_name: str):
        self.policy = policy
        self.env = env
        self.args = args
        self.model_name = model_name
        self.rtc_enabled = args.mode != "baseline"
        self.queue = ActionQueue(RTCConfig(enabled=self.rtc_enabled))
        self._latency_ms = collections.deque(maxlen=20)
        self._stop = threading.Event()
        self._prev_state = None
        self._last_action = None
        self._latest_obs = None
        self.safety = SafetyConfig(enabled=not args.safety_off, robot_type=args.robot_type)

    def _delay_ticks(self) -> int:
        if not self._latency_ms:
            return 7
        # 上限 16 tick（640ms）：防止编译等异常样本把 d 估爆（正常预算 ≤8）
        return min(max(1, math.ceil(float(np.mean(self._latency_ms)) / TICK_MS)), 16)

    def _inference_worker(self) -> None:
        while not self._stop.is_set():
            if self.queue.qsize() > 10:
                time.sleep(PERIOD)
                continue
            # 用 executor 最新观测，避免两个线程并发读 RealSense
            obs = self._latest_obs if self._latest_obs is not None else get_observation(self.env)
            cur_state = np.asarray(obs["state"], dtype=np.float32)
            t0 = time.perf_counter()
            if self.rtc_enabled:
                prev_raw = getattr(self.policy, "last_raw_chunk", None)
                prev = None
                if prev_raw is not None and self._prev_state is not None:
                    prev = self.policy.prepare_prev_chunk(prev_raw, self._prev_state, cur_state)
                d = self._delay_ticks()
                out = self.policy.infer(
                    obs,
                    prev_chunk_left_over=prev[d:] if prev is not None else None,
                    inference_delay=d,
                    execution_horizon=self.args.execution_horizon,
                )
            else:
                out = self.policy.infer(obs)
            self._latency_ms.append((time.perf_counter() - t0) * 1000.0)
            actions = np.asarray(out["actions"], dtype=np.float32)
            raw = (
                np.asarray(self.policy.last_raw_chunk, dtype=np.float32)
                if self.rtc_enabled
                else actions
            )
            self.queue.merge(raw, actions, self._delay_ticks())
            if self.rtc_enabled:
                self._prev_state = cur_state
            else:
                # baseline：执行完整个 chunk 再推理，避免多 chunk 交错导致漂移
                # （与平台 harness 的“推理一次→执行完→再推理”一致）
                while self.queue.qsize() > 0 and not self._stop.is_set():
                    time.sleep(PERIOD / 4)

    def warmup(self, n: int = 3) -> None:
        """开跑前预热：吃掉 JAX 编译时间，避免污染延迟窗口。"""
        print(f"policy warmup x{n} ...")
        obs = get_observation(self.env)
        for _ in range(n):
            self.policy.infer(obs)
        self._latency_ms.clear()
        print("warmup done")

    def _listen_early_stop(self, stop_ev: threading.Event) -> None:
        """非阻塞监听回车：按一次提前结束本集（不阻塞 25Hz 主循环）。"""
        import select

        while not stop_ev.is_set() and not self._episode_early.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            except (ValueError, OSError):
                return
            if ready:
                line = sys.stdin.readline()
                if line == "":  # stdin 已关闭/EOF（非交互），不误触发
                    continue
                self._episode_early.set()
                return

    def run_episode(self, recorder: Recorder) -> dict:
        self.queue.clear()
        self._latency_ms.clear()
        self._prev_state = None
        self._last_action = None
        self._stop.clear()
        self._episode_early = threading.Event()
        ep_stop = threading.Event()
        listener = threading.Thread(
            target=self._listen_early_stop, args=(ep_stop,), daemon=True
        )
        listener.start()
        print("本集按 回车 可提前结束（否则 30s 超时）")
        worker = threading.Thread(target=self._inference_worker, daemon=True)
        worker.start()
        t_start = time.monotonic()
        start_qpos = np.asarray(self.env.get_observation()["qpos"], dtype=np.float32)
        home_hit_since = None
        actions_sent = 0
        latencies = []
        ended_by = "timeout"
        try:
            while (
                time.monotonic() - t_start < self.args.episode_timeout_s
                and not self._episode_early.is_set()
            ):
                t0 = time.perf_counter()
                action = self.queue.get()
                if action is not None:
                    obs = self.env.get_observation()
                    # worker 需要 openpi 格式观测（state/images/prompt），
                    # 录像需要原始帧（HWC BGR），两个都从这里派生
                    self._latest_obs = {
                        "state": np.asarray(obs["qpos"], dtype=np.float32),
                        "images": {
                            cam: obs["images"][cam].swapaxes(0, 2).swapaxes(1, 2)
                            for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist")
                        },
                        "prompt": PROMPT,
                    }
                    cur_qpos = np.asarray(obs["qpos"], dtype=np.float32)
                    # 硬保护 1：关节角必须在物理范围内（rad/deg 混用会直接爆表）
                    if self.args.joint_limit_rad > 0:
                        max_j = float(np.abs(action[:13]).max())
                        if max_j > self.args.joint_limit_rad:
                            raise RuntimeError(
                                f"动作关节角超限 {max_j:.2f} rad > {self.args.joint_limit_rad}，"
                                "中止（疑似单位/映射错误）"
                            )
                    # 硬保护 2：非首动作的单步偏差收紧（0.9 太松，允许快速过冲）
                    if self._last_action is not None and self.args.max_step_rad > 0:
                        step = float(np.abs(action[:13] - self._last_action[:13]).max())
                        if step > self.args.max_step_rad:
                            raise RuntimeError(
                                f"单步动作偏差 {step:.3f} rad > {self.args.max_step_rad}，"
                                "中止（疑似持续上行/垃圾动作）\n"
                                f"  当前动作[:13]: {action[:13]}\n"
                                f"  上一动作[:13]: {self._last_action[:13]}\n"
                                f"  当前位姿[:13]: {cur_qpos[:13]}"
                            )
                    if self._last_action is None and self.args.pose_check_rad > 0:
                        # 首动作没有 last_action 可对比，直接与当前位姿比对，防瞬间移动
                        dev = float(np.abs(action[:13] - cur_qpos[:13]).max())
                        if dev > self.args.pose_check_rad:
                            raise RuntimeError(
                                f"首动作与当前位姿偏差 {dev:.3f} rad > "
                                f"{self.args.pose_check_rad}，已中止（防止瞬间移动；"
                                "请确认机械臂在起始位姿、观测正常）"
                            )
                        if dev > 0.02 and self.args.interp_first:
                            # 像旧 harness 的 dynamic_approach：分小步逼近首动作
                            n = min(int(np.ceil(dev / 0.05)), 25)
                            print(f"首动作偏差 {dev:.3f} rad，插值 {n} 步逼近")
                            for jnt in np.linspace(cur_qpos, action, n):
                                check_action(jnt, self._last_action, self.safety)
                                self._last_action = jnt.copy()
                                self.env.step(jnt)
                                self.env.step_gripper(jnt)
                                actions_sent += 1
                                time.sleep(PERIOD)
                            # 已到位，本 tick 不再重复发送 action
                            latencies.extend(list(self._latency_ms))
                            continue
                    check_action(action, self._last_action, self.safety)
                    self._last_action = action.copy()
                    self.env.step(action)  # RealEnv.step(action, single_arm=True)
                    self.env.step_gripper(action)
                    actions_sent += 1
                    recorder.write(obs["images"])
                    # 自动结束：回到起始位姿附近并保持 home_hold_s
                    if self.args.home_threshold_rad > 0:
                        dev = float(
                            np.abs(np.asarray(obs["qpos"], dtype=np.float32) - start_qpos).max()
                        )
                        elapsed = time.monotonic() - t_start
                        if elapsed >= self.args.min_episode_s and dev <= self.args.home_threshold_rad:
                            if home_hit_since is None:
                                home_hit_since = time.monotonic()
                            elif time.monotonic() - home_hit_since >= self.args.home_hold_s:
                                print(f"检测到回到起始位姿 (dev={dev:.3f} rad)，自动结束本集")
                                ended_by = "home"
                                break
                        else:
                            home_hit_since = None
                else:
                    time.sleep(PERIOD / 4)
                latencies.extend(list(self._latency_ms))
                rem = PERIOD - (time.perf_counter() - t0)
                if rem > 0:
                    time.sleep(rem)
        finally:
            ep_stop.set()
            self._stop.set()
            worker.join(timeout=5)
        if self._episode_early.is_set():
            ended_by = "manual"
            print("人工提前结束本集")
        return {
            "actions_sent": actions_sent,
            "mean_infer_ms": float(np.mean(latencies)) if latencies else None,
            "est_delay_ticks": self._delay_ticks(),
            "duration_s": round(time.monotonic() - t_start, 2),
            "ended_by": ended_by,
        }


# ---------------------------------------------------------------------------
# 5) probe：实机（或 hdf5）探测延迟 -> d 推荐
# ---------------------------------------------------------------------------
def probe(args: argparse.Namespace) -> None:
    spec = MODELS[args.probe_target]
    args.mode = args.probe_target
    policy, cfg, ckpt = build_policy(spec, args)

    if args.hdf5:
        from openpi_rtc.measure_latency import Hdf5ObservationSource

        src = Hdf5ObservationSource(args.hdf5)
        obs = src()
    else:
        env = make_env(args)
        obs = get_observation(env)

    lat = []
    for _ in range(args.warmup):
        policy.infer(obs)
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        policy.infer(obs)
        lat.append((time.perf_counter() - t0) * 1000.0)
    mean = float(np.mean(lat))
    med = float(np.median(lat))
    p95 = float(np.percentile(lat, 95))
    d = math.ceil(p95 / TICK_MS) + 1
    print("\n===== DELAY PROBE =====")
    print(f"target: {args.probe_target}  checkpoint: {ckpt}")
    print(f"latency ms: mean={mean:.1f} median={med:.1f} p95={p95:.1f} (n={len(lat)})")
    print(f"control tick = {TICK_MS:.0f} ms")
    print(f"recommended d = ceil(p95/{TICK_MS:.0f}) + 1 = {d}")
    if args.probe_target == "train_rtc":
        print("约束: 训练时 simulated_delay 需 >= d+1（当前默认 8 -> d<=7）")
    elif args.probe_target == "pir2":
        print("约束: 训练时 max_delay 需 >= d（当前默认 8 -> d<=8）")
    print("=========================\n")
    return d


# ---------------------------------------------------------------------------
# 6) main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    print(f"openpi -> {openpi.__file__}")
    print(f"mode={args.mode}  safety={'ON' if not args.safety_off else 'OFF'}")

    if args.mode == "probe":
        probe(args)
        return 0

    spec = MODELS[args.mode]
    policy, cfg, ckpt = build_policy(spec, args)
    model_name = f"{ckpt.parent.name}_{ckpt.name}"
    record_root = pathlib.Path(args.record_dir or OPENPI_MAIN / "records")

    env = make_env(args)
    if args.reset_pose:
        vals = [float(x) for x in args.reset_pose.split(",")]
        if len(vals) != 14:
            sys.exit("ERROR: --reset-pose 需要 14 个值（6关节+夹爪）×2臂")
        reset_pose = np.asarray(vals, dtype=np.float32)
    else:
        reset_pose = DEFAULT_RESET_POSE
    if args.auto_reset:
        print("启动自动复位到参考位姿 ...")
        auto_reset(env, reset_pose)
    home_qpos = np.asarray(env.get_observation()["qpos"], dtype=np.float32)
    if not args.auto_reset:
        print("(未开 --auto-reset：每集起点=当前人工摆放位姿)")

    runner = BenchRunner(policy, env, args, model_name)
    if args.warmup > 0:
        runner.warmup(args.warmup)
    else:
        print("warmup 已关闭（首次推理编译会占用第一集开头几秒）")
    for ep in range(1, args.episodes + 1):
        recorder = Recorder(record_root, model_name, args.mode)
        print(f"\n===== episode {ep}/{args.episodes} (mode={args.mode}) =====")
        stats = runner.run_episode(recorder)
        recorder.close()
        meta = {
            "mode": args.mode,
            "config": args.config or spec["config"],
            "checkpoint": str(ckpt),
            "prompt": PROMPT,
            "episode": ep,
            "inference_delay": args.inference_delay,
            "execution_horizon": args.execution_horizon,
            "schedule": args.schedule,
            "num_steps": args.num_steps,
            "episode_timeout_s": args.episode_timeout_s,
            "min_episode_s": args.min_episode_s,
            "home_threshold_rad": args.home_threshold_rad,
            "home_hold_s": args.home_hold_s,
            "pose_check_rad": args.pose_check_rad,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            **stats,
        }
        (record_root / model_name / args.mode).mkdir(parents=True, exist_ok=True)
        with open(record_root / model_name / args.mode / f"episode_{ep}.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"episode {ep}: {stats}")
        if ep < args.episodes:
            if args.auto_reset:
                auto_reset(env, home_qpos)
            input("请人工放置试管/复位场景后按回车开始下一集 ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
