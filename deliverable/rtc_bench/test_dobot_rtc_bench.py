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
    python rtc_bench/test_dobot_rtc_bench.py --mode pir2 --slow-channel --episodes 5
        # πR² 慢通道 + 单步流（论文 fast mode）：每次调用只跑一步 DiT，
        # 视觉/语言前缀异步缓存，每 slow_refresh_every tick 刷新一次；
        # d 由 warmup 按一步 DiT 延迟重新测量（<= 训练 max_delay 即可）

输出视频: openpi-main/records/<模型名>/<mode>/episode_N/{cam_high,
cam_left_wrist,cam_right_wrist}.avi + episode_N.json
"""

from __future__ import annotations

import argparse
import collections
import csv
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

# ---------------------------------------------------------------------------
# 版本自检：openpi_rtc 包必须是本交付物新版（旧副本缺 guidance_jacobian /
# get_left_over_processed，会报 TypeError 或运行期 AttributeError）。
# 若命中，打印“实际导入的文件路径”，避免改错副本。
# ---------------------------------------------------------------------------
_rtc_missing = []
if not hasattr(RTCConfig, "guidance_jacobian"):
    _rtc_missing.append("RTCConfig.guidance_jacobian")
if not hasattr(ActionQueue, "get_left_over_processed"):
    _rtc_missing.append("ActionQueue.get_left_over_processed")
try:
    import inspect as _inspect_pir2

    import openpi_rtc.pir2_train as _pir2_mod

    if "slow_channel" not in _inspect_pir2.signature(
        _pir2_mod.wrap_policy_for_pir2
    ).parameters:
        _rtc_missing.append("wrap_policy_for_pir2(slow_channel=...)")
except Exception:  # noqa: BLE001
    _rtc_missing.append("openpi_rtc.pir2_train 导入失败")
if _rtc_missing:
    import inspect as _inspect

    sys.exit(
        "ERROR: openpi_rtc 版本过旧，缺少 "
        + ", ".join(_rtc_missing)
        + "。\n"
        f"  实际导入 RTCConfig: {_inspect.getfile(RTCConfig)}\n"
        f"  实际导入 ActionQueue: {_inspect.getfile(ActionQueue)}\n"
        "  请用 deliverable/rtc_bench/openpi_rtc 整目录覆盖工控机的 "
        "rtc_bench/openpi_rtc（整个目录，不要只覆盖测试脚本），"
        "并删除 rtc_bench/openpi_rtc/__pycache__ 后重跑。"
    )

PROMPT = "Transfer the test tube from the right rack to the left rack."
CONTROL_HZ = 25.0
PERIOD = 1.0 / CONTROL_HZ
TICK_MS = PERIOD * 1000.0

# 默认参考复位位姿 = task_00031_entong 数据集首帧（114 集统计），单位：度。
#   左臂为理论固定值；右臂取浮动范围内的理论值（R0≈89.23→90 等，
#   各集 ±0.1rad 浮动属摆放抖动）；夹爪开=0.9911。
DEFAULT_RESET_POSE_DEG = np.array([
    -90.0, 30.0, -110.0, 20.0, 90.0, 90.0, 0.9911,
     90.0, 0.0, 90.0, 0.0, -90.0, -90.0, 0.9911,
], dtype=np.float32)


def pose_deg_to_rad(v) -> np.ndarray:
    """把 [12 关节度 + 2 夹爪 0~1] 转成弧度（夹爪不变）。"""
    v = np.asarray(v, dtype=np.float32).copy()
    v[0:6] = np.deg2rad(v[0:6])
    v[7:13] = np.deg2rad(v[7:13])
    return v


def rad_to_deg(a) -> np.ndarray:
    """发送边界：弧度 → 度数（仅 12 个关节，夹爪 0~1 不变）。

    工控机底层契约（对方确认）：
      * command_joint_state 收**度数**（不再内部 rad2deg）；
      * get_joint_state 仍返回**弧度**。
    因此 bench 内部/安全检查/位姿对比保持弧度，仅在 env.step 前转换。
    """
    a = np.asarray(a, dtype=np.float32).copy()
    a[0:6] = np.rad2deg(a[0:6])
    a[7:13] = np.rad2deg(a[7:13])
    return a


def send_gripper(env, action, arms: str) -> None:
    """夹爪路由（物理交叉已实测确认）。

    实测：_robot_l(192.168.5.1) 的夹爪对象驱动【任务】物理夹爪，
    _robot_r(192.168.5.2) 的夹爪对象驱动【未用】物理夹爪。
    方向已确认与归一化一致：move(255)=开、move(0)=关（1=开、0=关）。
    因此单臂/双臂都按"左臂在前"的动作布局把正确的切片发到正确的对象：
      任务夹爪值 action[13] -> _robot_l 经 action[7:]（[-1]=action[13]）
      未用夹爪值 action[6]  -> _robot_r 经 action[:7]（[-1]=action[6]）
    """
    action = np.asarray(action, dtype=np.float32)
    if arms == "right":
        env._robot_l.command_joint_state_gripper(action[7:])   # 任务夹爪
    elif arms == "left":
        env._robot_r.command_joint_state_gripper(action[:7])   # 左夹爪
    else:
        env._robot_l.command_joint_state_gripper(action[7:])   # 任务夹爪
        env._robot_r.command_joint_state_gripper(action[:7])   # 未用夹爪


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
        / "checkpoints/dobot/pi05-task_00031_entong-xtrainer/pir2_v2/49999",
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
    ap.add_argument("--execution-horizon", type=int, default=20,
                    help="引导前缀的软衰减区长度（默认 20：更久地跟随旧流，"
                         "避免过渡区新块自由规划锚回滞后观测位姿造成鼓包/卡顿）")
    ap.add_argument("--max-guidance-weight", type=float, default=10.0)
    ap.add_argument("--schedule", default="exp", choices=["exp", "linear", "ones", "zeros"])
    ap.add_argument("--guidance-jacobian", default="identity",
                    choices=["identity", "full"],
                    help="引导校正的 Jacobian：identity=lerobot 量产版（默认，稳定）；"
                         "full=Kinetix 完整 vjp（transformer 上易失效）")
    ap.add_argument("--num-steps", type=int, default=10, help="πR² 去噪步数（10 或 1 快模式）")
    ap.add_argument("--slow-channel", action=argparse.BooleanOptionalAction, default=False,
                    help="πR² 慢通道：异步缓存视觉/语言前缀，每次调用只跑一步 DiT "
                         "（论文 fast mode；仅 --mode pir2 生效）")
    ap.add_argument("--slow-refresh-every", type=int, default=5,
                    help="πR² 慢通道前缀每隔多少 tick 刷新一次（0..image_delay_max 之间的真实年龄）")
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
    ap.add_argument("--debug-timing", action=argparse.BooleanOptionalAction, default=True,
                    help="写 timing CSV 到 records/.../episode_N/ 并实时打印异常（默认开）")
    ap.add_argument("--rtc-switch-queue", type=int, default=16,
                    help="rtc 队列剩多少步时重新推理/切换（默认 16：轨迹中段切换，"
                         "引导目标=延续段；过小会在队尾切换→停顿小抽）")
    ap.add_argument("--handoff-blend", type=int, default=10,
                    help="merge 后前 N 拍用 Hermite 桥从旧命令平滑接到新块"
                         "稳定段（默认 10，越过引导衰减区鼓包；首拍=纯旧命令；"
                         "不依赖模型引导）")
    ap.add_argument("--warmup", type=int, default=1,
                    help="预热/测 d 次数（0=跳过，此时 d 用 --inference-delay 或 7）")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--safety-off", action="store_true")
    ap.add_argument("--auto-reset", action=argparse.BooleanOptionalAction, default=True,
                    help="开始前/集间自动回到参考位姿（默认开；--no-auto-reset 关闭）")
    ap.add_argument("--reset-pose", default="",
                    help="14 个逗号分隔的值：12 个关节角(度) + 2 个夹爪(0~1)，"
                         "覆盖默认参考位姿；空=用数据集首帧位姿")
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
                      anchor_correction=True,
                      guidance_jacobian=args.guidance_jacobian),
            norm_stats=norm_stats,
        )
    elif wrapper == "train_rtc":
        policy = wrap_policy_for_train_rtc(policy, args.inference_delay or 7, norm_stats=norm_stats)
    elif wrapper == "pir2":
        policy = wrap_policy_for_pir2(
            policy,
            args.inference_delay or 7,
            norm_stats=norm_stats,
            slow_channel=args.slow_channel,
            image_delay_max=max(0, args.slow_refresh_every),
            slow_refresh_every=args.slow_refresh_every,
            num_steps=args.num_steps,
        )
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
    max_dev = float(np.abs(curr - target).max())
    steps = min(int(max_dev / 0.01), 100)
    print(f"auto_reset: max_dev={max_dev:.3f} rad, steps={steps}")
    for jnt in np.linspace(curr, target, steps):
        env.step(rad_to_deg(jnt))  # 新契约：发送边界转度数
        time.sleep(PERIOD)
    time.sleep(1.0)  # 等机械臂稳定，避免下一集首动作观测未稳定
    after = np.asarray(env.get_observation()["qpos"], dtype=np.float32)
    print(f"auto_reset 完成: 与目标最大偏差 {float(np.abs(after - target).max()):.3f} rad")


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
        # πR² 单步流模式（论文 fast mode）：缓存前缀 + 每次调用一步 DiT。
        self.stream = bool(getattr(policy, "_slow_channel", False))
        self._stream_ready = False
        # 一步 DiT 流的推理耗时（tick 数）；触发推理要提前这么多拍，
        # 使 merge 恰好落在“在飞窗口”被消费完的时刻（避免重复执行/跳变）。
        self._stream_drain_ticks = None
        self._stream_latency = []
        self.queue = ActionQueue(RTCConfig(enabled=self.rtc_enabled))
        self._latency_ms = collections.deque(maxlen=20)
        self._stop = threading.Event()
        self._prev_state = None
        self._last_action = None
        self._latest_obs = None
        self._latest_raw = None
        self._inference_delay = None
        self.debug = args.debug_timing
        self.safety = SafetyConfig(enabled=not args.safety_off, robot_type=args.robot_type)

    def _delay_ticks(self) -> int:
        # d 在 warmup 时固定，运行期间不变（inference_delay 是 jit static 参数，
        # 变化会触发重编译，导致 d 自激到封顶 16）
        if self._inference_delay is not None:
            return self._inference_delay
        return self.args.inference_delay or 7

    def _inference_worker(self) -> None:
        while not self._stop.is_set():
            if self.stream:
                # 单步流：队列剩到“在飞窗口”边界（H-d）时推进一次。
                d = self._delay_ticks()
                H = self.policy._model.action_horizon
                d = max(1, min(d, H // 3))  # 与 wrapper/staircase 的 clamp 一致
                drain = self._stream_drain_ticks
                if drain is None:
                    drain = 1
                trigger = max(1, H - d + drain)
                if self.queue.qsize() > trigger:
                    time.sleep(PERIOD / 4)
                    continue
                obs = self._latest_obs if self._latest_obs is not None else get_observation(self.env)
                t0 = time.perf_counter()
                out = self.policy.infer_stream(
                    obs,
                    inference_delay=d,
                    warm=not self._stream_ready,
                )
                self._stream_ready = True
                infer_ms = (time.perf_counter() - t0) * 1000.0
                self._latency_ms.append(infer_ms)
                actions = np.asarray(out["actions"], dtype=np.float32)
                raw = np.asarray(out["raw_actions"], dtype=np.float32)
                # 流模式连续：不做 Hermite 桥/引导（缓冲本身就是连续计划），
                # 直接替换队尾，避免桥接逻辑与流缓冲冲突。
                self.queue.merge(raw, actions, 0)
                if self.debug:
                    self._infer_w.writerow([
                        round(t0, 6), round(infer_ms, 2), d, H,
                        "", "", "", "", "",
                        self.queue.qsize(), int(out.get("warm", False)),
                    ])
                    if infer_ms > 250:
                        print(f"[debug] 流推理 {infer_ms:.0f}ms d={d} "
                              f"warm={out.get('warm', False)} age={out.get('slow_age', '')}",
                              flush=True)
                continue
            # 队列剩 ~rtc_switch_queue 步时重新推理：轨迹中段切换，
            # 引导目标=旧块的延续段（不是终点），避免切换处停顿小抽。
            if self.queue.qsize() > self.args.rtc_switch_queue:
                time.sleep(PERIOD)
                continue
            # 用观测线程的最新观测（不要在这里读相机）
            obs = self._latest_obs if self._latest_obs is not None else get_observation(self.env)
            cur_state = np.asarray(obs["state"], dtype=np.float32)
            # ── 未来优化：状态锚点补偿（STATE ANCHOR）──────────────────────
            # 现状：新块自由规划锚在“观测位姿”（delta 动作 = 绝对−观测位姿），
            # 而观测位姿滞后命令流一个跟随误差（act−qpos≈0.05~0.1 rad）。
            # 模型因此“压住关节等物理位追上来”，形成剩余小停顿。
            # 方案：给模型输入 state 用 cur_state + λ·(命令位姿 − cur_state)，
            # λ∈[0,1]（命令位姿≈old_proc[0]，merge 前可拿）。λ=0.5 起步，
            # 观察停顿是否消失、图像/状态不一致是否影响任务完成。
            # 注意：prev 重锚（prepare_prev_chunk）也要用同一修正后 state。
            # ───────────────────────────────────────────────────────────────
            t0 = time.perf_counter()
            prev_len = 0
            executed_used = None
            prev0_maxabs = None
            if self.rtc_enabled:
                prev_raw = getattr(self.policy, "last_raw_chunk", None)
                prev = None
                if prev_raw is not None and self._prev_state is not None:
                    full = self.policy.prepare_prev_chunk(prev_raw, self._prev_state, cur_state)
                    h = full.shape[0]
                    # 引导目标 = 旧块在 merge 时刻的“剩余命令”（已重锚到
                    # cur_state），即新块前段应当延续旧块的命令流。
                    # 注意：不能把 tail 再减 tail[0] 锚到“观测位姿”——观测是
                    # 滞后的，旧块命令超前观测一个 lead（act-qpos），锚回观测
                    # 位姿会让新块首拍相对旧命令回撤一个 lead，正是切换后拉
                    # 的来源（实测 19 次 merge 的跳变与 lead 相关 r=-0.77）。
                    # executed = 触发时已执行 + 推理期间队列继续被消费的拍数，
                    # 用最近实测推理耗时估计后者（≈ infer_ms/TICK_MS）。
                    drain = self._inference_delay if self._inference_delay is not None else 3
                    if self._latency_ms:
                        drain = max(1, int(round(
                            float(np.mean(self._latency_ms)) / TICK_MS
                        )))
                    executed = max(0, min(h - self.queue.qsize() + drain, h - 1))
                    tail = full[executed:]
                    n = h - executed
                    if n >= h or len(tail) < 2:
                        # 旧流覆盖整个块，或只剩 1 拍：直接重复末值（保持）。
                        prev = np.repeat(tail[-1:], h, axis=0)
                        if n > 0:
                            prev[:n] = tail
                    else:
                        # 旧流剩余 + 限幅速度外推（最多延续 6 拍的速度），
                        # 避免重复末值=命令急停，以及 execution_horizon 变长后
                        # 引导目标突然“定格”造成过渡区鼓包。
                        vel = tail[-1] - tail[-2]
                        steps = np.arange(1, h - n + 1, dtype=np.float32)[:, None]
                        ext = tail[-1:] + vel * np.minimum(steps, 6.0)
                        prev = np.concatenate([tail, ext], axis=0)
                    executed_used = executed
                    # 引导目标第 0 拍（14 维）：取 12 个关节维的最大绝对值，
                    # 归一化空间，用于核对锚点是否落在旧命令流上。
                    tail0 = np.asarray(tail[0], dtype=np.float32)
                    prev0_maxabs = float(max(tail0[:6].max(), tail0[7:13].max()))
                d = self._delay_ticks()
                prev_len = h if prev is not None else 0
                out = self.policy.infer(
                    obs,
                    prev_chunk_left_over=prev,
                    inference_delay=d,
                    execution_horizon=self.args.execution_horizon,
                )
            else:
                out = self.policy.infer(obs)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            self._latency_ms.append(infer_ms)
            actions = np.asarray(out["actions"], dtype=np.float32)
            raw = (
                np.asarray(self.policy.last_raw_chunk, dtype=np.float32)
                if self.rtc_enabled
                else actions
            )
            d_used = self._delay_ticks()
            # 引导是否把新块前段拉到旧命令流：位置 0 的目标 vs 实际（归一化）。
            guide_gap = guide_target = guide_actual = ""
            if self.rtc_enabled and prev is not None:
                raw0 = np.asarray(raw)[0]
                prev0v = np.asarray(prev)[0]
                jmask = list(range(6)) + list(range(7, 13))
                gap = np.abs(raw0 - prev0v)[jmask]
                gi = jmask[int(np.argmax(gap))]
                guide_gap = f"{float(gap.max()):.4f}"
                guide_target = f"{float(prev0v[gi]):+.4f}"
                guide_actual = f"{float(raw0[gi]):+.4f}"
            # 队列侧过渡（不依赖模型引导）：merge 后前 K 拍用三次 Hermite
            # 桥，从旧流当前命令（P0，初始速度 0=先不继续下探）平滑接到新块
            # “稳定段”P1=actions[K]（K 越过引导衰减区的鼓包）。
            # 逐拍位置混合会先跟旧流下探再回升 → 物理小 V 形回撤（实测
            # qpos 下探 0.025 再回升 0.025）；Hermite 桥直接上升汇合，V 大幅
            # 缩小。首拍=P0 → 命令零跳变。
            old_proc = None
            if self.rtc_enabled:
                old_proc = self.queue.get_left_over_processed()
            if (
                self.rtc_enabled
                and old_proc is not None
                and len(old_proc) > 1
                and self.args.handoff_blend > 1
            ):
                K = min(self.args.handoff_blend, len(actions) - 2)
                if K >= 3:
                    P0 = np.asarray(old_proc[0], dtype=np.float32)
                    P1 = np.asarray(actions[K], dtype=np.float32)
                    v1 = (
                        np.asarray(actions[K + 1] - actions[K], dtype=np.float32)
                        if K + 1 < len(actions)
                        else np.zeros_like(P1)
                    )
                    # v0 朝 P1 方向起桥（约 3× 平均速度，幅值不超旧流速度），
                    # 避免“先保持→臂继续下探→再回升”的 V 形回撤。
                    vmax = (
                        float(np.abs(old_proc[0] - old_proc[1]).max())
                        if len(old_proc) > 1
                        else 0.01
                    )
                    v0 = np.clip(3.0 * (P1 - P0) / K, -vmax, vmax)
                    tt = np.linspace(0.0, 1.0, K + 1, dtype=np.float32)[:, None]
                    h00 = 2.0 * tt**3 - 3.0 * tt**2 + 1.0
                    h10 = tt**3 - 2.0 * tt**2 + tt
                    h01 = -2.0 * tt**3 + 3.0 * tt**2
                    h11 = tt**3 - tt**2
                    bridge = (
                        h00 * P0
                        + h10 * (K * v0)
                        + h01 * P1
                        + h11 * (K * v1)
                    )
                    actions = actions.copy()
                    actions[: K + 1] = bridge
            # rtc: 从新块第 0 位执行（不再跳 d），边界由引导+过渡衔接
            self.queue.merge(raw, actions, 0)
            clamped = 0
            if self.debug:
                self._infer_w.writerow([
                    round(t0, 6), round(infer_ms, 2), d_used, prev_len,
                    executed_used if executed_used is not None else "",
                    (f"{prev0_maxabs:.4f}" if prev0_maxabs is not None else ""),
                    guide_gap, guide_target, guide_actual,
                    self.queue.qsize(), clamped,
                ])
                if infer_ms > 250:
                    print(f"[debug] 推理 {infer_ms:.0f}ms d={d_used} prev_len={prev_len}", flush=True)
            if self.rtc_enabled:
                self._prev_state = cur_state
            else:
                # baseline：执行完整个 chunk 再推理，避免多 chunk 交错导致漂移
                # （与平台 harness 的“推理一次→执行完→再推理”一致）
                while self.queue.qsize() > 0 and not self._stop.is_set():
                    time.sleep(PERIOD / 4)

    def _observe_loop(self, stop_ev: threading.Event) -> None:
        """独立观测线程：读相机+qpos，更新最新观测（executor 不再读相机）。"""
        while not stop_ev.is_set():
            t0 = time.perf_counter()
            try:
                raw = self.env.get_observation()
            except Exception as e:  # noqa: BLE001
                time.sleep(PERIOD)
                continue
            read_ms = (time.perf_counter() - t0) * 1000.0
            self._latest_raw = raw
            self._latest_obs = {
                "state": np.asarray(raw["qpos"], dtype=np.float32),
                "images": {
                    cam: raw["images"][cam].swapaxes(0, 2).swapaxes(1, 2)
                    for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist")
                },
                "prompt": PROMPT,
            }
            if self.debug:
                self._obs_w.writerow([round(t0, 6), round(read_ms, 2)])
                if read_ms > 150:
                    print(f"[debug] 观测读取 {read_ms:.0f}ms", flush=True)
            time.sleep(PERIOD)

    def warmup(self, n: int = 3) -> None:
        """开跑前预热：吃掉 JAX 编译时间，并固定 d（运行期不再变）。

        关键：必须用 episode 完全相同的 jit 签名（inference_delay=d、
        prev=None 与 prev[d:] 两条路径）编译，否则首集/中途会重编译，
        导致队列饿死->灌满->一顿一顿。
        """
        print(f"policy warmup x{n} ...")
        obs = get_observation(self.env)
        if self.stream:
            self._stream_warmup(n)
            return
        # 1) 无引导路径：编译 + 测稳态延迟 -> 定最终 d
        for _ in range(max(1, n // 2)):
            self.policy.infer(obs)
        lat = []
        for _ in range(max(n, 3)):
            t0 = time.perf_counter()
            self.policy.infer(obs)
            lat.append((time.perf_counter() - t0) * 1000.0)
        d0 = (
            self.args.inference_delay
            if self.args.inference_delay
            else min(16, max(1, math.ceil(float(np.mean(lat)) / TICK_MS)))
        )
        self._inference_delay = d0
        if not self.rtc_enabled:
            self._latency_ms.clear()
            print(f"warmup done (baseline): _inference_delay={d0}")
            return
        # 2) 编译引导路径的两种签名（与 episode 完全一致）
        self.policy.infer(
            obs, prev_chunk_left_over=None, inference_delay=d0,
            execution_horizon=self.args.execution_horizon,
        )
        prev = np.asarray(self.policy.last_raw_chunk, dtype=np.float32)
        h = prev.shape[0]
        # 与 episode 相同的固定 H 形状：前面放“剩余”，后面重复末位
        prev_slice = np.repeat(prev[-1:], h, axis=0)
        n = h - min(d0, h - 1)
        if n > 0:
            prev_slice[:n] = prev[d0:]
        for _ in range(max(1, n)):
            self.policy.infer(
                obs, prev_chunk_left_over=prev_slice, inference_delay=d0,
                execution_horizon=self.args.execution_horizon,
            )
        self._latency_ms.clear()
        print(f"warmup done: _inference_delay={d0} (no-guidance mean {np.mean(lat):.1f} ms)")

    def _stream_warmup(self, n: int = 3) -> None:
        """πR² 单步流预热：编译慢通道刷新 + 标准流 warm start + 一步 DiT。

        先用一个试探 d 编译并测量一步 DiT 延迟，定下最终 d；若与试探值
        不同，用最终 d 重编译一次（避免首集运行时重编译）。
        """
        d0 = self.args.inference_delay or 3
        d0 = max(1, min(d0, 16))
        # 1) 慢通道前缀刷新路径 + 标准流 warm start 路径
        self.policy.refresh_slow(obs := get_observation(self.env))
        self.policy.warm_start(obs, d=d0)
        # 2) 一步 DiT 流路径：编译 + 测延迟 -> 定最终 d
        self._stream_step_latency(obs, d0, n)
        lat = self._stream_latency
        d_final = (
            self.args.inference_delay
            if self.args.inference_delay
            else min(16, max(1, math.ceil(float(np.mean(lat)) / TICK_MS)))
        )
        if d_final != d0:
            # 用最终 d 重编译（d 是 jit static 参数，不能运行时变）
            self.policy.warm_start(obs, d=d_final)
            self._stream_step_latency(obs, d_final, n)
            lat = self._stream_latency
        self._inference_delay = d_final
        self._stream_drain_ticks = max(
            1, min(d_final, int(round(float(np.mean(lat)) / TICK_MS)))
        )
        self._latency_ms.clear()
        if d_final > 8:
            print(
                f"[WARN] 一步 DiT 延迟对应 d={d_final}，超过 πR² 训练预算 "
                f"max_delay=8。动作会被钳制更多在飞步，质量下降；建议调大 "
                f"--max-delay 或换更快推理硬件。"
            )
        print(f"stream warmup done: _inference_delay={d_final} "
              f"(one-DiT-step mean {np.mean(lat):.1f} ms, "
              f"drain={self._stream_drain_ticks} ticks)")

    def _stream_step_latency(self, obs: dict, d: int, n: int) -> None:
        """编译并测量一步 DiT 延迟（排除慢通道前缀刷新的调用）。"""
        for _ in range(max(1, n // 2)):
            self.policy.infer_stream(obs, inference_delay=d)
        lat = []
        for _ in range(max(n, 3)):
            t0 = time.perf_counter()
            out = self.policy.infer_stream(obs, inference_delay=d)
            ms = (time.perf_counter() - t0) * 1000.0
            if not out.get("refreshed"):
                lat.append(ms)
        if len(lat) < 2:
            # 兜底（理论上 refresh 只在 age>=slow_refresh_every 时发生，
            # 测量窗口内不会全部命中；取保守 1 tick）
            lat = [float(TICK_MS)]
        self._stream_latency = lat

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
        self._stream_ready = False
        self._latency_ms.clear()
        self._prev_state = None
        self._last_action = None
        self._latest_obs = None  # 关键：清掉上一集残留观测，首帧必须复位后现读
        self._latest_raw = None
        self._stop.clear()
        self._episode_early = threading.Event()
        if self.debug:
            recorder.ep_dir.mkdir(parents=True, exist_ok=True)
            self._ctrl_f = open(recorder.ep_dir / "control.csv", "w", newline="")
            self._ctrl_w = csv.writer(self._ctrl_f)
            self._ctrl_w.writerow(["t_mono", "tick_delta_ms", "queue_size_before",
                                   "env_step_ms", "record_ms", "loop_total_ms",
                                   "sleep_ms", "action_delta_max_rad", "action_idx",
                                   "act_l0", "act_l1", "act_l2", "act_l3", "act_l4", "act_l5",
                                   "act_l6", "act_r0", "act_r1", "act_r2", "act_r3", "act_r4",
                                   "act_r5", "act_r6",
                                   "qpos_l0", "qpos_l1", "qpos_l2", "qpos_l3", "qpos_l4",
                                   "qpos_l5", "qpos_l6", "qpos_r0", "qpos_r1", "qpos_r2",
                                   "qpos_r3", "qpos_r4", "qpos_r5", "qpos_r6"])
            self._infer_f = open(recorder.ep_dir / "infer.csv", "w", newline="")
            self._infer_w = csv.writer(self._infer_f)
            self._infer_w.writerow(["t_mono", "infer_ms", "d", "prev_len",
                                    "executed", "prev0_maxabs",
                                    "guide_gap", "guide_target", "guide_actual",
                                    "queue_size_after_merge", "clamped"])
            self._obs_f = open(recorder.ep_dir / "obs.csv", "w", newline="")
            self._obs_w = csv.writer(self._obs_f)
            self._obs_w.writerow(["t_mono", "read_ms"])
        ep_stop = threading.Event()
        listener = threading.Thread(
            target=self._listen_early_stop, args=(ep_stop,), daemon=True
        )
        listener.start()
        print("本集按 回车 可提前结束（否则 30s 超时）")
        obs_stop = threading.Event()
        observer = threading.Thread(target=self._observe_loop, args=(obs_stop,), daemon=True)
        observer.start()
        worker = threading.Thread(target=self._inference_worker, daemon=True)
        worker.start()
        t_start = time.monotonic()
        start_qpos = np.asarray(self.env.get_observation()["qpos"], dtype=np.float32)
        home_hit_since = None
        actions_sent = 0
        latencies = []
        ended_by = "timeout"
        prev_tick = time.perf_counter()
        last_action_time = None
        empty_warned = False
        try:
            while (
                time.monotonic() - t_start < self.args.episode_timeout_s
                and not self._episode_early.is_set()
            ):
                t0 = time.perf_counter()
                queue_size_before = self.queue.qsize()
                action = self.queue.get()
                action_delta_max_rad = 0.0
                env_ms = rec_ms = 0.0
                if action is not None:
                    last_action_time = time.perf_counter()
                    empty_warned = False
                    if self._last_action is not None:
                        action_delta_max_rad = float(
                            np.abs(action[:13] - self._last_action[:13]).max()
                        )
                    raw = self._latest_raw
                    if raw is None:  # 观测线程还没就绪，兜底读一次
                        raw = self.env.get_observation()
                        self._latest_raw = raw
                    cur_qpos = np.asarray(raw["qpos"], dtype=np.float32)
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
                                self.env.step(rad_to_deg(jnt))
                                send_gripper(self.env, jnt, self.args.arms)
                                actions_sent += 1
                                time.sleep(PERIOD)
                            # 已到位，本 tick 不再重复发送 action
                            latencies.extend(list(self._latency_ms))
                            continue
                    check_action(action, self._last_action, self.safety)
                    self._last_action = action.copy()
                    t_env = time.perf_counter()
                    self.env.step(rad_to_deg(action))  # 新契约：发送边界转度数
                    send_gripper(self.env, action, self.args.arms)
                    env_ms = (time.perf_counter() - t_env) * 1000.0
                    actions_sent += 1
                    t_rec = time.perf_counter()
                    recorder.write(raw["images"])
                    rec_ms = (time.perf_counter() - t_rec) * 1000.0
                    # 自动结束：回到起始位姿附近并保持 home_hold_s
                    if self.args.home_threshold_rad > 0:
                        dev = float(np.abs(np.asarray(raw["qpos"], dtype=np.float32) - start_qpos).max())
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
                    if (
                        not empty_warned
                        and last_action_time is not None
                        and time.perf_counter() - last_action_time > 0.3
                    ):
                        print(
                            f"[debug] 队列空 {time.perf_counter() - last_action_time:.2f}s（在等推理）",
                            flush=True,
                        )
                        empty_warned = True
                latencies.extend(list(self._latency_ms))
                now = time.perf_counter()
                tick_delta_ms = (now - prev_tick) * 1000.0
                prev_tick = now
                loop_total_ms = (now - t0) * 1000.0
                rem = PERIOD - (time.perf_counter() - t0)
                if rem > 0:
                    time.sleep(rem)
                if self.debug and action is not None:
                    self._ctrl_w.writerow([
                        round(t0, 6), round(tick_delta_ms, 2), queue_size_before,
                        round(env_ms, 2), round(rec_ms, 2), round(loop_total_ms, 2),
                        round(max(0.0, rem * 1000.0), 2), round(action_delta_max_rad, 4),
                        actions_sent,
                        *[round(float(v), 4) for v in action[:6]],
                        round(float(action[6]), 4),
                        *[round(float(v), 4) for v in action[7:13]],
                        round(float(action[13]), 4),
                        *[round(float(v), 4) for v in cur_qpos[:6]],
                        round(float(cur_qpos[6]), 4),
                        *[round(float(v), 4) for v in cur_qpos[7:13]],
                        round(float(cur_qpos[13]), 4),
                    ])
                    if tick_delta_ms > 120:
                        print(
                            f"[debug] tick 间隔 {tick_delta_ms:.0f}ms "
                            f"env={env_ms:.0f} rec={rec_ms:.0f} q={queue_size_before}",
                            flush=True,
                        )
                    if action_delta_max_rad > 0.2:
                        print(
                            f"[debug] 动作突变 {action_delta_max_rad:.3f} rad (idx={actions_sent})",
                            flush=True,
                        )
        finally:
            ep_stop.set()
            obs_stop.set()
            self._stop.set()
            observer.join(timeout=2)
            worker.join(timeout=5)
            if self.debug:
                self._ctrl_f.close()
                self._infer_f.close()
                self._obs_f.close()
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
    if args.probe_target == "pir2" and args.slow_channel:
        # 单步流探测：慢通道刷新 + warm start + 一步 DiT
        policy.refresh_slow(obs)
        policy.warm_start(obs, d=3)
        for _ in range(max(1, args.warmup)):
            policy.infer_stream(obs, inference_delay=3)
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            out = policy.infer_stream(obs, inference_delay=3)
            ms = (time.perf_counter() - t0) * 1000.0
            if not out.get("refreshed"):
                lat.append(ms)  # 排除混入的前缀刷新时间
    else:
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
            sys.exit("ERROR: --reset-pose 需要 14 个值（12 关节度 + 2 夹爪）")
        reset_pose = pose_deg_to_rad(vals)
    else:
        reset_pose = pose_deg_to_rad(DEFAULT_RESET_POSE_DEG)
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
            "slow_channel": args.slow_channel,
            "slow_refresh_every": args.slow_refresh_every,
            "stream": bool(getattr(policy, "_slow_channel", False)),
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
