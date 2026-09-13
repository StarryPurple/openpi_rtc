#!/usr/bin/env python3
"""openpi_rtc 环境与接口契约探测（自包含，不依赖 git）。

用法（在 openpi-main 根目录，用其 venv python 3.11）:
    python probe_diag.py                 # 只读探测
    python probe_diag.py --move-test     # 加 3° 微动探针（人守急停）
    python probe_diag.py --with-gripper  # 允许夹爪初始化（默认跳过）

输出: /tmp/probe_diag_<时间戳>.log（贴回给开发即可）
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import inspect
import json
import numpy as np
import os
import pathlib
import platform
import shutil
import socket
import subprocess
import sys
import time


def log(msg: str = "") -> None:
    line = str(msg)
    print(line, flush=True)
    _LOG.write(line + "\n")
    _LOG.flush()


_LOG: object = None  # set in main


def find_root() -> pathlib.Path:
    """Locate openpi-main root (dir containing examples/xtrainer_real)."""
    here = pathlib.Path.cwd()
    for cand in [here, *here.parents]:
        if (cand / "examples" / "xtrainer_real").exists():
            return cand
    return here


def run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def section(title: str) -> None:
    log("\n" + "=" * 70)
    log(f"== {title}")
    log("=" * 70)


def file_snapshot(path: pathlib.Path, max_lines: int = 400) -> None:
    """Dump a file's unit/interface-relevant lines + mtime (no git needed)."""
    if not path.exists():
        log(f"[缺失] {path}")
        return
    st = path.stat()
    mtime = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    log(f"[文件] {path}  mtime={mtime}  size={st.st_size}")
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:  # noqa: BLE001
        log(f"[读取失败] {e}")
        return
    keys = (
        "deg2rad", "rad2deg", "GetAngle", "ServoJ", "JointMovJ",
        "command_joint_state", "get_joint_state", "get_observations",
        "gripper", "step(", "step_gripper", "reset_position", "arms",
        "def __init__", "class ", "com_list", "servo_pos", "ttyUSB",
        "DobotGripper", "EnableRobot", "SpeedFactor", "move(",
    )
    hits = [
        (i + 1, ln) for i, ln in enumerate(lines)
        if any(k in ln for k in keys)
    ]
    shown = 0
    for no, ln in hits:
        if shown >= max_lines:
            log(f"  ...（余 {len(hits) - shown} 行命中略）")
            break
        log(f"  L{no}: {ln.rstrip()}")
        shown += 1
    if not hits:
        log("  （无单位/接口关键字命中）")


def file_full(path: pathlib.Path) -> None:
    """整文件 dump（用于小文件：夹爪驱动等）。"""
    if not path.exists():
        log(f"[缺失] {path}")
        return
    st = path.stat()
    mtime = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    log(f"[全文] {path}  mtime={mtime}  size={st.st_size}")
    try:
        for no, ln in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            log(f"  L{no}: {ln}")
    except Exception as e:  # noqa: BLE001
        log(f"[读取失败] {e}")


def probe_robot(root: pathlib.Path, move_test: bool, with_gripper: bool) -> None:
    section("实机接口探测")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    try:
        from examples.xtrainer_real.real_env import RealEnv
        from examples.xtrainer_real.robots.dobot import DobotRobot
    except Exception as e:  # noqa: BLE001
        log(f"[导入失败] RealEnv/DobotRobot: {type(e).__name__}: {e}")
        return

    for name, fn in [
        ("RealEnv.__init__", RealEnv.__init__),
        ("RealEnv.step", RealEnv.step),
        ("RealEnv.step_gripper", RealEnv.step_gripper),
        ("RealEnv.get_observation", RealEnv.get_observation),
        ("DobotRobot.command_joint_state", DobotRobot.command_joint_state),
        ("DobotRobot.command_joint_state_movj", DobotRobot.command_joint_state_movj),
        ("DobotRobot.command_joint_state_gripper", DobotRobot.command_joint_state_gripper),
        ("DobotRobot.get_joint_state", DobotRobot.get_joint_state),
        ("DobotRobot.get_observations", DobotRobot.get_observations),
    ]:
        try:
            log(f"[签名] {name}{inspect.signature(fn)}")
        except Exception as e:  # noqa: BLE001
            log(f"[签名] {name}: {type(e).__name__}: {e}")

    # 反编译关键函数源码（比签名更直接）
    for name, fn in [
        ("DobotRobot.command_joint_state", DobotRobot.command_joint_state),
        ("DobotRobot.get_joint_state", DobotRobot.get_joint_state),
        ("RealEnv.step", RealEnv.step),
        ("RealEnv.step_gripper", RealEnv.step_gripper),
    ]:
        try:
            log(f"[源码] {name}:")
            for ln in inspect.getsource(fn).splitlines():
                log(f"    {ln}")
        except Exception as e:  # noqa: BLE001
            log(f"[源码] {name}: {type(e).__name__}: {e}")

    try:
        env = RealEnv(False, arms="right", no_gripper=not with_gripper)
    except Exception as e:  # noqa: BLE001
        log(f"[RealEnv 实例化失败] {type(e).__name__}: {e}")
        return

    # 只读：读 qpos（期望弧度）
    try:
        obs = env.get_observation()
        q = [float(x) for x in obs["qpos"]]
        log(f"[qpos] (期望弧度, ±1.57 量级, 左臂在前): {q}")
        if obs.get("images"):
            log(f"[images] keys={list(obs['images'].keys())}")
            for k, v in obs["images"].items():
                log(f"    {k}: shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")
    except Exception as e:  # noqa: BLE001
        log(f"[读 qpos 失败] {type(e).__name__}: {e}")

    if not move_test:
        log("[微动探针] 跳过（加 --move-test 启用；会真实移动右臂 J0 约 3°）")
        return

    # 微动探针（新契约：command_joint_state 收度数，get_joint_state 返回弧度）：
    # 右臂 J0 +5°（0.0873 rad），以度数发送，读回弧度增量应≈+0.0873。
    try:
        q0 = [float(x) for x in env.get_observation()["qpos"]]
        cmd = np_from_list(q0)
        cmd[7] += 0.0873  # 右臂 J0 +5°（弧度，左臂在前右臂起点下标 7）
        # 发送边界：仅 12 关节转度数，夹爪 0~1 不变（新契约）
        cmd_deg = cmd.copy()
        cmd_deg[0:6] = np.rad2deg(cmd[0:6])
        cmd_deg[7:13] = np.rad2deg(cmd[7:13])
        log(f"[微动] 发送右臂 J0 +5°（当前弧度 J0={q0[7]:.4f}，发角度 {cmd_deg[7]:.2f}）")
        env.step(cmd_deg)
        time.sleep(1.5)
        q1 = [float(x) for x in env.get_observation()["qpos"]]
        log(f"[微动] 读回 J0={q1[7]:.4f} rad，增量={q1[7] - q0[7]:+.4f}（期望≈+0.0873）")
        # 恢复
        cmd2 = np_from_list(q1)
        cmd2[7] = q0[7]
        cmd2_deg = cmd2.copy()
        cmd2_deg[0:6] = np.rad2deg(cmd2[0:6])
        cmd2_deg[7:13] = np.rad2deg(cmd2[7:13])
        env.step(cmd2_deg)
        time.sleep(1.2)
        q2 = [float(x) for x in env.get_observation()["qpos"]]
        log(f"[微动] 恢复后 J0={q2[7]:.4f} rad（期望回到≈{q0[7]:.4f}）")
    except Exception as e:  # noqa: BLE001
        log(f"[微动探针失败] {type(e).__name__}: {e}")


def np_from_list(vals):
    import numpy as np
    return np.asarray(vals, dtype=np.float32)


def env_probe(root: pathlib.Path) -> None:
    section("环境与残留进程")
    log(f"[cwd] {pathlib.Path.cwd()}")
    log(f"[python] {sys.version.split()[0]}  {sys.executable}")
    log(f"[host] {socket.gethostname()}  {platform.platform()}")
    for mod in ("numpy", "jax", "torch", "flax"):
        try:
            m = importlib.import_module(mod)
            log(f"[dep] {mod} {getattr(m, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            log(f"[dep] {mod}: {type(e).__name__}")
    try:
        import openpi
        log(f"[openpi] {openpi.__file__}")
    except Exception as e:  # noqa: BLE001
        log(f"[openpi] 导入失败: {type(e).__name__}: {e}")
    try:
        import openpi_rtc
        log(f"[openpi_rtc] {openpi_rtc.__file__}")
        from openpi_rtc.rtc_config import RTCConfig
        log(f"[版本] RTCConfig.guidance_jacobian={'有' if hasattr(RTCConfig, 'guidance_jacobian') else '无'}")
        from openpi_rtc.action_queue import ActionQueue
        log(f"[版本] ActionQueue.get_left_over_processed={'有' if hasattr(ActionQueue, 'get_left_over_processed') else '无'}")
    except Exception as e:  # noqa: BLE001
        log(f"[openpi_rtc] 导入失败: {type(e).__name__}: {e}")

    rc, out = run(["ps", "aux"])
    log("[ps] 相关进程:")
    for ln in out.splitlines():
        if any(k in ln.lower() for k in ("python", "dobot", "realsense", "obsutil")):
            log(f"    {ln}")
    rc, out = run(["sh", "-c", "netstat -tlnp 2>/dev/null | grep -E '29999|30003|30004'"])
    log(f"[端口] 29999/30003/30004 占用:\n{out or '    （无输出/无占用）'}")
    rc, out = run(["sh", "-c", "ls -l /dev/ttyUSB* 2>/dev/null"])
    log(f"[串口] ttyUSB:\n{out or '    （无）'}")
    if shutil.which("nvidia-smi"):
        rc, out = run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv"])
        log(f"[gpu]\n{out}")
    rc, out = run(["df", "-h", str(root)])
    log(f"[磁盘]\n{out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--move-test", action="store_true",
                    help="微动探针：右臂 J0 +3° 读回并恢复（人守急停）")
    ap.add_argument("--with-gripper", action="store_true",
                    help="允许夹爪初始化（默认跳过，避免开关夹爪副作用）")
    args = ap.parse_args()

    global _LOG
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = pathlib.Path(f"/tmp/probe_diag_{ts}.log")
    _LOG = open(log_path, "w", encoding="utf-8")
    log(f"# openpi_rtc 探测报告  {ts}")

    root = find_root()
    log(f"[root] 判定 openpi-main 根目录 = {root}")

    env_probe(root)

    section("关键文件快照（含修改时间，替代 git diff）")
    for rel in [
        "examples/xtrainer_real/robots/dobot.py",
        "examples/xtrainer_real/real_env.py",
        "examples/xtrainer_real/constants.py",
        "examples/xtrainer_real/gripper/dobot_gripper.py",
        "examples/xtrainer_real/cameras/camera.py",
        "examples/xtrainer_real/cameras/realsense_camera.py",
        "rtc_bench/test_dobot_rtc_bench.py",
        "rtc_bench/openpi_rtc/rtc_train.py",
        "rtc_bench/openpi_rtc/pir2_train.py",
        "rtc_bench/openpi_rtc/integrate_openpi.py",
        "rtc_bench/openpi_rtc/safety.py",
    ]:
        file_snapshot(root / rel)

    file_full(root / "examples/xtrainer_real/gripper/dobot_gripper.py")

    probe_robot(root, args.move_test, args.with_gripper)

    section("结束")
    log(f"报告已写入: {log_path}")
    _LOG.close()
    print(f"\n请把 {log_path} 发回给开发。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
