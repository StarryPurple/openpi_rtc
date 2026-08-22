#!/usr/bin/env python3
"""One-shot environment check for the robot PC (工控机).

Runs every check from CHECKLIST.md Phase 1 and writes a report you can send
back as-is. Read-only except for connecting to cameras/robots:

    python3 openpi_rtc/check_robot_pc.py [--control-repo <dobot控制仓库根>] [--with-robot]

Exit code 0 = all critical checks passed; 1 = something to fix.
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import subprocess
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _run(cmd: list[str], timeout: float = 8.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except (FileNotFoundError, OSError) as e:
        return 127, f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "timeout"


class Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures = 0

    def _add(self, status: str, name: str, detail: str = "") -> None:
        line = f"[{status:>4}] {name}" + (f"  ({detail})" if detail else "")
        print(line)
        self.lines.append(line)

    def ok(self, name: str, detail: str = "") -> None:
        self._add("OK", name, detail)

    def warn(self, name: str, detail: str = "") -> None:
        self._add("WARN", name, detail)

    def fail(self, name: str, detail: str = "") -> None:
        self._add("FAIL", name, detail)
        self.failures += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control-repo", default=None,
                    help="dobot_control 仓库根目录（import dobot_control 用）")
    ap.add_argument("--with-robot", action="store_true",
                    help="额外做机械臂只读冒烟（需要机械臂已上电）")
    ap.add_argument("--report", default=None,
                    help="报告输出路径（默认 robot_pc_check_<时间>.txt）")
    args = ap.parse_args()

    rep = Reporter()
    import socket

    print("=" * 60)
    print("工控机环境检查  host=%s  %s" % (
        socket.gethostname(),
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    print("=" * 60)

    # ---------------- A. hardware / network ----------------
    print("\n--- A. 硬件 / 网络 ---")
    rc, out = _run(["nvidia-smi"])
    if rc == 0:
        gpu_line = next(
            (
                l.strip()
                for l in out.splitlines()
                if "NVIDIA" in l and "Driver Version" not in l and "|" in l
            ),
            "",
        )
        rep.ok("GPU (nvidia-smi)", gpu_line[:80] or "detected")
    else:
        rep.fail("GPU (nvidia-smi)", "未检测到 GPU —— 25Hz 推理的前提")

    for ip, side in (("192.168.5.1", "左臂"), ("192.168.5.2", "右臂")):
        rc, _ = _run(["ping", "-c", "2", "-W", "1", ip])
        if rc == 0:
            rep.ok(f"网络 {side} {ip}", "ping 通")
        else:
            rep.fail(f"网络 {side} {ip}", "ping 不通")

    devs = [f"/dev/{p}" for p in ("ttyUSB0", "ttyUSB1", "ttyACM0", "ttyACM1")]
    present = [p for p in devs if pathlib.Path(p).exists()]
    if present:
        rep.ok("串口", ", ".join(present))
        if not {"ttyUSB0", "ttyUSB1"}.issubset({p.rsplit("/", 1)[-1] for p in present}):
            rep.warn("夹爪串口", "缺少 ttyUSB0/1（夹爪）")
        if not {"ttyACM0", "ttyACM1"}.issubset({p.rsplit("/", 1)[-1] for p in present}):
            rep.warn("主手串口", "缺少 ttyACM0/1（主手）")
    else:
        rep.fail("串口", "ttyUSB*/ttyACM* 均缺失")

    rc, out = _run(["lsusb"])
    n_rs = out.lower().count("realsense") if rc == 0 else 0
    if n_rs >= 3:
        rep.ok("RealSense 相机", f"{n_rs} 个")
    else:
        rep.fail("RealSense 相机", f"只检测到 {n_rs} 个（需要 3）")

    # ---------------- B. Python 环境 ----------------
    print("\n--- B. Python 环境 ---")
    if args.control_repo:
        sys.path.insert(0, str(pathlib.Path(args.control_repo)))
    try:
        import dobot_control  # noqa: F401

        rep.ok("import dobot_control", "OK")
    except Exception as e:  # noqa: BLE001
        rep.fail("import dobot_control", repr(e)[:120])

    try:
        import jax

        devices = jax.devices()
        has_gpu = any(d.platform == "gpu" for d in devices)
        detail = f"jax {jax.__version__}, devices={devices}"
        if has_gpu:
            rep.ok("jax / GPU", detail)
        else:
            rep.fail("jax / GPU", detail + " —— 没有 GPU 设备")
    except Exception as e:  # noqa: BLE001
        rep.fail("jax / GPU", repr(e)[:120])

    try:
        import openpi_rtc  # noqa: F401

        rep.ok("import openpi_rtc", "OK")
    except Exception as e:  # noqa: BLE001
        rep.fail("import openpi_rtc", repr(e)[:120])

    # ---------------- C. 低风险冒烟 ----------------
    print("\n--- C. 低风险冒烟 ---")
    try:
        from dobot_control.cameras.realsense_camera import get_device_ids

        ids = get_device_ids()
        rep.ok("相机枚举", str(ids))
        expected = {"218622271430", "218622270365", "218622276272"}
        if not expected.issubset(set(ids)):
            rep.warn("相机 serial", f"与 ini 配置不完全一致: {ids}")
    except Exception as e:  # noqa: BLE001
        rep.fail("相机枚举", repr(e)[:120])

    if args.with_robot:
        for ip, side in (("192.168.5.1", "左臂"), ("192.168.5.2", "右臂")):
            try:
                from dobot_control.robots.dobot import DobotRobot

                r = DobotRobot(robot_ip=ip)
                joints = r.get_joint_state()
                rep.ok(f"读取 {side} 关节", f"shape={joints.shape}, 前3={joints[:3]}")
            except Exception as e:  # noqa: BLE001
                rep.fail(f"读取 {side} 关节", repr(e)[:120])
    else:
        print("(未加 --with-robot，跳过机械臂只读冒烟；确认上电后重跑)")

    # ---------------- summary ----------------
    print("\n" + "=" * 60)
    if rep.failures == 0:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print(f"RESULT: {rep.failures} FAILED (见上)")
    print("=" * 60)

    report_path = args.report or (
        f"robot_pc_check_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
    )
    with open(report_path, "w") as f:
        f.write("\n".join(rep.lines) + "\n")
    print(f"\n报告已保存到: {pathlib.Path(report_path).resolve()}")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
