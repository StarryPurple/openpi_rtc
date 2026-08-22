#!/usr/bin/env python3
"""Safety checks for policy-driven execution on the XTrainer platform.

Checks applied per action, in order:
  1. finite values + gripper clipped to [0, 1];
  2. joint safety: left J3 < 0, right J3 > 0 (the platform's convention);
  3. per-step servo delta limit (with pi-wrap), default 0.9 rad/tick;
  4. FK pose protection (working-zone + TCP Z-speed), on by default.
     ``robot_type`` must match the actual unit (Nova 2 / Nova 5) or the
     forward-kinematics working zone will be wrong.

Any violation raises ValueError -> run_robot.py aborts the episode (fail
fast). Thresholds are configurable in :class:`SafetyConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyConfig:
    enabled: bool = True
    check_joints: bool = True
    check_delta: bool = True
    max_delta_rad: float = 0.9  # run_control's servo_action_check default
    check_pose: bool = True     # FK working-zone + TCP Z-speed protection
    robot_type: str = "Nova 2"  # must match the real unit (Nova 2 / Nova 5)
    total_time_s: float = 0.04  # 25 Hz control period


# ---------------------------------------------------------------------------
# ported helpers (run_control.py)
# ---------------------------------------------------------------------------
def dh_transformation_matrix(theta, d, a, alpha):
    theta = np.asarray(theta, dtype=np.float64)
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ]
    )


def forward_kinematics(q0, q1, q2, q3, q4, q5, y, r_type):
    if r_type == "Nova 2":
        dh = [
            (q0, 0.2234, 0, np.pi / 2),
            (q1 - np.pi / 2, 0, -0.280, 0),
            (q2, 0, -0.225, 0),
            (q3 - np.pi / 2, 0.1175, 0, np.pi / 2),
            (q4, 0.120, 0, -np.pi / 2),
            (q5, 0.088, 0, 0),
        ]
    elif r_type == "Nova 5":
        dh = [
            (q0, 0.240, 0, np.pi / 2),
            (q1 - np.pi / 2, 0, -0.400, 0),
            (q2, 0, -0.330, 0),
            (q3 - np.pi / 2, 0.135, 0, np.pi / 2),
            (q4, 0.120, 0, -np.pi / 2),
            (q5, 0.088, 0, 0),
        ]
    else:
        raise ValueError(f"unknown robot type: {r_type}")
    t = np.eye(4)
    for params in dh:
        t = np.dot(t, dh_transformation_matrix(*params))
    t_tool = np.eye(4)
    t_tool[:3, 3] = np.array([0, y, 0.2])
    return np.dot(t, t_tool)[:3, 3]


def claw_width(coef):
    claw_servo = 2.3818 - coef * 1.5401
    cos_claw_servo = np.cos(claw_servo)
    return 0.03 * cos_claw_servo + 0.5 * np.sqrt(
        0.0036 * cos_claw_servo**2 + 0.0028
    )


def calculate_vel_pos(action, last_action, total_time, r_type):
    action = np.asarray(action, dtype=np.float64)
    claw_left = claw_width(action[6])
    claw_right = claw_width(action[13])
    positions, vel = {}, {}
    if last_action is None:
        vel = None
    for side, claw in (("left", claw_left), ("right", claw_right)):
        for paw, coef in (("left", 1), ("right", -1)):
            c = claw * coef
            joints = action[0:6] if side == "left" else action[7:13]
            positions[f"{side}_{paw}"] = forward_kinematics(*joints, c, r_type)
            if vel is not None:
                last_action = np.asarray(last_action, dtype=np.float64)
                last_joints = last_action[0:6] if side == "left" else last_action[7:13]
                last_pos = forward_kinematics(*last_joints, c, r_type)
                vel[f"{side}_{paw}"] = (positions[f"{side}_{paw}"] - last_pos) / total_time
    return positions, vel


def is_within_safe_position(position, x_range, y_range, z_min):
    return (
        x_range[0] <= position[0] <= x_range[1]
        and y_range[0] <= position[1] <= y_range[1]
        and position[2] > z_min
    )


def check_pose_protection(action, last_action, config: SafetyConfig):
    """Working-zone + TCP Z-speed check (FK with ``config.robot_type``)."""
    positions, vel = calculate_vel_pos(
        action, last_action, config.total_time_s, config.robot_type
    )
    warnings = []
    zones = {
        "left": ((-450, 300), (-750, -160), 20),
        "right": ((-250, 450), (-750, -160), 20),
    }
    for side, (xr, yr, zmin) in zones.items():
        for paw in ("left", "right"):
            pos_mm = positions[f"{side}_{paw}"] * 1000
            if not is_within_safe_position(pos_mm, xr, yr, zmin):
                warnings.append(f"{side}_{paw} out of safe zone: {pos_mm}")
        if vel is not None:
            if vel[f"{side}_left"][2] < -2.5 or vel[f"{side}_right"][2] < -2.5:
                warnings.append(f"{side} arm TCP moving too fast in Z")
    if warnings:
        raise ValueError("pose protection: " + "; ".join(warnings))


def check_joint_safety(action):
    action = np.asarray(action, dtype=np.float64)
    if not (action[2] < 0):
        raise ValueError(f"left J3 out of safe position: {action[2]:.4f}")
    if not (action[9] > 0):
        raise ValueError(f"right J3 out of safe position: {action[9]:.4f}")


def servo_action_check(action, last_action, step_len=0.9):
    action = np.asarray(action, dtype=np.float64)
    if last_action is None:
        return
    last_action = np.asarray(last_action, dtype=np.float64)
    dev = action - last_action
    for j in range(14):
        if j in (6, 13):  # grippers move freely
            continue
        if abs(dev[j]) > step_len:
            # allow a 2*pi wrap (angle periodicity)
            pi_2_cal = dev[j] / np.pi
            if 1.85 < abs(pi_2_cal) < 2.15:
                continue
            raise ValueError(
                f"servo delta too big at joint {j}: {dev[j]:.3f} rad "
                f"(limit {step_len})"
            )


def check_action(action, last_action, config: SafetyConfig) -> None:
    """Run all enabled safety checks; raises ValueError on violation."""
    if not config.enabled:
        return
    action = np.asarray(action, dtype=np.float32)
    if not np.all(np.isfinite(action)):
        raise ValueError("action contains NaN/Inf")
    if config.check_joints:
        check_joint_safety(action)
    if config.check_delta:
        servo_action_check(action, last_action, config.max_delta_rad)
    if config.check_pose:
        check_pose_protection(action, last_action, config)
