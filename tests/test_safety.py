"""CPU tests for the ported robot safety checks."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from openpi_rtc.safety import SafetyConfig, check_action


def reset_action() -> np.ndarray:
    left = np.deg2rad([-90, 30, -110, 20, 90, 90, 20])
    right = np.deg2rad([90, -30, 110, -20, -90, -90, 20])
    return np.concatenate([left, right]).astype(np.float32)


def test_valid_action_passes():
    cfg = SafetyConfig(enabled=True)
    check_action(reset_action(), None, cfg)  # must not raise
    print("valid action passes OK")


def test_nan_rejected():
    a = reset_action()
    a[3] = np.nan
    try:
        check_action(a, None, SafetyConfig())
        raise AssertionError("NaN should be rejected")
    except ValueError:
        pass
    print("NaN rejected OK")


def test_joint_safety():
    a = reset_action()
    a[2] = 0.1  # left J3 must be < 0
    try:
        check_action(a, None, SafetyConfig())
        raise AssertionError("left J3 violation should be rejected")
    except ValueError:
        pass
    b = reset_action()
    b[9] = -0.1  # right J3 must be > 0
    try:
        check_action(b, None, SafetyConfig())
        raise AssertionError("right J3 violation should be rejected")
    except ValueError:
        pass
    print("joint safety OK")


def test_delta_limit_and_wrap():
    cfg = SafetyConfig(enabled=True, check_delta=True, max_delta_rad=0.9)
    last = reset_action()
    big = last.copy()
    big[1] += 1.5  # too large a jump
    try:
        check_action(big, last, cfg)
        raise AssertionError("big delta should be rejected")
    except ValueError:
        pass
    wrap = last.copy()
    wrap[1] = wrap[1] + 2 * np.pi  # same physical angle
    check_action(wrap, last, cfg)  # 2*pi wrap must pass
    grip = last.copy()
    grip[13] = 0.1  # gripper jumps freely
    check_action(grip, last, cfg)
    print("delta limit + 2pi wrap OK")


def test_pose_protection():
    cfg = SafetyConfig(enabled=True, check_pose=True, robot_type="Nova 2")
    # A wildly wrong action should leave the safe zone.
    bad = reset_action()
    bad[:6] = 0.0
    try:
        check_action(bad, None, cfg)
        raise AssertionError("pose protection should reject out-of-zone action")
    except ValueError:
        pass
    print("pose protection OK")


if __name__ == "__main__":
    test_valid_action_passes()
    test_nan_rejected()
    test_joint_safety()
    test_delta_limit_and_wrap()
    test_pose_protection()
    print("safety tests passed")
