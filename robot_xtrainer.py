#!/usr/bin/env python3
"""Xtrainer (Dobot Nova dual-arm) robot adapter for openpi_rtc.

Implements the four callbacks that ``run_robot.py`` needs, wired to the
control scripts found on the robot PC (``dobot_control/``) and verified
against the task_00031_yulong training data:

* State/action ordering is **LEFT-FIRST**: [left 6 joints, left gripper,
  right 6 joints, right gripper] (14,). This matches the raw HDF5 (dims 0-6
  = left arm parked, dims 7-13 = right arm moving) and ``real_env.py``. The
  convert script's motor labels say "right-first" but that is a mislabel —
  do NOT swap arms.
* Joints are absolute positions in RADIANS; ``DobotRobot.command_joint_state``
  converts to degrees internally (ServoJ, 30 ms cycle).
* Gripper is normalized 0..1 with **1 = open, 0 = closed** (verified: demo
  first frames ~0.986 open, mid-episode 0.0 while holding the tube).
  ``command_joint_state`` already maps it to the servo (x255).
* Cameras: three RealSense streams; the top view is cropped [150:420,
  220:480] and resized to 640x480 (mirrors ``run_control.py``).

Prerequisites on the robot PC: Dobot arms powered + on 192.168.5.1 / .2,
RealSense serials as in ``scripts/dobot_config/dobot_settings.ini``, and the
``dobot_control`` package importable (run from that repo root, or install it).

Run:
    uv run python run_robot.py \
        --checkpoint <dir> --mode rtc \
        --robot openpi_rtc.robot_xtrainer:XtrainerRobot
"""

from __future__ import annotations

import time

import numpy as np

DEFAULT_PROMPT = "Transfer the test tube from the right rack to the left rack."

# From scripts/dobot_config/dobot_settings.ini [CAMERA] / [GRIPPER_*].
CAMERA_SERIALS = {
    "top": "218622271430",
    "left": "218622270365",
    "right": "218622276272",
}
ROBOT_IPS = {"left": "192.168.5.1", "right": "192.168.5.2"}
# Reset poses from manipulate_utils.robot_pose_init (degrees -> radians,
# LEFT-FIRST, last element is the gripper command 0..1-ish).
RESET_LEFT = np.deg2rad([-90, 30, -110, 20, 90, 90, 20])
RESET_RIGHT = np.deg2rad([90, -30, 110, -20, -90, -90, 20])


class XtrainerRobot:
    """Adapter: direct DobotRobot connections + RealSense cameras."""

    def __init__(
        self,
        prompt: str = DEFAULT_PROMPT,
        episode_timeout_s: float = 60.0,
        safety_enabled: bool = True,
        robot_type: str = "Nova 2",
    ):
        from dobot_control.cameras.realsense_camera import RealSenseCamera
        from dobot_control.robots.dobot import DobotRobot
        from openpi_rtc.safety import SafetyConfig

        self.prompt = prompt
        self.episode_timeout_s = episode_timeout_s
        self._episode_start = time.monotonic()
        self._last_action = None
        self._safety = SafetyConfig(enabled=safety_enabled, robot_type=robot_type)

        # Slave arms (direct connection, same as examples/xtrainer_real).
        self.robot_l = DobotRobot(robot_ip=ROBOT_IPS["left"])   # LEFT arm
        self.robot_r = DobotRobot(robot_ip=ROBOT_IPS["right"])  # RIGHT arm

        # Cameras (flip flags from run_control.py: top/right flip=True,
        # left flip=False).
        self.cam_top = RealSenseCamera(flip=True, device_id=CAMERA_SERIALS["top"])
        self.cam_left = RealSenseCamera(flip=False, device_id=CAMERA_SERIALS["left"])
        self.cam_right = RealSenseCamera(flip=True, device_id=CAMERA_SERIALS["right"])

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def _arm_state(self, robot) -> np.ndarray:
        """[6 joints rad, gripper 0..1] with the REAL gripper reading."""
        joints = np.asarray(robot.get_observations()["joint_positions"], dtype=np.float32)
        joints = joints.copy()
        # get_joint_state appends a constant 1.0; read the actual servo
        # (external range 0..255 -> normalized, 1 = open).
        joints[6] = robot.gripper.get_current_position() / 255.0
        return joints

    def _camera_frame(self, cam, *, top: bool) -> np.ndarray:
        image, _ = cam.read()
        if top:
            image = image[150:420, 220:480]
            import cv2

            image = cv2.resize(image, (640, 480))
        return image[:, :, ::-1]  # RGB -> BGR, matches saved JPEGs

    def get_observation(self) -> dict:
        """Openpi observation dict, LEFT-FIRST (matches training data)."""
        left = self._arm_state(self.robot_l)
        right = self._arm_state(self.robot_r)
        state = np.concatenate([left, right]).astype(np.float32)

        images = {
            "cam_high": self._camera_frame(self.cam_top, top=True),
            "cam_left_wrist": self._camera_frame(self.cam_left, top=False),
            "cam_right_wrist": self._camera_frame(self.cam_right, top=False),
        }
        return {"state": state, "images": images, "prompt": self.prompt}

    # ------------------------------------------------------------------
    # action execution
    # ------------------------------------------------------------------
    def execute_action(self, action_np: np.ndarray) -> None:
        """Send one absolute joint-position action (14,), LEFT-FIRST."""
        from openpi_rtc.safety import check_action

        action = np.asarray(action_np, dtype=np.float32)
        assert action.shape == (14,), action.shape
        # Defensive gripper clip (1 = open, 0 = closed).
        action[6] = float(np.clip(action[6], 0.0, 1.0))
        action[13] = float(np.clip(action[13], 0.0, 1.0))
        # Safety checks before anything is sent to the arms.
        check_action(action, self._last_action, self._safety)
        self._last_action = action.copy()
        # command_joint_state takes [6 joints rad, gripper 0..1] per arm and
        # converts internally (rad->deg ServoJ, gripper x255).
        self.robot_l.command_joint_state(action[:7])
        self.robot_r.command_joint_state(action[7:])

    # ------------------------------------------------------------------
    # episode control
    # ------------------------------------------------------------------
    def _interpolate_to(self, target: np.ndarray) -> None:
        """Step both arms smoothly to a 14-dim target (mirrors robot_pose_init)."""
        from dobot_control.env import Rate

        curr = np.concatenate(
            [self._arm_state(self.robot_l), self._arm_state(self.robot_r)]
        )
        max_delta = float(np.abs(curr - target).max())
        steps = max(1, min(int(max_delta / 0.01), 100))
        rate = Rate(25.0)
        for jnt in np.linspace(curr, target, steps):
            self.execute_action(jnt)
            rate.sleep()

    def reset_episode(self) -> None:
        """Home both arms, then wait for the scene to be reset manually."""
        target = np.concatenate([RESET_LEFT, RESET_RIGHT]).astype(np.float32)
        self._interpolate_to(target)
        time.sleep(1.0)
        # TODO: place the test tube back in the right rack (manual), then
        # confirm the scene matches the demo initial frames.
        self._episode_start = time.monotonic()

    def episode_done(self) -> bool:
        """Timeout fallback; wire success/failure detection or a manual stop."""
        return time.monotonic() - self._episode_start > self.episode_timeout_s
