# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
import collections
import time
from typing import Optional, List
import dm_env
import numpy as np
from examples.xtrainer_real import constants
from examples.xtrainer_real.robots.dobot import DobotRobot
from examples.xtrainer_real.cameras.camera_front import ImageRecorder
import threading

# This is the reset position that is used by the standard Xtrainer runtime.
DEFAULT_RESET_POSITION = {"left",  (-1.57, 0, -1.57, 0, 1.57, 1.57), "right", (1.57, 0, 1.57, 0, -1.57, -1.57)}


class RealEnv:
    def __init__(
        self,
        init_node,
        *,
        reset_position: Optional[List[float]] = None,
        setup_robots: bool = True,
        arms: str = "both",
        no_gripper: bool = False,
    ):
        # reset_position = START_ARM_POSE[:6]
        self._reset_position = reset_position[:6] if reset_position else DEFAULT_RESET_POSITION
        self.arms = arms

        self._robot_l = DobotRobot(robot_ip="192.168.5.1", no_gripper=no_gripper)
        self._robot_r = DobotRobot(robot_ip="192.168.5.2", no_gripper=no_gripper)
        self.image_recorder = ImageRecorder()

    def setup_robots(self):
        print("no need")

    def get_qpos(self):
        left_qpos_raw = self._robot_l.get_observations()
        right_qpos_raw = self._robot_r.get_observations()
        return np.concatenate([left_qpos_raw, right_qpos_raw])

    def get_qvel(self):
        return np.array([0, 0, 0, 0])

    def get_effort(self):
        return [0, 0]

    def get_images(self):
        return self.image_recorder.get_images()

    def set_gripper_pose(self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized):
        print("useless")

    def _reset_joints(self):
        print("useless")

    def _reset_gripper(self):
        print("useless")

    def get_observation(self):
        obs = {}
        obs["qpos"] = self.get_qpos()
        # obs["qvel"] = self.get_qvel()
        # obs["effort"] = self.get_effort()
        obs["images"] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def reset(self, *, fake=False):
        print("useless")

    def step(self, action, single_arm=True):
        if self.arms == "left" and single_arm:
            self._robot_l.command_joint_state(action[:7])
        elif self.arms == "right" and single_arm:
            self._robot_r.command_joint_state(action[7:])
        else:
            self._robot_l.command_joint_state(action[:7])
            self._robot_r.command_joint_state(action[7:])
        # return dm_env.TimeStep(
        #     step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        # )

    def step_movj(self, action):
        if self.arms == "left":
            self._robot_l.command_joint_state_movj(action[:7])
        elif self.arms == "right":
            self._robot_r.command_joint_state_movj(action[7:])
        else:
            self._robot_l.command_joint_state_movj(action[:7])
            self._robot_r.command_joint_state_movj(action[7:])

    def step_gripper(self, action):
        if self.arms == "left":
            self._robot_l.command_joint_state_gripper(action[:7])
        elif self.arms == "right":
            self._robot_r.command_joint_state_gripper(action[7:])
        else:
            self._robot_l.command_joint_state_gripper(action[:7])
            self._robot_r.command_joint_state_gripper(action[7:])
        # return dm_env.TimeStep(
        #     step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        # )


# def get_action(master_bot_left, master_bot_right):
#     action = np.zeros(14)  # 6 joint + 1 gripper, for two arms
#     # Arm actions
#     action[:6] = master_bot_left.dxl.joint_states.position[:6]
#     action[7 : 7 + 6] = master_bot_right.dxl.joint_states.position[:6]
#     # Gripper actions
#     action[6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
#     action[7 + 6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])
#
#     return action


def make_real_env(init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True) -> RealEnv:
    return RealEnv(init_node, reset_position=reset_position, setup_robots=setup_robots)
