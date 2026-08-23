"""
Real environment for Galaxea R1 Pro robot.

This environment provides an interface for controlling the R1 Pro robot in real-world scenarios.
"""

import collections
import time
from typing import Optional, List, Dict
import dm_env
import numpy as np

from .robot_interface.interfaces import R1ProInterface
from .gripper.galaxea_g1 import GalaxeaR1ProGripper
from . import constants


# Default reset position for the robot
DEFAULT_RESET_POSITION = {
    "left_arm": constants.DEFAULT_LEFT_ARM_RESET,
    "right_arm": constants.DEFAULT_RIGHT_ARM_RESET,
    "torso": constants.DEFAULT_TORSO_RESET,
}


class RealEnv:
    """
    Environment for real R1 Pro robot bi-manual manipulation.

    Action space:      [left_arm_qpos (7),             # absolute joint position
                        left_gripper_position (1),      # normalized gripper position (0: open, 1: closed)
                        right_arm_qpos (7),             # absolute joint position
                        right_gripper_position (1)]     # normalized gripper position (0: open, 1: closed)
                        Total: 16 dimensions

    Observation space: {"qpos": Concat[ left_arm_qpos (7),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position
                                        right_arm_qpos (7),         # absolute joint position
                                        right_gripper_position (1)] # normalized gripper position
                        "qvel": Concat[ left_arm_qvel (7),          # absolute joint velocity
                                        left_gripper_velocity (1),  # normalized gripper velocity
                                        right_arm_qvel (7),         # absolute joint velocity
                                        right_gripper_qvel (1)]     # normalized gripper velocity
                        "images": {"head": (H, W, 3),               # RGB image
                                   "left_wrist": (H, W, 3),         # RGB image
                                   "right_wrist": (H, W, 3)}        # RGB image
                        "base_pose": 6D pose of the mobile base (x, y, z, roll, pitch, yaw)
                        "base_velocity": 3D velocity (vx, vy, w)
    """

    def __init__(
        self,
        init_node: bool = True,
        reset_position: Optional[Dict[str, np.ndarray]] = None,
        setup_robot: bool = True,
        enable_cameras: bool = True,
        enable_torso: bool = False,
        enable_mobile_base: bool = True,
    ):
        """
        Initialize the R1 Pro real environment.

        Args:
            init_node: Whether to initialize ROS node
            reset_position: Dictionary with keys 'left_arm', 'right_arm', 'torso' for reset positions
            setup_robot: Whether to setup the robot (grippers, etc.)
            enable_cameras: Whether to enable camera observations
            enable_torso: Whether to enable torso control
            enable_mobile_base: Whether to enable mobile base control
        """
        self._reset_position = reset_position if reset_position else DEFAULT_RESET_POSITION
        self._enable_torso = enable_torso
        self._enable_mobile_base = enable_mobile_base

        # Initialize grippers
        left_gripper = GalaxeaR1ProGripper(
            left_or_right="left",
            gripper_close_stroke=constants.GRIPPER_CLOSE_STROKE,
            gripper_open_stroke=constants.GRIPPER_OPEN_STROKE,
        )
        right_gripper = GalaxeaR1ProGripper(
            left_or_right="right",
            gripper_close_stroke=constants.GRIPPER_CLOSE_STROKE,
            gripper_open_stroke=constants.GRIPPER_OPEN_STROKE,
        )

        # Initialize R1 Pro interface
        self._robot = R1ProInterface(
            left_arm_joint_state_topic=constants.LEFT_ARM_JOINT_STATE_TOPIC,
            left_arm_joint_target_position_topic=constants.LEFT_ARM_JOINT_TARGET_TOPIC,
            left_gripper=left_gripper,
            right_arm_joint_state_topic=constants.RIGHT_ARM_JOINT_STATE_TOPIC,
            right_arm_joint_target_position_topic=constants.RIGHT_ARM_JOINT_TARGET_TOPIC,
            right_gripper=right_gripper,
            torso_joint_state_topic=constants.TORSO_JOINT_STATE_TOPIC,
            torso_joint_target_position_topic=constants.TORSO_JOINT_TARGET_TOPIC,
            odometry_topic=constants.ODOMETRY_TOPIC,
            mobile_base_vel_cmd_topic=constants.MOBILE_BASE_CMD_TOPIC,
            mobile_base_state_topic=constants.MOBILE_BASE_STATE_TOPIC,
            enable_rgbd=enable_cameras,
            rgb_topics= None,
            depth_topics= None,
            publisher_node_name="r1pro_real_env_publisher" if init_node else None,
        )

        self._enable_cameras = enable_cameras
        
        # Wait for robot to be ready
        time.sleep(1.0)

    def _reset_arms(self):
        """Reset arms to default position."""
        left_arm_reset = self._reset_position.get("left_arm", constants.DEFAULT_LEFT_ARM_RESET)
        right_arm_reset = self._reset_position.get("right_arm", constants.DEFAULT_RIGHT_ARM_RESET)

        self._robot.control(
            arm_controller="joint_position",
            arm_cmd={
                "left": left_arm_reset,
                "right": right_arm_reset,
            },
            torso_controller="joint_position",
            torso_cmd=self._reset_position.get("torso", constants.DEFAULT_TORSO_RESET) if self._enable_torso else None,
        )

        # Wait for arms to reach position
        time.sleep(2.0)

    def _reset_grippers(self):
        """Reset grippers: close then open."""
        # Close grippers
        self._robot.control(
            arm_controller="joint_position",
            arm_cmd={
                "left": None,
                "right": None,
            },
            gripper_cmd={
                "left": 1.0,  # closed
                "right": 1.0,  # closed
            },
        )
        time.sleep(1.0)

        # Open grippers
        self._robot.control(
            arm_controller="joint_position",
            arm_cmd={
                "left": None,
                "right": None,
            },
            gripper_cmd={
                "left": 0.0,  # open
                "right": 0.0,  # open
            },
        )
        time.sleep(0.5)

    def get_qpos(self) -> np.ndarray:
        """
        Get current joint positions.

        Returns:
            np.ndarray: [left_arm (7), left_gripper (1), right_arm (7), right_gripper (1)]
        """
        joint_positions = self._robot.last_joint_position
        gripper_states = self._robot.last_gripper_state

        left_arm_qpos = joint_positions["left_arm"]
        right_arm_qpos = joint_positions["right_arm"]

        # Get gripper positions (normalized)
        if gripper_states["left_gripper"] is not None:
            left_gripper_pos = gripper_states["left_gripper"]["gripper_position"]
            left_gripper_normalized = np.array([constants.GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)])
        else:
            left_gripper_normalized = np.array([0.0])

        if gripper_states["right_gripper"] is not None:
            right_gripper_pos = gripper_states["right_gripper"]["gripper_position"]
            right_gripper_normalized = np.array([constants.GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)])
        else:
            right_gripper_normalized = np.array([0.0])
        print(left_arm_qpos.shape, left_gripper_normalized.shape)
        return np.concatenate([
            left_arm_qpos,
            left_gripper_normalized,
            right_arm_qpos,
            right_gripper_normalized
        ])
    def get_qpos_mobile(self) -> np.ndarray:
        """
        获取当前状态（19维），适配 R1Pro 格式。
        
        Returns:
            np.ndarray: [chassis(3), left_arm(7), right_arm(7), left_gripper(1), right_gripper(1)]
        """
        # 1. 获取各个组件的原始数据
        joint_positions = self._robot.last_joint_position
        gripper_states = self._robot.last_gripper_state
        # 获取底盘位姿 (假设返回 [x, y, yaw])
        # 如果你的机器人接口不同，请修改此处
        
        chassis_vel = self._robot.last_mobile_velocity["mobile_base"]
        
        left_arm_qpos = joint_positions["left_arm"]    # (7,)
        right_arm_qpos = joint_positions["right_arm"]  # (7,)

        # 2. 处理左夹爪位置 (归一化)
        if gripper_states.get("left_gripper") is not None:
            left_gripper_pos = gripper_states["left_gripper"]["gripper_position"]
            left_gripper_normalized = np.array([constants.GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)])
        else:
            left_gripper_normalized = np.array([0.0])

        # 3. 处理右夹爪位置 (归一化)
        if gripper_states.get("right_gripper") is not None:
            right_gripper_pos = gripper_states["right_gripper"]["gripper_position"]
            right_gripper_normalized = np.array([constants.GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)])
        else:
            right_gripper_normalized = np.array([0.0])
       
        # 4. 严格按照 R1Pro 19维顺序拼接
        # [0:3] 底盘, [3:10] 左臂, [10:17] 右臂, [17] 左爪, [18] 右爪
        full_qpos = np.concatenate([
            chassis_vel,              # 3维
            left_arm_qpos,            # 7维
            right_arm_qpos,           # 7维
            left_gripper_normalized,  # 1维
            right_gripper_normalized   # 1维
        ])

        # 调试打印，确保总长度为 19
        print(f"qpos shape: {full_qpos.shape}") 
        return full_qpos
    def get_qvel(self) -> np.ndarray:
        """
        Get current joint velocities.

        Returns:
            np.ndarray: [left_arm (7), left_gripper (1), right_arm (7), right_gripper (1)]
        """
        return np.zeros(16)

    def get_images(self) -> Dict[str, np.ndarray]:
        """
        Get current camera images.

        Returns:
            Dict[str, np.ndarray]: Dictionary of camera images
        """
        if not self._enable_cameras:
            return {}

        rgb_data = self._robot.last_rgb
        # print(rgb_data.keys())
        if rgb_data is None:
            return {}

        images = {}
        for cam_name in constants.CAMERA_NAMES:
        
            if cam_name in rgb_data and rgb_data[cam_name] is not None:
                images[cam_name] = rgb_data[cam_name]["img"]

        return images

    def get_base_pose(self) -> np.ndarray:
        """Get mobile base pose (x, y, z, roll, pitch, yaw)."""
        if not self._enable_mobile_base:
            return np.zeros(6)
        return self._robot.curr_base_pose

    def get_base_velocity(self) -> np.ndarray:
        """Get mobile base velocity (vx, vy, w)."""
        if not self._enable_mobile_base:
            return np.zeros(3)
        return self._robot.curr_base_velocity

    def get_observation(self) -> collections.OrderedDict:
        """
        Get current observation.

        Returns:
            OrderedDict with keys: 'qpos', 'qvel', 'images', 'base_pose', 'base_velocity'
        """
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        obs["qvel"] = self.get_qvel()
        obs["images"] = self.get_images()
        print(obs["images"].keys())

        if self._enable_mobile_base:
            obs["base_pose"] = self.get_base_pose()
            obs["base_velocity"] = self.get_base_velocity()

        return obs
    def get_observation_mobile(self) -> collections.OrderedDict:
        """
        Get current observation.

        Returns:
            OrderedDict with keys: 'qpos', 'qvel', 'images', 'base_pose', 'base_velocity'
        """
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos_mobile()
        obs["qvel"] = self.get_qvel()
        obs["images"] = self.get_images()
        print(obs["images"].keys())

        if self._enable_mobile_base:
            obs["base_pose"] = self.get_base_pose()
            obs["base_velocity"] = self.get_base_velocity()

        return obs
    def get_reward(self) -> float:
        """Get reward (always 0 for real robot)."""
        return 0.0

    def reset(self, *, fake: bool = False) -> dm_env.TimeStep:
        """
        Reset the environment.

        Args:
            fake: If True, skip actual robot reset

        Returns:
            dm_env.TimeStep: Initial timestep
        """
        if not fake:
            self._reset_arms()
            self._reset_grippers()

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation()
        )

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """
        Execute one step in the environment.

        Args:
            action: np.ndarray of shape (16,) containing:
                    [left_arm (7), left_gripper (1), right_arm (7), right_gripper (1)]

        Returns:
            dm_env.TimeStep: Timestep after action execution
        """
        assert len(action) == 16, f"Expected action of length 16, got {len(action)}"

        # Parse action
        left_arm_action = action[:7]
        left_gripper_action = action[7]
        right_arm_action = action[8:15]
        right_gripper_action = action[15]

        # Execute action
        self._robot.control(
            arm_controller="joint_position",
            arm_cmd={
                "left": left_arm_action,
                "right": right_arm_action,
            },
            gripper_cmd={
                "left": float(left_gripper_action),
                "right": float(right_gripper_action),
            },
            torso_controller="joint_position",
            torso_cmd=None,  # Don't control torso during episodes
            base_cmd=None,   # Don't control base during episodes
        )

        # Sleep for control timestep
        time.sleep(constants.DT)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation()
        )
    def step_mobile(self, action: np.ndarray) -> dm_env.TimeStep:
        """
        Execute one step in the environment.

        Args:
            action: np.ndarray of shape (19,) containing:
                    [mobile_action(3),left_arm (7), right_arm (7),left_gripper (1), right_gripper (1)]

        Returns:
            dm_env.TimeStep: Timestep after action execution
        """
        assert len(action) == 19, f"Expected action of length 19, got {len(action)}"

       # parse action
        mobile_action = action[:3]
        left_arm_action = action[3:10]
        right_arm_action = action[10:17] 
        left_gripper_action = action[17]
        right_gripper_action = action[18]

        # Execute action
        self._robot.control(
            arm_controller="joint_position",
            arm_cmd={
                "left": left_arm_action,
                "right": right_arm_action,
            },
            gripper_cmd={
                "left": float(left_gripper_action),
                "right": float(right_gripper_action),
            },
            torso_controller="joint_position",
            torso_cmd=None,  # Don't control torso during episodes
            base_cmd=mobile_action,   # Don't control base during episodes
        )

        # Sleep for control timestep
        time.sleep(constants.DT)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation()
        )
    def close(self):
        """Close the environment and cleanup resources."""
        self._robot.close()


def make_real_env(
    init_node: bool = True,
    reset_position: Optional[Dict[str, np.ndarray]] = None,
    setup_robot: bool = True,
    enable_cameras: bool = True,
    enable_torso: bool = False,
    enable_mobile_base: bool = False,
) -> RealEnv:
    """
    Factory function to create R1 Pro real environment.

    Args:
        init_node: Whether to initialize ROS node
        reset_position: Dictionary with reset positions for 'left_arm', 'right_arm', 'torso'
        setup_robot: Whether to setup the robot
        enable_cameras: Whether to enable camera observations
        enable_torso: Whether to enable torso control
        enable_mobile_base: Whether to enable mobile base control

    Returns:
        R1ProRealEnv instance
    """
    return RealEnv(
        init_node=init_node,
        reset_position=reset_position,
        setup_robot=setup_robot,
        enable_cameras=enable_cameras,
        enable_torso=enable_torso,
        enable_mobile_base=enable_mobile_base,
    )
