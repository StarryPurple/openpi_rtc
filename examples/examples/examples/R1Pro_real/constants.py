"""
Constants for Galaxea R1 Pro robot.

This file defines the robot-specific constants including:
- Joint names and limits
- Gripper parameters
- Control parameters
"""

import numpy as np

# Control parameters
DT = 0.01  # Control loop timestep (100Hz)

# Joint names for R1 Pro arms (7 DOF each)
LEFT_ARM_JOINT_NAMES = [
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_joint7",
]

RIGHT_ARM_JOINT_NAMES = [
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
]

TORSO_JOINT_NAMES = [
    "torso_joint1",
    "torso_joint2",
    "torso_joint3",
    "torso_joint4",
]

# Joint limits (from R1ProKinematics and R1ProInterface)
TORSO_JOINT_HIGH = np.array([1.8326, 2.5307, 1.8326, 3.0543])
TORSO_JOINT_LOW = np.array([-1.1345, -2.7925, -2.0944, -3.0543])

LEFT_ARM_JOINT_HIGH = np.array([2.8798, 3.2289, 1.0, 0.0, 2.8798, 1.6581, 2.8798])
LEFT_ARM_JOINT_LOW = np.array([-2.8798, 0.0, -1.0, -3.3161, -2.8798, -1.6581, -2.8798])

RIGHT_ARM_JOINT_HIGH = np.array([2.8798, 3.2289, 1.0, 0.0, 2.8798, 1.6581, 2.8798])
RIGHT_ARM_JOINT_LOW = np.array([-2.8798, 0.0, -1.0, -3.3161, -2.8798, -1.6581, -2.8798])

# Default reset positions (neutral pose)
DEFAULT_LEFT_ARM_RESET = np.array([0.0, 1.5, 0.0, -1.5, 0.0, 0.0, 0.0])
DEFAULT_RIGHT_ARM_RESET = np.array([0.0, 1.5, 0.0, -1.5, 0.0, 0.0, 0.0])
DEFAULT_TORSO_RESET = np.array([0.0, 0.0, 0.0, 0.0])

# Gripper parameters (for Galaxea R1 Pro gripper)
# Gripper stroke range: 10.0 (closed) to 90.0 (open)
GRIPPER_CLOSE_STROKE = 10.0
GRIPPER_OPEN_STROKE = 90.0

# Gripper normalization functions
# Normalized gripper position: 0.0 (open) to 1.0 (closed)
GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - GRIPPER_OPEN_STROKE) / (
    GRIPPER_CLOSE_STROKE - GRIPPER_OPEN_STROKE
)

GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: (
    x * (GRIPPER_CLOSE_STROKE - GRIPPER_OPEN_STROKE) + GRIPPER_OPEN_STROKE
)

# Camera names
CAMERA_NAMES = [
    "head_left",
    "head_right",
    "left_wrist",
    "right_wrist",
]

# ROS topics (default values, can be overridden in real_env.py)
LEFT_ARM_JOINT_STATE_TOPIC = "/hdas/feedback_arm_left"
LEFT_ARM_JOINT_TARGET_TOPIC = "/motion_target/target_joint_state_arm_left"
RIGHT_ARM_JOINT_STATE_TOPIC = "/hdas/feedback_arm_right"
RIGHT_ARM_JOINT_TARGET_TOPIC = "/motion_target/target_joint_state_arm_right"

TORSO_JOINT_STATE_TOPIC = "/hdas/feedback_torso"
TORSO_JOINT_TARGET_TOPIC = "/motion_target/target_joint_state_torso"

LEFT_GRIPPER_CONTROL_TOPIC = "/motion_target/target_position_gripper_left"
LEFT_GRIPPER_FEEDBACK_TOPIC = "/hdas/feedback_gripper_left"
RIGHT_GRIPPER_CONTROL_TOPIC = "/motion_target/target_position_gripper_right"
RIGHT_GRIPPER_FEEDBACK_TOPIC = "/hdas/feedback_gripper_right"

ODOMETRY_TOPIC = "/camera/odom/sample"
MOBILE_BASE_CMD_TOPIC = "/motion_target/target_speed_chassis"
MOBILE_BASE_STATE_TOPIC = "/motion_control/chassis_speed"
# Default RGB topics
RGB_TOPICS = {
    "head": "/zed_multi_cams/zed2_head/zed_nodelet_head/rgb/image_rect_color",
    "left_wrist": "/zed_multi_cams/zed2_left_wrist/zed_nodelet_left_wrist/rgb/image_rect_color",
    "right_wrist": "/zed_multi_cams/zed2_right_wrist/zed_nodelet_right_wrist/rgb/image_rect_color",
}

# Default depth topics
DEPTH_TOPICS = {
    "head": "/zed_multi_cams/zed2_head/zed_nodelet_head/depth/depth_registered",
    "left_wrist": "/zed_multi_cams/zed2_left_wrist/zed_nodelet_left_wrist/depth/depth_registered",
    "right_wrist": "/zed_multi_cams/zed2_right_wrist/zed_nodelet_right_wrist/depth/depth_registered",
}
