from typing import Literal, Optional, Dict, Union
from functools import partial
# import sys
# sys.path.append("..")
import rospy
from geometry_msgs.msg import TwistStamped,Twist
from sensor_msgs.msg import JointState, PointCloud2, Image, CompressedImage
# from hdas_msg.msg import MotorControl
import ros_numpy
import numpy as np
from cv_bridge import CvBridge
import cv2
from .utils import *
from ..kinematics import R1Kinematics, R1ProKinematics
from ..gripper.base import BaseGripper
from ..mobile_base import Odom


class R1Interface:
    torso_joint_high = np.array([1.8326, 2.5307, 1.8326, 3.0543])
    torso_joint_low = np.array([-1.1345, -2.7925, -2.0944, -3.0543])
    left_arm_joint_high = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
    left_arm_joint_low = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])
    right_arm_joint_high = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
    right_arm_joint_low = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])

    def __init__(
        self,
        *,
        # ====== left arm ======
        left_arm_joint_state_topic: str = "/hdas/feedback_arm_left",
        left_arm_joint_target_position_topic: str = "/motion_target/target_joint_state_arm_left",
        left_gripper: Optional[BaseGripper] = None,
        # ====== right arm ======
        right_arm_joint_state_topic: str = "/hdas/feedback_arm_right",
        right_arm_joint_target_position_topic: str = "/motion_target/target_joint_state_arm_right",
        right_gripper: Optional[BaseGripper] = None,
        # ====== torso ======
        torso_joint_state_topic: str = "/hdas/feedback_torso",
        torso_joint_target_position_topic: str = "/motion_target/target_joint_state_torso",
        # ====== mobile base ======
        odometry_topic: str = "/camera/odom/sample",
        wait_for_first_odom_msg: bool = False,
        mobile_base_vel_cmd_topic: str = "/motion_target/target_speed_chassis",
        mobile_base_cmd_threshold: Union[np.ndarray, float] = np.array(
            [0.01, 0.01, 0.05]
        ),
        mobile_base_cmd_limit: Union[np.ndarray, float] = np.array([0.3, 0.3, 0.4]),
        # ====== cameras ======
        enable_pointcloud: bool = False,
        pointcloud_topic: str = "/r1_jetson/fused_pcd",
        enable_rgbd: bool = False,
        rgb_topics: Optional[Dict[str, str]] = None,
        depth_topics: Optional[Dict[str, str]] = None,
        # ====== common ======
        publisher_node_name: str = "joint_state_publisher",
        publisher_node_queue_size: int = 10,
        state_buffer_size: int = 1000,
        arm_joint_control_step_interval: float = 0.4,
        torso_joint_control_step_interval: float = 0.1,
        control_freq: float = 100.0,
        on_arm_cmd_out_of_range: Literal["raise", "clip"] = "clip",
        on_torso_cmd_out_of_range: Literal["raise", "clip"] = "clip",
    ):
        self._kin_model = R1Kinematics()

        self._left_arm_joint_state_buffer = None
        self._right_arm_joint_state_buffer = None
        self._torso_joint_state_buffer = None
        self._state_buffer_size = state_buffer_size
        self._pcd = None
        self._rgb = None
        self._depth = None

        # ros node initialization
        rospy.init_node(publisher_node_name, anonymous=True)
        self._rate = rospy.Rate(control_freq)

        # command publishers
        self._left_arm_joint_target_position_pub = rospy.Publisher(
            left_arm_joint_target_position_topic,
            JointState,
            queue_size=publisher_node_queue_size,
        )
        self._right_arm_joint_target_position_pub = rospy.Publisher(
            right_arm_joint_target_position_topic,
            JointState,
            queue_size=publisher_node_queue_size,
        )
        self._torso_joint_target_position_pub = rospy.Publisher(
            torso_joint_target_position_topic,
            JointState,
            queue_size=publisher_node_queue_size,
        )

        # sensor subscribers
        self._left_arm_joint_state_sub = rospy.Subscriber(
            left_arm_joint_state_topic,
            JointState,
            self._left_arm_state_callback,
        )
        self._right_arm_joint_state_sub = rospy.Subscriber(
            right_arm_joint_state_topic,
            JointState,
            self._right_arm_state_callback,
        )
        self._torso_joint_state_sub = rospy.Subscriber(
            torso_joint_state_topic,
            JointState,
            self._torso_state_callback,
        )
        
        if enable_pointcloud:
            self._pcd_sub = rospy.Subscriber(
                pointcloud_topic,
                PointCloud2,
                self._pcd_callback,
            )
        if enable_rgbd:
            rgb_topics = rgb_topics or {
                "head": "/zed_multi_cams/zed2_head/zed_nodelet_head/rgb/image_rect_color",
                "left_wrist": "/zed_multi_cams/zed2_left_wrist/zed_nodelet_left_wrist/rgb/image_rect_color",
                "right_wrist": "/zed_multi_cams/zed2_right_wrist/zed_nodelet_right_wrist/rgb/image_rect_color",
            }
            depth_topics = depth_topics or {
                "head": "/zed_multi_cams/zed2_head/zed_nodelet_head/depth/depth_registered",
                "left_wrist": "/zed_multi_cams/zed2_left_wrist/zed_nodelet_left_wrist/depth/depth_registered",
                "right_wrist": "/zed_multi_cams/zed2_right_wrist/zed_nodelet_right_wrist/depth/depth_registered",
            }
            self._rgb = {k: None for k in rgb_topics}
            self._depth = {k: None for k in depth_topics}
            self._cv_bridge = CvBridge()
            self._rgb_subs = {
                k: rospy.Subscriber(v, CompressedImage, partial(self._rgb_callback, name=k))
                for k, v in rgb_topics.items()
            }
            self._depth_subs = {
                k: rospy.Subscriber(v, Image, partial(self._depth_callback, name=k))
                for k, v in depth_topics.items()
            }
        else:
            self._cv_bridge = self._rgb_subs = self._depth_subs = None

        # mobile base
        self._odom = Odom(
            odom_topic=odometry_topic,
            T_odom2base=self._kin_model.T_odom2base,
            wait_for_first_msg=wait_for_first_odom_msg,
        )
        if isinstance(mobile_base_cmd_threshold, float):
            mobile_base_cmd_threshold = np.array(
                [
                    mobile_base_cmd_threshold,
                ]
                * 3
            )
        if isinstance(mobile_base_cmd_limit, float):
            mobile_base_cmd_limit = np.array(
                [
                    mobile_base_cmd_limit,
                ]
                * 3
            )
        assert mobile_base_cmd_threshold.shape == mobile_base_cmd_limit.shape == (3,)
        self._mobile_base_cmd_threshold = mobile_base_cmd_threshold
        self._mobile_base_cmd_limit = mobile_base_cmd_limit
        self._mobile_base_vel_cmd_pub = rospy.Publisher(
            mobile_base_vel_cmd_topic, TwistStamped, queue_size=publisher_node_queue_size
        )

        self._left_gripper, self._right_gripper = left_gripper, right_gripper
        if self._left_gripper is not None:
            self._left_gripper.init_hook()
        if self._right_gripper is not None:
            self._right_gripper.init_hook()

        assert on_arm_cmd_out_of_range in ["raise", "clip"]
        self._on_arm_cmd_out_of_range = on_arm_cmd_out_of_range
        assert on_torso_cmd_out_of_range in ["raise", "clip"]
        self._on_torso_cmd_out_of_range = on_torso_cmd_out_of_range

        self._arm_joint_control_step_interval = arm_joint_control_step_interval
        self._torso_joint_control_step_interval = torso_joint_control_step_interval

    def control(
        self,
        *,
        arm_controller: Literal["joint_position"] = "joint_position",
        arm_cmd: Dict[str, Optional[np.ndarray]],
        gripper_cmd: Optional[Dict[str, float]] = None,
        torso_controller: Literal["joint_position"] = "joint_position",
        torso_cmd: Optional[np.ndarray] = None,
        base_cmd: Optional[np.ndarray] = None,
    ):
        assert (
            arm_controller == "joint_position"
        ), f"Invalid arm controller {arm_controller}"
        assert (
            torso_controller == "joint_position"
        ), f"Invalid torso controller {torso_controller}"
        assert "left" in arm_cmd and "right" in arm_cmd
        if gripper_cmd is not None:
            assert "left" in gripper_cmd and "right" in gripper_cmd

        # base control
        base_cmd = np.zeros((3,)) if base_cmd is None else base_cmd
        try:
            self._mobile_base_control(base_cmd)
        except rospy.ROSInterruptException:
            pass

        # upper-body control
        try:
            self._upper_body_joint_position_control(
                left_arm_target_q=arm_cmd["left"],
                right_arm_target_q=arm_cmd["right"],
                torso_target_q=torso_cmd,
            )
        except rospy.ROSInterruptException:
            pass

        # optional gripper control
        if gripper_cmd is not None:
            try:
                self._gripper_control(gripper_cmd)
            except rospy.ROSInterruptException:
                pass

    def _mobile_base_control(self, cmd: np.ndarray):
        set_zero = np.abs(cmd) < self._mobile_base_cmd_threshold
        cmd[set_zero] = 0
        cmd = np.clip(cmd, -self._mobile_base_cmd_limit, self._mobile_base_cmd_limit)
        _cmd = TwistStamped()
        _cmd.linear.x = cmd[0]
        _cmd.linear.y = cmd[1]
        _cmd.angular.z = cmd[2]
        self._mobile_base_vel_cmd_pub.publish(_cmd)

    def stop_mobile_base(self):
        self._mobile_base_vel_cmd_pub.publish(TwistStamped())

    def _upper_body_joint_position_control(
        self,
        left_arm_target_q: Optional[np.ndarray] = None,
        right_arm_target_q: Optional[np.ndarray] = None,
        torso_target_q: Optional[np.ndarray] = None,
    ):
        # shape check
        if left_arm_target_q is not None:
            assert left_arm_target_q.shape == (
                6,
            ), f"Expected left_arm_target_q to have shape (6,), got {left_arm_target_q.shape}"
        if right_arm_target_q is not None:
            assert right_arm_target_q.shape == (
                6,
            ), f"Expected right_arm_target_q to have shape (6,), got {right_arm_target_q.shape}"
        if torso_target_q is not None:
            assert torso_target_q.shape == (
                4,
            ), f"Expected torso_target_q to have shape (4,), got {torso_target_q.shape}"

        # value out-of-range check
        if left_arm_target_q is not None:
            left_in_range = np.logical_and(
                left_arm_target_q >= self.left_arm_joint_low,
                left_arm_target_q <= self.left_arm_joint_high,
            )
            left_invalid_joint_idxs = np.where(~left_in_range)[0]
            if len(left_invalid_joint_idxs) > 0:
                for idx in left_invalid_joint_idxs:
                    msg = (
                        f"Left arm joint {idx + 1} target position {left_arm_target_q[idx]} is out of range "
                        f"[{self.left_arm_joint_low[idx]}, {self.left_arm_joint_high[idx]}]."
                    )
                    if self._on_arm_cmd_out_of_range == "clip":
                        msg += " Clipping to range."
                        left_arm_target_q[idx] = np.clip(
                            left_arm_target_q[idx],
                            self.left_arm_joint_low[idx],
                            self.left_arm_joint_high[idx],
                        )
                    else:
                        raise ValueError(msg)
                    rospy.logwarn(msg)

        if right_arm_target_q is not None:
            right_in_range = np.logical_and(
                right_arm_target_q >= self.right_arm_joint_low,
                right_arm_target_q <= self.right_arm_joint_high,
            )
            right_invalid_joint_idxs = np.where(~right_in_range)[0]
            if len(right_invalid_joint_idxs) > 0:
                for idx in right_invalid_joint_idxs:
                    msg = (
                        f"Right arm joint {idx + 1} target position {right_arm_target_q[idx]} is out of range "
                        f"[{self.right_arm_joint_low[idx]}, {self.right_arm_joint_high[idx]}]."
                    )
                    if self._on_arm_cmd_out_of_range == "clip":
                        msg += " Clipping to range."
                        right_arm_target_q[idx] = np.clip(
                            right_arm_target_q[idx],
                            self.right_arm_joint_low[idx],
                            self.right_arm_joint_high[idx],
                        )
                    else:
                        raise ValueError(msg)
                    rospy.logwarn(msg)

        if torso_target_q is not None:
            torso_in_range = np.logical_and(
                torso_target_q >= self.torso_joint_low,
                torso_target_q <= self.torso_joint_high,
            )
            torso_invalid_joint_idxs = np.where(~torso_in_range)[0]
            if len(torso_invalid_joint_idxs) > 0:
                for idx in torso_invalid_joint_idxs:
                    msg = (
                        f"Torso joint {idx + 1} target position {torso_target_q[idx]} is out of range "
                        f"[{self.torso_joint_low[idx]}, {self.torso_joint_high[idx]}]."
                    )
                    if self._on_torso_cmd_out_of_range == "clip":
                        msg += " Clipping to range."
                        torso_target_q[idx] = np.clip(
                            torso_target_q[idx],
                            self.torso_joint_low[idx],
                            self.torso_joint_high[idx],
                        )
                        rospy.logwarn(msg)
                    else:
                        raise ValueError(msg)

        # compute # steps for cmds
        last_two_arms_joint_positions = self.last_joint_position
        left_arm_joint_state = JointState()
        left_arm_joint_state.position = last_two_arms_joint_positions["left_arm"]
        left_arm_n_steps = (
            int(
                np.ceil(
                    np.max(
                        np.abs(
                            (
                                left_arm_target_q
                                - np.array(left_arm_joint_state.position)
                            )
                            / self._arm_joint_control_step_interval
                        )
                    )
                )
            )
            if left_arm_target_q is not None
            else 0
        )
        right_arm_joint_state = JointState()
        right_arm_joint_state.position = last_two_arms_joint_positions["right_arm"]
        right_arm_n_steps = (
            int(
                np.ceil(
                    np.max(
                        np.abs(
                            (
                                right_arm_target_q
                                - np.array(right_arm_joint_state.position)
                            )
                            / self._arm_joint_control_step_interval
                        )
                    )
                )
            )
            if right_arm_target_q is not None
            else 0
        )
        last_torso_joint_positions = self.last_joint_position["torso"]
        torso_joint_state = JointState()
        torso_joint_state.position = last_torso_joint_positions
        torso_n_steps = (
            int(
                np.ceil(
                    np.max(
                        np.abs(
                            (torso_target_q - np.array(torso_joint_state.position))
                            / self._torso_joint_control_step_interval
                        )
                    )
                )
            )
            if torso_target_q is not None
            else 0
        )

        # control loop
        max_n_steps = max(left_arm_n_steps, right_arm_n_steps, torso_n_steps)
        if max_n_steps > 1:
            if left_arm_target_q is not None:
                left_arm_increment = (
                    (left_arm_target_q - np.array(left_arm_joint_state.position))
                    / (left_arm_n_steps - 1)
                    if left_arm_n_steps > 1
                    else (left_arm_target_q - np.array(left_arm_joint_state.position))
                )
            else:
                left_arm_increment = np.zeros(6)
            if right_arm_target_q is not None:
                right_arm_increment = (
                    (right_arm_target_q - np.array(right_arm_joint_state.position))
                    / (right_arm_n_steps - 1)
                    if right_arm_n_steps > 1
                    else (right_arm_target_q - np.array(right_arm_joint_state.position))
                )
            else:
                right_arm_increment = np.zeros(6)
            if torso_target_q is not None:
                torso_increment = (
                    (torso_target_q - np.array(torso_joint_state.position))
                    / (torso_n_steps - 1)
                    if torso_n_steps > 1
                    else (torso_target_q - np.array(torso_joint_state.position))
                )
            else:
                torso_increment = np.zeros(4)

            for step in range(max_n_steps - 1):
                if left_arm_target_q is not None:
                    left_arm_joint_state.header.stamp = rospy.Time.now()
                    if step <= left_arm_n_steps - 1:
                        left_arm_joint_state.position = (
                            np.array(left_arm_joint_state.position) + left_arm_increment
                        ).tolist()
                    else:
                        left_arm_joint_state.position = left_arm_target_q.tolist()
                    self._left_arm_joint_target_position_pub.publish(
                        left_arm_joint_state
                    )
                if right_arm_target_q is not None:
                    right_arm_joint_state.header.stamp = rospy.Time.now()
                    if step <= right_arm_n_steps - 1:
                        right_arm_joint_state.position = (
                            np.array(right_arm_joint_state.position)
                            + right_arm_increment
                        ).tolist()
                    else:
                        right_arm_joint_state.position = right_arm_target_q.tolist()
                    self._right_arm_joint_target_position_pub.publish(
                        right_arm_joint_state
                    )
                if torso_target_q is not None:
                    torso_joint_state.header.stamp = rospy.Time.now()
                    if step <= torso_n_steps - 1:
                        torso_joint_state.position = (
                            np.array(torso_joint_state.position) + torso_increment
                        ).tolist()
                    else:
                        torso_joint_state.position = torso_target_q.tolist()
                self._torso_joint_target_position_pub.publish(torso_joint_state)
                self._rate.sleep()

        # ensure the last position is the target position
        if left_arm_target_q is not None:
            left_arm_joint_state.header.stamp = rospy.Time.now()
            left_arm_joint_state.position = left_arm_target_q.tolist()
            self._left_arm_joint_target_position_pub.publish(left_arm_joint_state)
        if right_arm_target_q is not None:
            right_arm_joint_state.header.stamp = rospy.Time.now()
            right_arm_joint_state.position = right_arm_target_q.tolist()
            self._right_arm_joint_target_position_pub.publish(right_arm_joint_state)
        if torso_target_q is not None:
            torso_joint_state.header.stamp = rospy.Time.now()
            torso_joint_state.position = torso_target_q.tolist()
            self._torso_joint_target_position_pub.publish(torso_joint_state)
        if not (
            left_arm_target_q is None
            and right_arm_target_q is None
            and torso_target_q is None
        ):
            self._rate.sleep()

    def _gripper_control(self, gripper_action: Dict[str, float]):
        left_gripper_action = gripper_action["left"]
        right_gripper_action = gripper_action["right"]
        assert (
            1 >= left_gripper_action >= 0
        ), "Invalid left gripper action, must be between 0 and 1"
        assert (
            1 >= right_gripper_action >= 0
        ), "Invalid right gripper action, must be between 0 and 1"

        if self._left_gripper is not None:
            self._left_gripper.act(left_gripper_action)
        if self._right_gripper is not None:
            self._right_gripper.act(right_gripper_action)

    def _pcd_callback(self, pcd: PointCloud2):
        timestamp = pcd.header.stamp.secs + pcd.header.stamp.nsecs * 1e-9
        pcd = ros_numpy.point_cloud2.pointcloud2_to_array(pcd)
        pcd_xyz, pcd_xyz_mask = get_xyz_points(pcd, remove_nans=True)
        pcd = ros_numpy.point_cloud2.split_rgb_field(pcd)
        pcd_rgb = np.zeros(pcd.shape + (3,), dtype=np.uint8)
        pcd_rgb[..., 0] = pcd["r"]
        pcd_rgb[..., 1] = pcd["g"]
        pcd_rgb[..., 2] = pcd["b"]
        pcd_rgb = pcd_rgb[pcd_xyz_mask]
        self._pcd = {
            "xyz": pcd_xyz,
            "rgb": pcd_rgb,
            "stamp": timestamp,
        }

    def _rgb_callback(self, rgb_msg: Image, name: str):
        timestamp = rgb_msg.header.stamp.secs + rgb_msg.header.stamp.nsecs * 1e-9
        img = np.asarray(self._cv_bridge.imgmsg_to_cv2(rgb_msg, "rgb8"))
        self._rgb[name] = {
            "img": img,
            "stamp": timestamp,
        }

    def _depth_callback(self, depth_msg: Image, name: str):
        timestamp = depth_msg.header.stamp.secs + depth_msg.header.stamp.nsecs * 1e-9
        depth_image = self._cv_bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        self._depth[name] = {
            "depth": depth_image,
            "stamp": timestamp,
        }

    def _left_arm_state_callback(self, data: JointState):
        new_state = {
            "joint_position": np.clip(
                np.array([data.position[:7]]),
                self.left_arm_joint_low,
                self.left_arm_joint_high,
            ),
            "joint_velocity": np.array([data.velocity][:7]),
            "joint_effort": np.array([data.effort][:7]),
            "seq": np.array([data.header.seq]),
            "stamp": np.array(
                [data.header.stamp.secs + data.header.stamp.nsecs * 1e-9]
            ),
        }

        if self._left_arm_joint_state_buffer is None:
            self._left_arm_joint_state_buffer = new_state
        else:
            self._left_arm_joint_state_buffer = any_concat(
                [
                    self._left_arm_joint_state_buffer,
                    new_state,
                ],
                dim=0,
            )
            self._left_arm_joint_state_buffer = any_slice(
                self._left_arm_joint_state_buffer, np.s_[-self._state_buffer_size :]
            )

    def _right_arm_state_callback(self, data: JointState):
        print(1)
        new_state = {
            "joint_position": np.clip(
                np.array([data.position[:7]]),
                self.right_arm_joint_low,
                self.right_arm_joint_high,
            ),
            "joint_velocity": np.array([data.velocity][:7]),
            "joint_effort": np.array([data.effort][:7]),
            "seq": np.array([data.header.seq]),
            "stamp": np.array(
                [data.header.stamp.secs + data.header.stamp.nsecs * 1e-9]
            ),
        }

        

        if self._right_arm_joint_state_buffer is None:
            self._right_arm_joint_state_buffer = new_state
        else:
            self._right_arm_joint_state_buffer = any_concat(
                [
                    self._right_arm_joint_state_buffer,
                    new_state,
                ],
                dim=0,
            )
            self._right_arm_joint_state_buffer = any_slice(
                self._right_arm_joint_state_buffer, np.s_[-self._state_buffer_size :]
            )

    def _torso_state_callback(self, data: JointState):
        new_state = {
            "joint_position": np.clip(
                np.array([data.position[:4]]),
                self.torso_joint_low,
                self.torso_joint_high,
            ),
            "joint_velocity": np.array([data.velocity][:4]),
            "joint_effort": np.array([data.effort][:4]),
            "seq": np.array([data.header.seq]),
            "stamp": np.array(
                [data.header.stamp.secs + data.header.stamp.nsecs * 1e-9]
            ),
        }

        if self._torso_joint_state_buffer is None:
            self._torso_joint_state_buffer = new_state
        else:
            self._torso_joint_state_buffer = any_concat(
                [
                    self._torso_joint_state_buffer,
                    new_state,
                ],
                dim=0,
            )
            self._torso_joint_state_buffer = any_slice(
                self._torso_joint_state_buffer, np.s_[-self._state_buffer_size :]
            )
   
    def close(self):
        # stop the base
        self.stop_mobile_base()
        if self._left_gripper is not None:
            self._left_gripper.close()
        if self._right_gripper is not None:
            self._right_gripper.close()
        rospy.signal_shutdown("Shutting down node")

    def __del__(self):
        self.close()

    @property
    def last_joint_position(self) -> Dict[str, np.ndarray]:
        return {
            "left_arm": any_slice(self._left_arm_joint_state_buffer, -1)[
                "joint_position"
            ],
            "right_arm": any_slice(self._right_arm_joint_state_buffer, -1)[
                "joint_position"
            ],
            "torso": any_slice(self._torso_joint_state_buffer, -1)["joint_position"],
        }
    
   

    @property
    def last_gripper_state(self):
        return {
            "left_gripper": (
                None
                if self._left_gripper is None
                else any_slice(self._left_gripper.state_buffer, -1)
            ),
            "right_gripper": (
                None
                if self._right_gripper is None
                else any_slice(self._right_gripper.state_buffer, -1)
            ),
        }

    @property
    def last_pointcloud(self):
        return {
            "xyz": self._pcd["xyz"],
            "rgb": self._pcd["rgb"],
            "stamp": self._pcd["stamp"],
        }

    @property
    def last_rgb(self):
        return (
            {
                k: {
                    "img": v["img"],
                    "stamp": v["stamp"],
                }
                for k, v in self._rgb.items()
            }
            if self._rgb is not None
            else None
        )

    @property
    def last_depth(self):
        return (
            {
                k: {
                    "depth": v["depth"],
                    "stamp": v["stamp"],
                }
                for k, v in self._depth.items()
            }
            if self._depth is not None
            else None
        )

    @property
    def curr_base_pose(self):
        return self._odom.curr_base_pose

    @property
    def curr_base_position(self):
        return self._odom.curr_base_position

    @property
    def curr_base_orientation(self):
        return self._odom.curr_base_orientation

    @property
    def curr_base_velocity(self):
        return self._odom.curr_base_velocity


class R1ProInterface:
    torso_joint_high = np.array([1.8326, 2.5307, 1.8326, 3.0543])
    torso_joint_low = np.array([-1.1345, -2.7925, -2.0944, -3.0543])
    left_arm_joint_high = np.array([2.8798, 3.2289, 1, 0, 2.8798, 1.6581, 2.8798])
    left_arm_joint_low = np.array([-2.8798, -3, -1,  -3.3161, -2.8798, -1.6581, -2.8798])
    right_arm_joint_high = np.array([2.8798, 3.2289,1,  0, 2.8798, 1.6581, 2.8798])
    right_arm_joint_low = np.array([-2.8798, -3, -1, -3.3161, -2.8798, -1.6581, -2.8798])

    def __init__(
        self,
        *,
        # ====== left arm ======
        left_arm_joint_state_topic: str = "/hdas/feedback_arm_left",
        left_arm_joint_target_position_topic: str = "/motion_target/target_joint_state_arm_left",
        left_gripper: Optional[BaseGripper] = None,
        # ====== right arm ======
        right_arm_joint_state_topic: str = "/hdas/feedback_arm_right",
        right_arm_joint_target_position_topic: str = "/motion_target/target_joint_state_arm_right",
        right_gripper: Optional[BaseGripper] = None,
        # ====== torso ======
        torso_joint_state_topic: str = "/hdas/feedback_torso",
        torso_joint_target_position_topic: str = "/motion_target/target_joint_state_torso",
        # ====== mobile base ======
        odometry_topic: str = "/camera/odom/sample",
        wait_for_first_odom_msg: bool = False,
        mobile_base_state_topic: str = "/motion_control/chassis_speed",
        mobile_base_vel_cmd_topic: str = "/motion_target/target_speed_chassis",
        mobile_base_cmd_threshold: Union[np.ndarray, float] = np.array(
            [0.01, 0.01, 0.05]
        ),
        mobile_base_cmd_limit: Union[np.ndarray, float] = np.array([0.3, 0.3, 0.4]),
        # ====== cameras ======
        enable_pointcloud: bool = False,
        # ----------------------------------------
        pointcloud_topic: str = "/r1_jetson/fused_pcd",
        enable_rgbd: bool = False,
        rgb_topics: Optional[Dict[str, str]] = None,
        depth_topics: Optional[Dict[str, str]] = None,
        # ====== common ======
        publisher_node_name: str = "joint_state_publisher",
        publisher_node_queue_size: int = 10,
        state_buffer_size: int = 1000,
        arm_joint_control_step_interval: float = 0.4,
        torso_joint_control_step_interval: float = 0.1,
        control_freq: float = 100.0,
        on_arm_cmd_out_of_range: Literal["raise", "clip"] = "clip",
        on_torso_cmd_out_of_range: Literal["raise", "clip"] = "clip",
    ):
        print("Initializing R1ProInterface")
        self._kin_model = R1ProKinematics()

        self._left_arm_joint_state_buffer = None
        self._right_arm_joint_state_buffer = None
        self._torso_joint_state_buffer = None
        self._mobile_base_state_buffer = None
        
        self._state_buffer_size = state_buffer_size
        self._pcd = None
        self._rgb = None
        self._depth = None

        # ros node initialization
        rospy.init_node(publisher_node_name, anonymous=True)
        self._rate = rospy.Rate(control_freq)

        # command publishers
        self._left_arm_joint_target_position_pub = rospy.Publisher(
            left_arm_joint_target_position_topic,
            JointState,
            queue_size=publisher_node_queue_size,
        )
        self._right_arm_joint_target_position_pub = rospy.Publisher(
            right_arm_joint_target_position_topic,
            JointState,
            queue_size=publisher_node_queue_size,
        )
        self._torso_joint_target_position_pub = rospy.Publisher(
            torso_joint_target_position_topic,
            JointState,
            queue_size=publisher_node_queue_size,
        )
        # sensor subscribers
        self._left_arm_joint_state_sub = rospy.Subscriber(
            left_arm_joint_state_topic,
            JointState,
            self._left_arm_state_callback,
        )
        self._right_arm_joint_state_sub = rospy.Subscriber(
            right_arm_joint_state_topic,
            JointState,
            self._right_arm_state_callback,
        )
        self._torso_joint_state_sub = rospy.Subscriber(
            torso_joint_state_topic,
            JointState,
            self._torso_state_callback,
        )
        self._mobile_base_state_sub = rospy.Subscriber(
            mobile_base_state_topic,
            TwistStamped,
            self._mobile_state_callback,
        )
       
        if enable_pointcloud:
            self._pcd_sub = rospy.Subscriber(
                pointcloud_topic,
                PointCloud2,
                self._pcd_callback,
            )
        if enable_rgbd:
            rgb_topics = rgb_topics or {
                "head_left": "/hdas/camera_head/left_raw/image_raw_color/compressed",
                "head_right": "/hdas/camera_head/right_raw/image_raw_color/compressed",

                "left_wrist": "/hdas/camera_wrist_left/color/image_rect_raw/compressed",
                "right_wrist": "/hdas/camera_wrist_right/color/image_rect_raw/compressed",
            }
            depth_topics = depth_topics
            # or {
            #     "head": "/zed_multi_cams/zed2_head/zed_nodelet_head/depth/depth_registered",
            #     "left_wrist": "/hdas/camera_wrist_left/depth/image_rect_raw",
            #     "right_wrist": "/hdas/camera_wrist_right/depth/image_rect_raw",
            # }
            self._rgb = {k: None for k in rgb_topics}
            # self._depth = {k: None for k in depth_topics}
            self._cv_bridge = CvBridge()
            # print({k: v for k, v in rgb_topics.items()})
            self._rgb_subs = {
                k: rospy.Subscriber(
                    v, CompressedImage, partial(self.callback_head_image, name=k)
                )
                for k, v in rgb_topics.items()
            }
            # print(self._rgb_subs)
            # self._depth_subs = {
            #     k: rospy.Subscriber(v, Image, partial(self._depth_callback, name=k))
            #     for k, v in depth_topics.items()
            # }
        else:
            self._cv_bridge = self._rgb_subs = self._depth_subs = None

        # mobile base
        self._odom = Odom(
            odom_topic=odometry_topic,
            T_odom2base=self._kin_model.T_odom2base,
            wait_for_first_msg=wait_for_first_odom_msg,
        )
        if isinstance(mobile_base_cmd_threshold, float):
            mobile_base_cmd_threshold = np.array(
                [
                    mobile_base_cmd_threshold,
                ]
                * 3
            )
        if isinstance(mobile_base_cmd_limit, float):
            mobile_base_cmd_limit = np.array(
                [
                    mobile_base_cmd_limit,
                ]
                * 3
            )
        assert mobile_base_cmd_threshold.shape == mobile_base_cmd_limit.shape == (3,)
        self._mobile_base_cmd_threshold = mobile_base_cmd_threshold
        self._mobile_base_cmd_limit = mobile_base_cmd_limit
        self._mobile_base_vel_cmd_pub = rospy.Publisher(
            mobile_base_vel_cmd_topic, TwistStamped, queue_size=publisher_node_queue_size
        )

        self._left_gripper, self._right_gripper = left_gripper, right_gripper
        if self._left_gripper is not None:
            self._left_gripper.init_hook()
        if self._right_gripper is not None:
            self._right_gripper.init_hook()

        assert on_arm_cmd_out_of_range in ["raise", "clip"]
        self._on_arm_cmd_out_of_range = on_arm_cmd_out_of_range
        assert on_torso_cmd_out_of_range in ["raise", "clip"]
        self._on_torso_cmd_out_of_range = on_torso_cmd_out_of_range

        self._arm_joint_control_step_interval = arm_joint_control_step_interval
        self._torso_joint_control_step_interval = torso_joint_control_step_interval

    def control(
        self,
        *,
        arm_controller: Literal["joint_position"] = "joint_position",
        arm_cmd: Dict[str, Optional[np.ndarray]],
        gripper_cmd: Optional[Dict[str, float]] = None,
        torso_controller: Literal["joint_position"] = "joint_position",
        torso_cmd: Optional[np.ndarray] = None,
        base_cmd: Optional[np.ndarray] = None,
    ):
        assert (
            arm_controller == "joint_position"
        ), f"Invalid arm controller {arm_controller}"
        assert (
            torso_controller == "joint_position"
        ), f"Invalid torso controller {torso_controller}"
        assert "left" in arm_cmd and "right" in arm_cmd
        if gripper_cmd is not None:
            assert "left" in gripper_cmd and "right" in gripper_cmd

        # base control
        base_cmd = np.zeros((3,)) if base_cmd is None else base_cmd
        try:
            self._mobile_base_control(base_cmd)
        except rospy.ROSInterruptException:
            pass

        # upper-body control
        try:
            self._upper_body_joint_position_control(
                left_arm_target_q=arm_cmd["left"],
                right_arm_target_q=arm_cmd["right"],
                torso_target_q=torso_cmd,
            )
        except rospy.ROSInterruptException:
            pass

        # optional gripper control
        if gripper_cmd is not None:
            try:
                self._gripper_control(gripper_cmd)
            except rospy.ROSInterruptException:
                pass

    def _mobile_base_control(self, cmd: np.ndarray):
        set_zero = np.abs(cmd) < self._mobile_base_cmd_threshold
        cmd[set_zero] = 0
        cmd = np.clip(cmd, -self._mobile_base_cmd_limit, self._mobile_base_cmd_limit)
        _cmd = TwistStamped()
        _cmd.twist.linear.x = cmd[0]
        _cmd.twist.linear.y = cmd[1]
        _cmd.twist.angular.z = cmd[2]
        print(cmd[0], cmd[1], cmd[2])
        self._mobile_base_vel_cmd_pub.publish(_cmd)

    def stop_mobile_base(self):
        self._mobile_base_vel_cmd_pub.publish(TwistStamped())

    def _upper_body_joint_position_control(
        self,
        left_arm_target_q: Optional[np.ndarray] = None,
        right_arm_target_q: Optional[np.ndarray] = None,
        torso_target_q: Optional[np.ndarray] = None,
    ):
        # shape check, 7 for R1 Pro
        if left_arm_target_q is not None:
            assert left_arm_target_q.shape == (
                7,
            ), f"Expected left_arm_target_q to have shape (6,), got {left_arm_target_q.shape}"
        if right_arm_target_q is not None:
            assert right_arm_target_q.shape == (
                7,
            ), f"Expected right_arm_target_q to have shape (6,), got {right_arm_target_q.shape}"
        if torso_target_q is not None:
            assert torso_target_q.shape == (
                4,
            ), f"Expected torso_target_q to have shape (4,), got {torso_target_q.shape}"

        # value out-of-range check
        if left_arm_target_q is not None:
            left_in_range = np.logical_and(
                left_arm_target_q >= self.left_arm_joint_low,
                left_arm_target_q <= self.left_arm_joint_high,
            )
            left_invalid_joint_idxs = np.where(~left_in_range)[0]
            if len(left_invalid_joint_idxs) > 0:
                for idx in left_invalid_joint_idxs:
                    msg = (
                        f"Left arm joint {idx + 1} target position {left_arm_target_q[idx]} is out of range "
                        f"[{self.left_arm_joint_low[idx]}, {self.left_arm_joint_high[idx]}]."
                    )
                    if self._on_arm_cmd_out_of_range == "clip":
                        msg += " Clipping to range."
                        left_arm_target_q[idx] = np.clip(
                            left_arm_target_q[idx],
                            self.left_arm_joint_low[idx],
                            self.left_arm_joint_high[idx],
                        )
                    else:
                        raise ValueError(msg)
                    rospy.logwarn(msg)

        if right_arm_target_q is not None:
            right_in_range = np.logical_and(
                right_arm_target_q >= self.right_arm_joint_low,
                right_arm_target_q <= self.right_arm_joint_high,
            )
            right_invalid_joint_idxs = np.where(~right_in_range)[0]
            if len(right_invalid_joint_idxs) > 0:
                for idx in right_invalid_joint_idxs:
                    msg = (
                        f"Right arm joint {idx + 1} target position {right_arm_target_q[idx]} is out of range "
                        f"[{self.right_arm_joint_low[idx]}, {self.right_arm_joint_high[idx]}]."
                    )
                    if self._on_arm_cmd_out_of_range == "clip":
                        msg += " Clipping to range."
                        right_arm_target_q[idx] = np.clip(
                            right_arm_target_q[idx],
                            self.right_arm_joint_low[idx],
                            self.right_arm_joint_high[idx],
                        )
                    else:
                        raise ValueError(msg)
                    rospy.logwarn(msg)

        if torso_target_q is not None:
            torso_in_range = np.logical_and(
                torso_target_q >= self.torso_joint_low,
                torso_target_q <= self.torso_joint_high,
            )
            torso_invalid_joint_idxs = np.where(~torso_in_range)[0]
            if len(torso_invalid_joint_idxs) > 0:
                for idx in torso_invalid_joint_idxs:
                    msg = (
                        f"Torso joint {idx + 1} target position {torso_target_q[idx]} is out of range "
                        f"[{self.torso_joint_low[idx]}, {self.torso_joint_high[idx]}]."
                    )
                    if self._on_torso_cmd_out_of_range == "clip":
                        msg += " Clipping to range."
                        torso_target_q[idx] = np.clip(
                            torso_target_q[idx],
                            self.torso_joint_low[idx],
                            self.torso_joint_high[idx],
                        )
                        rospy.logwarn(msg)
                    else:
                        raise ValueError(msg)

        # compute # steps for cmds
        last_two_arms_joint_positions = self.last_joint_position
        left_arm_joint_state = JointState()
        left_arm_joint_state.position = last_two_arms_joint_positions["left_arm"]
        left_arm_n_steps = (
            int(
                np.ceil(
                    np.max(
                        np.abs(
                            (
                                left_arm_target_q
                                - np.array(left_arm_joint_state.position)
                            )
                            / self._arm_joint_control_step_interval
                        )
                    )
                )
            )
            if left_arm_target_q is not None
            else 0
        )
        right_arm_joint_state = JointState()
        right_arm_joint_state.position = last_two_arms_joint_positions["right_arm"]
        right_arm_n_steps = (
            int(
                np.ceil(
                    np.max(
                        np.abs(
                            (
                                right_arm_target_q
                                - np.array(right_arm_joint_state.position)
                            )
                            / self._arm_joint_control_step_interval
                        )
                    )
                )
            )
            if right_arm_target_q is not None
            else 0
        )
        last_torso_joint_positions = self.last_joint_position["torso"]
        torso_joint_state = JointState()
        torso_joint_state.position = last_torso_joint_positions
        torso_n_steps = (
            int(
                np.ceil(
                    np.max(
                        np.abs(
                            (torso_target_q - np.array(torso_joint_state.position))
                            / self._torso_joint_control_step_interval
                        )
                    )
                )
            )
            if torso_target_q is not None
            else 0
        )

        # control loop
        max_n_steps = max(left_arm_n_steps, right_arm_n_steps, torso_n_steps)
        if max_n_steps > 1:
            if left_arm_target_q is not None:
                left_arm_increment = (
                    (left_arm_target_q - np.array(left_arm_joint_state.position))
                    / (left_arm_n_steps - 1)
                    if left_arm_n_steps > 1
                    else (left_arm_target_q - np.array(left_arm_joint_state.position))
                )
            else:
                left_arm_increment = np.zeros(6)
            if right_arm_target_q is not None:
                right_arm_increment = (
                    (right_arm_target_q - np.array(right_arm_joint_state.position))
                    / (right_arm_n_steps - 1)
                    if right_arm_n_steps > 1
                    else (right_arm_target_q - np.array(right_arm_joint_state.position))
                )
            else:
                right_arm_increment = np.zeros(6)
            if torso_target_q is not None:
                torso_increment = (
                    (torso_target_q - np.array(torso_joint_state.position))
                    / (torso_n_steps - 1)
                    if torso_n_steps > 1
                    else (torso_target_q - np.array(torso_joint_state.position))
                )
            else:
                torso_increment = np.zeros(4)

            for step in range(max_n_steps - 1):
                if left_arm_target_q is not None:
                    left_arm_joint_state.header.stamp = rospy.Time.now()
                    if step <= left_arm_n_steps - 1:
                        left_arm_joint_state.position = (
                            np.array(left_arm_joint_state.position) + left_arm_increment
                        ).tolist()
                    else:
                        left_arm_joint_state.position = left_arm_target_q.tolist()
                    self._left_arm_joint_target_position_pub.publish(
                        left_arm_joint_state
                    )
                if right_arm_target_q is not None:
                    right_arm_joint_state.header.stamp = rospy.Time.now()
                    if step <= right_arm_n_steps - 1:
                        right_arm_joint_state.position = (
                            np.array(right_arm_joint_state.position)
                            + right_arm_increment
                        ).tolist()
                    else:
                        right_arm_joint_state.position = right_arm_target_q.tolist()
                    self._right_arm_joint_target_position_pub.publish(
                        right_arm_joint_state
                    )
                if torso_target_q is not None:
                    torso_joint_state.header.stamp = rospy.Time.now()
                    if step <= torso_n_steps - 1:
                        torso_joint_state.position = (
                            np.array(torso_joint_state.position) + torso_increment
                        ).tolist()
                    else:
                        torso_joint_state.position = torso_target_q.tolist()
                self._torso_joint_target_position_pub.publish(torso_joint_state)
                self._rate.sleep()

        # ensure the last position is the target position
        if left_arm_target_q is not None:
            left_arm_joint_state.header.stamp = rospy.Time.now()
            left_arm_joint_state.position = left_arm_target_q.tolist()
            self._left_arm_joint_target_position_pub.publish(left_arm_joint_state)
        if right_arm_target_q is not None:
            right_arm_joint_state.header.stamp = rospy.Time.now()
            right_arm_joint_state.position = right_arm_target_q.tolist()
            self._right_arm_joint_target_position_pub.publish(right_arm_joint_state)
        if torso_target_q is not None:
            torso_joint_state.header.stamp = rospy.Time.now()
            torso_joint_state.position = torso_target_q.tolist()
            self._torso_joint_target_position_pub.publish(torso_joint_state)
        if not (
            left_arm_target_q is None
            and right_arm_target_q is None
            and torso_target_q is None
        ):
            self._rate.sleep()

    def _gripper_control(self, gripper_action: Dict[str, float]):
        left_gripper_action = gripper_action["left"]
        right_gripper_action = gripper_action["right"]
        assert (
            1 >= left_gripper_action >= 0
        ), "Invalid left gripper action, must be between 0 and 1"
        assert (
            1 >= right_gripper_action >= 0
        ), "Invalid right gripper action, must be between 0 and 1"

        if self._left_gripper is not None:
            self._left_gripper.act(left_gripper_action)
        if self._right_gripper is not None:
            self._right_gripper.act(right_gripper_action)

    def _pcd_callback(self, pcd: PointCloud2):
        timestamp = pcd.header.stamp.secs + pcd.header.stamp.nsecs * 1e-9
        pcd = ros_numpy.point_cloud2.pointcloud2_to_array(pcd)
        pcd_xyz, pcd_xyz_mask = get_xyz_points(pcd, remove_nans=True)
        pcd = ros_numpy.point_cloud2.split_rgb_field(pcd)
        pcd_rgb = np.zeros(pcd.shape + (3,), dtype=np.uint8)
        pcd_rgb[..., 0] = pcd["r"]
        pcd_rgb[..., 1] = pcd["g"]
        pcd_rgb[..., 2] = pcd["b"]
        pcd_rgb = pcd_rgb[pcd_xyz_mask]
        self._pcd = {
            "xyz": pcd_xyz,
            "rgb": pcd_rgb,
            "stamp": timestamp,
        }
    def callback_head_image(self, msg, name):
        # global img_list
        timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
        img = np.zeros((3, 480, 640, 3), dtype=np.uint8)
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img[0] = cv2.resize(cv_image, (640, 480))
            self._rgb[name] = {
                "img": img[0],
                "stamp": timestamp,
            }
            # print(self._rgb[name]["img"].shape)
        except Exception as e:
            rospy.logerr(f"Head image callback error: {e}")
    def _rgb_callback(self, rgb_msg: Image, name: str):
        timestamp = rgb_msg.header.stamp.secs + rgb_msg.header.stamp.nsecs * 1e-9
        img = np.asarray(self._cv_bridge.imgmsg_to_cv2(rgb_msg, "rgb8"))
        self._rgb[name] = {
            "img": img,
            "stamp": timestamp,
        }

    def _depth_callback(self, depth_msg: Image, name: str):
        timestamp = depth_msg.header.stamp.secs + depth_msg.header.stamp.nsecs * 1e-9
        depth_image = self._cv_bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        self._depth[name] = {
            "depth": depth_image,
            "stamp": timestamp,
        }

    def _left_arm_state_callback(self, data: JointState):
        new_state = {
            "joint_position": np.clip(
                np.array([data.position[:7]]),
                self.left_arm_joint_low,
                self.left_arm_joint_high,
            ),
            "joint_velocity": np.array([data.velocity][:7]),
            "joint_effort": np.array([data.effort][:7]),
            "seq": np.array([data.header.seq]),
            "stamp": np.array(
                [data.header.stamp.secs + data.header.stamp.nsecs * 1e-9]
            ),
        }

        if self._left_arm_joint_state_buffer is None:
            self._left_arm_joint_state_buffer = new_state
        else:
            self._left_arm_joint_state_buffer = any_concat(
                [
                    self._left_arm_joint_state_buffer,
                    new_state,
                ],
                dim=0,
            )
            self._left_arm_joint_state_buffer = any_slice(
                self._left_arm_joint_state_buffer, np.s_[-self._state_buffer_size :]
            )
    def _mobile_state_callback(self, data: TwistStamped):
        # 完全对应话题格式：linear (x, y, z) 和 angular (x, y, z)
        new_state = {
            "linear": np.array([[
                data.twist.linear.x,
                data.twist.linear.y,
                data.twist.linear.z
            ]]),
            "angular": np.array([[
                data.twist.angular.x,
                data.twist.angular.y,
                data.twist.angular.z
            ]]),
            "stamp": np.array([data.header.stamp.secs + data.header.stamp.nsecs * 1e-9]),
        }

        if self._mobile_base_state_buffer is None:
            self._mobile_base_state_buffer = new_state
        else:
            # 拼接新的观测数据
            self._mobile_base_state_buffer = any_concat(
                [
                    self._mobile_base_state_buffer,
                    new_state,
                ],
                dim=0,
            )
            # 保持 buffer 长度
            self._mobile_base_state_buffer = any_slice(
                self._mobile_base_state_buffer, np.s_[-self._state_buffer_size :]
            )

    def _right_arm_state_callback(self, data: JointState):
        
        new_state = {
            "joint_position": np.clip(
                np.array([data.position[:7]]),
                self.right_arm_joint_low,
                self.right_arm_joint_high,
            ),
            "joint_velocity": np.array([data.velocity][:7]),
            "joint_effort": np.array([data.effort][:7]),
            "seq": np.array([data.header.seq]),
            "stamp": np.array(
                [data.header.stamp.secs + data.header.stamp.nsecs * 1e-9]
            ),
        }

        if self._right_arm_joint_state_buffer is None:
            self._right_arm_joint_state_buffer = new_state
        else:
            self._right_arm_joint_state_buffer = any_concat(
                [
                    self._right_arm_joint_state_buffer,
                    new_state,
                ],
                dim=0,
            )
            self._right_arm_joint_state_buffer = any_slice(
                self._right_arm_joint_state_buffer, np.s_[-self._state_buffer_size :]
            )

    def _torso_state_callback(self, data: JointState):
        new_state = {
            "joint_position": np.clip(
                np.array([data.position[:4]]),
                self.torso_joint_low,
                self.torso_joint_high,
            ),
            "joint_velocity": np.array([data.velocity][:4]),
            "joint_effort": np.array([data.effort][:4]),
            "seq": np.array([data.header.seq]),
            "stamp": np.array(
                [data.header.stamp.secs + data.header.stamp.nsecs * 1e-9]
            ),
        }

        if self._torso_joint_state_buffer is None:
            self._torso_joint_state_buffer = new_state
        else:
            self._torso_joint_state_buffer = any_concat(
                [
                    self._torso_joint_state_buffer,
                    new_state,
                ],
                dim=0,
            )
            self._torso_joint_state_buffer = any_slice(
                self._torso_joint_state_buffer, np.s_[-self._state_buffer_size :]
            )

    def close(self):
        # stop the base
        self.stop_mobile_base()
        if self._left_gripper is not None:
            self._left_gripper.close()
        if self._right_gripper is not None:
            self._right_gripper.close()
        rospy.signal_shutdown("Shutting down node")

    def __del__(self):
        self.close()

    @property
    def last_joint_position(self) -> Dict[str, np.ndarray]:
        return {
            "left_arm": any_slice(self._left_arm_joint_state_buffer, -1)[
                "joint_position"
            ],
            "right_arm": any_slice(self._right_arm_joint_state_buffer, -1)[
                "joint_position"
            ],
            "torso": any_slice(self._torso_joint_state_buffer, -1)["joint_position"],
        }
    @property
    def last_mobile_velocity(self) -> Dict[str, np.ndarray]:
        return {
            "mobile_base": any_slice(self._mobile_base_state_buffer, -1)[
                "linear"
            ]
        }
    @property
    def last_gripper_state(self):
        return {
            "left_gripper": (
                None
                if self._left_gripper is None
                else any_slice(self._left_gripper.state_buffer, -1)
            ),
            "right_gripper": (
                None
                if self._right_gripper is None
                else any_slice(self._right_gripper.state_buffer, -1)
            ),
        }
    @property
    def last_pointcloud(self):
        return {
            "xyz": self._pcd["xyz"],
            "rgb": self._pcd["rgb"],
            "stamp": self._pcd["stamp"],
        }

    @property
    def last_rgb(self):
        # print(self._rgb)
        # for k, v in self._rgb.items():
            # print(k, v)
        return (
            {
                k: {
                    "img": v["img"] if v is not None else None,
                    "stamp": v["stamp"] if v is not None else None,
                }
                for k, v in self._rgb.items()
            }
            if self._rgb is not None
            else None
        )

    @property
    def last_depth(self):
        return (
            {
                k: {
                    "depth": v["depth"],
                    "stamp": v["stamp"],
                }
                for k, v in self._depth.items()
            }
            if self._depth is not None
            else None
        )

    @property
    def curr_base_pose(self):
        return self._odom.curr_base_pose

    @property
    def curr_base_position(self):
        return self._odom.curr_base_position

    @property
    def curr_base_orientation(self):
        return self._odom.curr_base_orientation

    @property
    def curr_base_velocity(self):
        return self._odom.curr_base_velocity
