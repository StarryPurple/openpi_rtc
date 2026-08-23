import re
import os
import numpy as np
import pybullet as pb

from brs_ctrl.asset_root import ASSET_ROOT


class R1ProKinematics:
    torso_joint_high = np.array([1.8326, 2.5307, 1.8326, 3.0543])
    torso_joint_low = np.array([-1.1345, -2.7925, -2.0944, -3.0543])
    left_arm_joint_high = np.array([2.8798, 3.2289, 1,  0, 2.8798, 1.6581, 2.8798])
    left_arm_joint_low = np.array([-2.8798, -3, -1,  -3.3161, -2.8798, -1.6581, -2.8798])
    right_arm_joint_high = np.array([2.8798, 3.2289, 1, 0, 2.8798, 1.6581, 2.8798])
    right_arm_joint_low = np.array([-2.8798, -3, -1, -3.3161, -2.8798, -1.6581, -2.8798])

    def __init__(
        self,
    ):
        urdf_path = os.path.join(ASSET_ROOT, "robot", "r1_pro", "r1_pro.urdf")

        self._pb_client_id = pb.connect(pb.DIRECT)
        self._pb_robot_id = pb.loadURDF(
            str(urdf_path),
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._pb_client_id,
        )
        pb.resetBasePositionAndOrientation(
            self._pb_robot_id,
            [0, 0, 0],
            [0, 0, 0, 1],
            physicsClientId=self._pb_client_id,
        )

        self._pb_num_joints = pb.getNumJoints(
            self._pb_robot_id, physicsClientId=self._pb_client_id
        )
        for i in range(self._pb_num_joints):
            pb.resetJointState(
                self._pb_robot_id, i, 0, physicsClientId=self._pb_client_id
            )

        self._left_arm_joint_idxs = [[] for _ in range(6)]
        self._right_arm_joint_idxs = [[] for _ in range(6)]
        self._torso_joint_idxs = [[] for _ in range(4)]

        self._link_name_to_index = {
            pb.getBodyInfo(self._pb_robot_id, physicsClientId=self._pb_client_id)[
                0
            ].decode("UTF-8"): -1,
        }

        left_joint_pattern = re.compile(r"left_arm_joint[1-6]")
        right_joint_pattern = re.compile(r"right_arm_joint[1-6]")
        torso_joint_pattern = re.compile(r"torso_joint[1-4]")

        for _id in range(
            pb.getNumJoints(self._pb_robot_id, physicsClientId=self._pb_client_id)
        ):
            joint_info = pb.getJointInfo(
                self._pb_robot_id, _id, physicsClientId=self._pb_client_id
            )
            joint_name = joint_info[1].decode("UTF-8")
            if left_joint_pattern.match(joint_name):
                idx = int(joint_name[-1]) - 1
                self._left_arm_joint_idxs[idx].append(_id)
            elif right_joint_pattern.match(joint_name):
                idx = int(joint_name[-1]) - 1
                self._right_arm_joint_idxs[idx].append(_id)
            elif torso_joint_pattern.match(joint_name):
                idx = int(joint_name[-1]) - 1
                self._torso_joint_idxs[idx].append(_id)
            link_name = joint_info[12].decode("UTF-8")
            self._link_name_to_index[link_name] = _id
        self._left_arm_joint_idxs = [idx[0] for idx in self._left_arm_joint_idxs]
        self._right_arm_joint_idxs = [idx[0] for idx in self._right_arm_joint_idxs]
        self._torso_joint_idxs = [idx[0] for idx in self._torso_joint_idxs]

        self._odom2base_link = None

    def get_link_poses_in_base_link(
        self,
        *,
        left_wrist_camera_link_name: str = "left_zedm_left_camera_frame",
        right_wrist_camera_link_name: str = "right_zedm_left_camera_frame",
        head_camera_link_name: str = "zed2_left_camera_frame",
        left_eef_link_name: str = "left_gripping_point",
        right_eef_link_name: str = "right_gripping_point",
        head_link_name: str = "head_point",
        curr_left_arm_joint: np.ndarray,
        curr_right_arm_joint: np.ndarray,
        curr_torso_joint: np.ndarray,
        return_matrix: bool = True,
    ):
        """
        Get the pose of the head camera in the base link frame.
        """
        assert (
            curr_left_arm_joint.ndim == 1 and len(curr_left_arm_joint) == 7
        ), "Must provide 6 left arm joint angles."
        assert np.all(self.left_arm_joint_low <= curr_left_arm_joint) and np.all(
            curr_left_arm_joint <= self.left_arm_joint_high
        ), "Left arm joint angles out of range."

        assert (
            curr_right_arm_joint.ndim == 1 and len(curr_right_arm_joint) == 7
        ), "Must provide 6 right arm joint angles."
        assert np.all(self.right_arm_joint_low <= curr_right_arm_joint) and np.all(
            curr_right_arm_joint <= self.right_arm_joint_high
        ), "Right arm joint angles out of range."

        assert (
            curr_torso_joint.ndim == 1 and len(curr_torso_joint) == 4
        ), "Must provide 4 torso joint angles."
        assert np.all(self.torso_joint_low <= curr_torso_joint) and np.all(
            curr_torso_joint <= self.torso_joint_high
        ), "Torso joint angles out of range."

        for idx, q in zip(self._left_arm_joint_idxs, curr_left_arm_joint):
            pb.resetJointState(
                self._pb_robot_id, idx, q, physicsClientId=self._pb_client_id
            )

        for idx, q in zip(self._right_arm_joint_idxs, curr_right_arm_joint):
            pb.resetJointState(
                self._pb_robot_id, idx, q, physicsClientId=self._pb_client_id
            )

        for idx, q in zip(self._torso_joint_idxs, curr_torso_joint):
            pb.resetJointState(
                self._pb_robot_id, idx, q, physicsClientId=self._pb_client_id
            )

        # get link poses
        left_wrist_camera_link_idx = self._link_name_to_index[
            left_wrist_camera_link_name
        ]
        right_wrist_camera_link_idx = self._link_name_to_index[
            right_wrist_camera_link_name
        ]
        head_camera_link_idx = self._link_name_to_index[head_camera_link_name]
        left_eef_link_idx = self._link_name_to_index[left_eef_link_name]
        right_eef_link_idx = self._link_name_to_index[right_eef_link_name]
        head_link_idx = self._link_name_to_index[head_link_name]

        (
            left_wrist_camera_ls,
            right_wrist_camera_ls,
            head_camera_ls,
            left_eef_ls,
            right_eef_ls,
            head_eef_ls,
        ) = pb.getLinkStates(
            self._pb_robot_id,
            [
                left_wrist_camera_link_idx,
                right_wrist_camera_link_idx,
                head_camera_link_idx,
                left_eef_link_idx,
                right_eef_link_idx,
                head_link_idx,
            ],
            physicsClientId=self._pb_client_id,
        )
        left_wrist_camera_position, left_wrist_camera_quaternion = (
            np.array(left_wrist_camera_ls[0]),
            left_wrist_camera_ls[1],
        )
        right_wrist_camera_position, right_wrist_camera_quaternion = (
            np.array(right_wrist_camera_ls[0]),
            right_wrist_camera_ls[1],
        )
        head_camera_position, head_camera_quaternion = (
            np.array(head_camera_ls[0]),
            head_camera_ls[1],
        )
        left_eef_position, left_eef_quaternion = (
            np.array(left_eef_ls[0]),
            left_eef_ls[1],
        )
        right_eef_position, right_eef_quaternion = (
            np.array(right_eef_ls[0]),
            right_eef_ls[1],
        )
        head_eef_position, head_eef_quaternion = (
            np.array(head_eef_ls[0]),
            head_eef_ls[1],
        )
        if not return_matrix:
            return {
                "left_wrist_camera": (
                    left_wrist_camera_position,
                    left_wrist_camera_quaternion,
                ),
                "right_wrist_camera": (
                    right_wrist_camera_position,
                    right_wrist_camera_quaternion,
                ),
                "head_camera": (head_camera_position, head_camera_quaternion),
                "left_eef": (left_eef_position, left_eef_quaternion),
                "right_eef": (right_eef_position, right_eef_quaternion),
                "head_eef": (head_eef_position, head_eef_quaternion),
            }
        else:
            left_wrist_rotaion_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    left_wrist_camera_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            T_left_wrist = np.eye(4)
            T_left_wrist[:3, :3] = left_wrist_rotaion_matrix
            T_left_wrist[:3, 3] = left_wrist_camera_position

            right_wrist_rotaion_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    right_wrist_camera_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            T_right_wrist = np.eye(4)
            T_right_wrist[:3, :3] = right_wrist_rotaion_matrix
            T_right_wrist[:3, 3] = right_wrist_camera_position

            head_rotation_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    head_camera_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            T_head = np.eye(4)
            T_head[:3, :3] = head_rotation_matrix
            T_head[:3, 3] = head_camera_position

            left_eef_rotation_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    left_eef_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            T_left_eef = np.eye(4)
            T_left_eef[:3, :3] = left_eef_rotation_matrix
            T_left_eef[:3, 3] = left_eef_position

            right_eef_rotation_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    right_eef_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            T_right_eef = np.eye(4)
            T_right_eef[:3, :3] = right_eef_rotation_matrix
            T_right_eef[:3, 3] = right_eef_position

            head_eef_rotation_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    head_eef_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            T_head_eef = np.eye(4)
            T_head_eef[:3, :3] = head_eef_rotation_matrix
            T_head_eef[:3, 3] = head_eef_position

            return {
                "left_wrist_camera": T_left_wrist,
                "right_wrist_camera": T_right_wrist,
                "head_camera": T_head,
                "left_eef": T_left_eef,
                "right_eef": T_right_eef,
                "head": T_head_eef,
            }

    @property
    def T_odom2base(self):
        if self._odom2base_link is None:
            odom_frame_idx = self._link_name_to_index["t265_pose_tracking_frame"]
            ls = pb.getLinkState(
                self._pb_robot_id, odom_frame_idx, physicsClientId=self._pb_client_id
            )
            odom_position, odom_quaternion = np.array(ls[0]), ls[1]
            odom_rotation_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    odom_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            self._odom2base_link = np.eye(4)
            self._odom2base_link[:3, :3] = odom_rotation_matrix
            self._odom2base_link[:3, 3] = odom_position
        return self._odom2base_link


class R1Kinematics:
    torso_joint_high = np.array([1.8326, 2.5307, 1.8326, 3.0543])
    torso_joint_low = np.array([-1.1345, -2.7925, -2.0944, -3.0543])
    left_arm_joint_high = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
    left_arm_joint_low = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])
    right_arm_joint_high = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
    right_arm_joint_low = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])

    def __init__(
        self,
    ):
        urdf_path = os.path.join(ASSET_ROOT, "robot", "r1_pro", "r1_pro.urdf")

        self._pb_client_id = pb.connect(pb.DIRECT)
        self._pb_robot_id = pb.loadURDF(
            str(urdf_path),
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._pb_client_id,
        )
        pb.resetBasePositionAndOrientation(
            self._pb_robot_id,
            [0, 0, 0],
            [0, 0, 0, 1],
            physicsClientId=self._pb_client_id,
        )

        self._pb_num_joints = pb.getNumJoints(
            self._pb_robot_id, physicsClientId=self._pb_client_id
        )
        for i in range(self._pb_num_joints):
            pb.resetJointState(
                self._pb_robot_id, i, 0, physicsClientId=self._pb_client_id
            )

        self._left_arm_joint_idxs = [[] for _ in range(6)]
        self._right_arm_joint_idxs = [[] for _ in range(6)]
        self._torso_joint_idxs = [[] for _ in range(4)]

        self._link_name_to_index = {
            pb.getBodyInfo(self._pb_robot_id, physicsClientId=self._pb_client_id)[
                0
            ].decode("UTF-8"): -1,
        }

        left_joint_pattern = re.compile(r"left_arm_joint[1-6]")
        right_joint_pattern = re.compile(r"right_arm_joint[1-6]")
        torso_joint_pattern = re.compile(r"torso_joint[1-4]")

        for _id in range(
            pb.getNumJoints(self._pb_robot_id, physicsClientId=self._pb_client_id)
        ):
            joint_info = pb.getJointInfo(
                self._pb_robot_id, _id, physicsClientId=self._pb_client_id
            )
            joint_name = joint_info[1].decode("UTF-8")
            if left_joint_pattern.match(joint_name):
                idx = int(joint_name[-1]) - 1
                self._left_arm_joint_idxs[idx].append(_id)
            elif right_joint_pattern.match(joint_name):
                idx = int(joint_name[-1]) - 1
                self._right_arm_joint_idxs[idx].append(_id)
            elif torso_joint_pattern.match(joint_name):
                idx = int(joint_name[-1]) - 1
                self._torso_joint_idxs[idx].append(_id)
            link_name = joint_info[12].decode("UTF-8")
            self._link_name_to_index[link_name] = _id
        self._left_arm_joint_idxs = [idx[0] for idx in self._left_arm_joint_idxs]
        self._right_arm_joint_idxs = [idx[0] for idx in self._right_arm_joint_idxs]
        self._torso_joint_idxs = [idx[0] for idx in self._torso_joint_idxs]

        self._odom2base_link = None

    def get_link_poses_in_base_link(
        self,
        *,
        left_wrist_camera_link_name: str = "left_zedm_left_camera_frame",
        right_wrist_camera_link_name: str = "right_zedm_left_camera_frame",
        head_camera_link_name: str = "zed2_left_camera_frame",
        left_eef_link_name: str = "left_gripping_point",
        right_eef_link_name: str = "right_gripping_point",
        head_link_name: str = "head_point",
        curr_left_arm_joint: np.ndarray,
        curr_right_arm_joint: np.ndarray,
        curr_torso_joint: np.ndarray,
        return_matrix: bool = True,
    ):
        """
        Get the pose of the head camera in the base link frame.
        """
        assert (
            curr_left_arm_joint.ndim == 1 and len(curr_left_arm_joint) == 6
        ), "Must provide 6 left arm joint angles."
        assert np.all(self.left_arm_joint_low <= curr_left_arm_joint) and np.all(
            curr_left_arm_joint <= self.left_arm_joint_high
        ), "Left arm joint angles out of range."

        assert (
            curr_right_arm_joint.ndim == 1 and len(curr_right_arm_joint) == 6
        ), "Must provide 6 right arm joint angles."
        assert np.all(self.right_arm_joint_low <= curr_right_arm_joint) and np.all(
            curr_right_arm_joint <= self.right_arm_joint_high
        ), "Right arm joint angles out of range."

        assert (
            curr_torso_joint.ndim == 1 and len(curr_torso_joint) == 4
        ), "Must provide 4 torso joint angles."
        assert np.all(self.torso_joint_low <= curr_torso_joint) and np.all(
            curr_torso_joint <= self.torso_joint_high
        ), "Torso joint angles out of range."

        for idx, q in zip(self._left_arm_joint_idxs, curr_left_arm_joint):
            pb.resetJointState(
                self._pb_robot_id, idx, q, physicsClientId=self._pb_client_id
            )

        for idx, q in zip(self._right_arm_joint_idxs, curr_right_arm_joint):
            pb.resetJointState(
                self._pb_robot_id, idx, q, physicsClientId=self._pb_client_id
            )

        for idx, q in zip(self._torso_joint_idxs, curr_torso_joint):
            pb.resetJointState(
                self._pb_robot_id, idx, q, physicsClientId=self._pb_client_id
            )

        # get link poses
        left_wrist_camera_link_idx = self._link_name_to_index[
            left_wrist_camera_link_name
        ]
        right_wrist_camera_link_idx = self._link_name_to_index[
            right_wrist_camera_link_name
        ]
        head_camera_link_idx = self._link_name_to_index[head_camera_link_name]
        left_eef_link_idx = self._link_name_to_index[left_eef_link_name]
        right_eef_link_idx = self._link_name_to_index[right_eef_link_name]
        head_link_idx = self._link_name_to_index[head_link_name]

        (
            left_wrist_camera_ls,
            right_wrist_camera_ls,
            head_camera_ls,
            left_eef_ls,
            right_eef_ls,
            head_eef_ls,
        ) = pb.getLinkStates(
            self._pb_robot_id,
            [
                left_wrist_camera_link_idx,
                right_wrist_camera_link_idx,
                head_camera_link_idx,
                left_eef_link_idx,
                right_eef_link_idx,
                head_link_idx,
            ],
            physicsClientId=self._pb_client_id,
        )
        left_wrist_camera_position, left_wrist_camera_quaternion = (
            np.array(left_wrist_camera_ls[0]),
            left_wrist_camera_ls[1],
        )
        right_wrist_camera_position, right_wrist_camera_quaternion = (
            np.array(right_wrist_camera_ls[0]),
            right_wrist_camera_ls[1],
        )
        head_camera_position, head_camera_quaternion = (
            np.array(head_camera_ls[0]),
            head_camera_ls[1],
        )
        left_eef_position, left_eef_quaternion = (
            np.array(left_eef_ls[0]),
            left_eef_ls[1],
        )
        right_eef_position, right_eef_quaternion = (
            np.array(right_eef_ls[0]),
            right_eef_ls[1],
        )
        head_eef_position, head_eef_quaternion = (
            np.array(head_eef_ls[0]),
            head_eef_ls[1],
        )
        if not return_matrix:
            return {
                "left_wrist_camera": (
                    left_wrist_camera_position,
                    left_wrist_camera_quaternion,
                ),
                "right_wrist_camera": (
                    right_wrist_camera_position,
                    right_wrist_camera_quaternion,
                ),
                "head_camera": (head_camera_position, head_camera_quaternion),
                "left_eef": (left_eef_position, left_eef_quaternion),
                "right_eef": (right_eef_position, right_eef_quaternion),
                "head_eef": (head_eef_position, head_eef_quaternion),
            }
        else:
            left_wrist_rotaion_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    left_wrist_camera_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            T_left_wrist = np.eye(4)
            T_left_wrist[:3, :3] = left_wrist_rotaion_matrix
            T_left_wrist[:3, 3] = left_wrist_camera_position

            right_wrist_rotaion_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    right_wrist_camera_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            T_right_wrist = np.eye(4)
            T_right_wrist[:3, :3] = right_wrist_rotaion_matrix
            T_right_wrist[:3, 3] = right_wrist_camera_position

            head_rotation_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    head_camera_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            T_head = np.eye(4)
            T_head[:3, :3] = head_rotation_matrix
            T_head[:3, 3] = head_camera_position

            left_eef_rotation_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    left_eef_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            T_left_eef = np.eye(4)
            T_left_eef[:3, :3] = left_eef_rotation_matrix
            T_left_eef[:3, 3] = left_eef_position

            right_eef_rotation_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    right_eef_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            T_right_eef = np.eye(4)
            T_right_eef[:3, :3] = right_eef_rotation_matrix
            T_right_eef[:3, 3] = right_eef_position

            head_eef_rotation_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    head_eef_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            T_head_eef = np.eye(4)
            T_head_eef[:3, :3] = head_eef_rotation_matrix
            T_head_eef[:3, 3] = head_eef_position

            return {
                "left_wrist_camera": T_left_wrist,
                "right_wrist_camera": T_right_wrist,
                "head_camera": T_head,
                "left_eef": T_left_eef,
                "right_eef": T_right_eef,
                "head": T_head_eef,
            }

    @property
    def T_odom2base(self):
        if self._odom2base_link is None:
            odom_frame_idx = self._link_name_to_index["t265_pose_tracking_frame"]
            ls = pb.getLinkState(
                self._pb_robot_id, odom_frame_idx, physicsClientId=self._pb_client_id
            )
            odom_position, odom_quaternion = np.array(ls[0]), ls[1]
            odom_rotation_matrix = np.array(
                pb.getMatrixFromQuaternion(
                    odom_quaternion, physicsClientId=self._pb_client_id
                )
            ).reshape(3, 3)
            self._odom2base_link = np.eye(4)
            self._odom2base_link[:3, :3] = odom_rotation_matrix
            self._odom2base_link[:3, 3] = odom_position
        return self._odom2base_link
