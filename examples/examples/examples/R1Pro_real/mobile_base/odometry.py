import numpy as np
import quaternion
import rospy
from nav_msgs.msg import Odometry


class Odom:
    def __init__(
        self,
        *,
        odom_topic: str = "/camera/odom/sample",
        T_odom2base: np.ndarray,
        wait_for_first_msg: bool = True,
    ):
        self._sub = rospy.Subscriber(
            odom_topic,
            Odometry,
            self._callback,
        )
        self._T_odom2base = T_odom2base
        self._curr_base_pose = None
        self._curr_base_position = None
        self._curr_base_orientation = None
        self._curr_base_velocity = None

        if wait_for_first_msg:
            print("Waiting for odometry topic...")
            rospy.wait_for_message(odom_topic, Odometry)

    def _callback(self, data: Odometry):
        x, y, z = (
            data.pose.pose.position.x,
            data.pose.pose.position.y,
            data.pose.pose.position.z,
        )
        quat = np.quaternion(
            data.pose.pose.orientation.w,
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
        )
        rotation_matrix = quaternion.as_rotation_matrix(quat)
        curr_odom_pose = np.eye(4)
        curr_odom_pose[:3, :3] = rotation_matrix
        curr_odom_pose[:3, 3] = np.array([x, y, z])
        self._curr_base_pose = self._T_odom2base @ curr_odom_pose
        self._curr_base_position = self._curr_base_pose[:3, 3]
        self._curr_base_orientation = self._curr_base_pose[:3, :3]

        vx, vy, vz = (
            data.twist.twist.linear.x,
            data.twist.twist.linear.y,
            data.twist.twist.linear.z,
        )
        vx = 0 if abs(vx) <= 1e-2 else vx
        vy = 0 if abs(vy) <= 1e-2 else vy
        base_xyz_vel = self._T_odom2base[:3, :3] @ np.array([vx, vy, vz])
        v_yaw = data.twist.twist.angular.z
        v_yaw = 0 if abs(v_yaw) <= 5e-3 else v_yaw
        self._curr_base_velocity = np.array([base_xyz_vel[0], base_xyz_vel[1], v_yaw])

    @property
    def curr_base_pose(self):
        return self._curr_base_pose

    @property
    def curr_base_position(self):
        return self._curr_base_position

    @property
    def curr_base_orientation(self):
        return self._curr_base_orientation

    @property
    def curr_base_velocity(self):
        return self._curr_base_velocity
