from typing import Any, Union, Literal

import rospy
import numpy as np
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState

from .utils import *
from .base import BaseGripper


class GalaxeaR1ProGripper(BaseGripper):
    def __init__(
        self,
        left_or_right: Literal["left", "right"],
        gripper_position_control_topic: str = "/motion_target/target_position_gripper_{left_or_right}",
        gripper_feedback_topic: str = "/hdas/feedback_gripper_{left_or_right}",
        gripper_close_stroke: float = 10.0,
        gripper_open_stroke: float = 90.0,
        publisher_queue_size: int = 1,
        state_buffer_size: int = 1000,
    ):
        assert left_or_right in ["left", "right"]
        self._gripper_position_control_topic = gripper_position_control_topic.format(
            left_or_right=left_or_right
        )
        self._gripper_feedback_topic = gripper_feedback_topic.format(
            left_or_right=left_or_right
        )
        self._gripper_close_stroke = gripper_close_stroke
        self._gripper_open_stroke = gripper_open_stroke
        self._publisher_queue_size = publisher_queue_size
        self._state_buffer_size = state_buffer_size

        self._gripper_position_control_pub = None
        self._gripper_feedback_sub = None
        self._gripper_state_buffer = None

    def init_hook(self):
        self._gripper_position_control_pub = rospy.Publisher(
            self._gripper_position_control_topic,
            JointState,
            queue_size=self._publisher_queue_size,
            latch=True,
        )
        self._gripper_feedback_sub = rospy.Subscriber(
            self._gripper_feedback_topic,
            JointState,
            self._gripper_state_callback,
        )

    def act(self, action: Union[float, np.ndarray]):
        assert isinstance(action, float) or isinstance(
            action, int
        ), "GalaxeaG1Gripper only support a single number as action"
        stroke = self._gripper_close_stroke + (1 - action) * (
            self._gripper_open_stroke - self._gripper_close_stroke
        )
        gripper_msg = JointState()
        gripper_msg.position = [max(min(stroke, 100), 0)]
        self._gripper_position_control_pub.publish(gripper_msg)

    def get_state(self, data: JointState) -> Any:
        new_state = {
            "gripper_position": np.array([data.position[0]]),
            "gripper_velocity": np.array([data.velocity[0]]),
            "gripper_effort": np.array([data.effort[0]]),
            "seq": np.array([data.header.seq]),
            "stamp": np.array(
                [data.header.stamp.secs + data.header.stamp.nsecs * 1e-9]
            ),
        }
        return new_state

    def close(self):
        pass  # ros topic publisher closed outside

    def _gripper_state_callback(self, data: JointState):
        new_state = self.get_state(data)
        if self._gripper_state_buffer is None:
            self._gripper_state_buffer = new_state
            return
        self._gripper_state_buffer = any_concat(
            [
                self._gripper_state_buffer,
                new_state,
            ],
            dim=0,
        )
        self._gripper_state_buffer = any_slice(
            self._gripper_state_buffer, np.s_[-self._state_buffer_size :]
        )

    @property
    def state_buffer(self):
        return self._gripper_state_buffer
    
    
class GalaxeaR1Gripper(BaseGripper):
    def __init__(
        self,
        left_or_right: Literal["left", "right"],
        gripper_position_control_topic: str = "/motion_control/position_control_gripper_{left_or_right}",
        gripper_feedback_topic: str = "/hdas/feedback_gripper_{left_or_right}",
        gripper_close_stroke: float = 10.0,
        gripper_open_stroke: float = 90.0,
        publisher_queue_size: int = 1,
        state_buffer_size: int = 1000,
    ):
        assert left_or_right in ["left", "right"]
        self._gripper_position_control_topic = gripper_position_control_topic.format(
            left_or_right=left_or_right
        )
        self._gripper_feedback_topic = gripper_feedback_topic.format(
            left_or_right=left_or_right
        )
        self._gripper_close_stroke = gripper_close_stroke
        self._gripper_open_stroke = gripper_open_stroke
        self._publisher_queue_size = publisher_queue_size
        self._state_buffer_size = state_buffer_size

        self._gripper_position_control_pub = None
        self._gripper_feedback_sub = None
        self._gripper_state_buffer = None

    def init_hook(self):
        self._gripper_position_control_pub = rospy.Publisher(
            self._gripper_position_control_topic,
            Float32,
            queue_size=self._publisher_queue_size,
            latch=True,
        )
        self._gripper_feedback_sub = rospy.Subscriber(
            self._gripper_feedback_topic,
            JointState,
            self._gripper_state_callback,
        )

    def act(self, action: Union[float, np.ndarray]):
        assert isinstance(action, float) or isinstance(
            action, int
        ), "GalaxeaG1Gripper only support a single number as action"
        stroke = self._gripper_close_stroke + (1 - action) * (
            self._gripper_open_stroke - self._gripper_close_stroke
        )
        gripper_msg = Float32()
        gripper_msg.data = max(min(stroke, 100), 0)
        self._gripper_position_control_pub.publish(gripper_msg)

    def get_state(self, data: JointState) -> Any:
        new_state = {
            "gripper_position": np.array([data.position[0]]),
            "gripper_velocity": np.array([data.velocity[0]]),
            "gripper_effort": np.array([data.effort[0]]),
            "seq": np.array([data.header.seq]),
            "stamp": np.array(
                [data.header.stamp.secs + data.header.stamp.nsecs * 1e-9]
            ),
        }
        return new_state

    def close(self):
        pass  # ros topic publisher closed outside

    def _gripper_state_callback(self, data: JointState):
        new_state = self.get_state(data)
        if self._gripper_state_buffer is None:
            self._gripper_state_buffer = new_state
            return
        self._gripper_state_buffer = any_concat(
            [
                self._gripper_state_buffer,
                new_state,
            ],
            dim=0,
        )
        self._gripper_state_buffer = any_slice(
            self._gripper_state_buffer, np.s_[-self._state_buffer_size :]
        )

    @property
    def state_buffer(self):
        return self._gripper_state_buffer
