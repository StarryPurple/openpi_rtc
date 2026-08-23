import numpy as np
from typing import Any, Union
from abc import ABC, abstractmethod

from sensor_msgs.msg import JointState


class BaseGripper(ABC):
    """
    The base class for all different grippers.
    This class is agnostic to the low-level control of a specific gripper.
    But subclasses should implement the following hook methods.
    """

    @abstractmethod
    def init_hook(self):
        """
        Initialize gripper ROS topic publishers and subscribers.
        """
        pass

    @abstractmethod
    def act(self, action: Union[float, np.ndarray]):
        """
        Control the gripper.
        The input action can be:
            - float within the range [0, 1]. 0 means fully open, and 1 means fully closed.
            - np.ndarray within [0, 1]. Normalized action for each DoF.
        """
        pass

    @abstractmethod
    def get_state(self, data: JointState) -> Any:
        """
        Get the current state of the gripper. Need to return gripper state with a leading time dimension (T, ...).

        data: sensor_msgs.msg.JointState: object passed to _state_callback in interfaces.py
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close the gripper instance.
        """
        pass

    @property
    def state_buffer(self):
        """
        Return the state buffer.
        """
        raise NotImplementedError
