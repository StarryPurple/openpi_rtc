"""RTC (Real-Time Chunking) configuration for the JAX openpi stack.

Reference: "Real-Time Execution of Action Chunking Flow Policies"
(arXiv:2506.07339); the attention schedule follows huggingface/lerobot's RTC
module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RTCAttentionSchedule(str, Enum):
    LINEAR = "linear"
    EXP = "exp"
    ONES = "ones"
    ZEROS = "zeros"


@dataclass
class RTCConfig:
    """Real-Time Chunking (RTC) inference configuration.

    ``anchor_correction`` accounts for the fact that openpi XTrainer policies
    are trained on *delta* actions (whole-chunk shift by the observation state):
    a raw chunk generated from state ``s_prev`` must be re-anchored to the new
    observation state before it is used as guidance target. Keep it enabled for
    the pi05 XTrainer checkpoints.
    """

    enabled: bool = False
    # How many timesteps of the new chunk are blended with the executed tail
    # of the previous chunk (soft prefix attention).
    execution_horizon: int = 10
    # Clip for the guidance weight (paper: kappa = 5 for 5 denoising steps;
    # lerobot uses 10.0 for pi0/pi0.5 with 10 steps).
    max_guidance_weight: float = 10.0
    prefix_attention_schedule: RTCAttentionSchedule | str = RTCAttentionSchedule.EXP
    # Re-anchor the previous raw chunk to the current observation state
    # (delta-action policies only; see class docstring).
    anchor_correction: bool = True
    debug: bool = False

    def __post_init__(self) -> None:
        if self.max_guidance_weight <= 0:
            raise ValueError("max_guidance_weight must be positive")
        if isinstance(self.prefix_attention_schedule, str):
            self.prefix_attention_schedule = RTCAttentionSchedule(self.prefix_attention_schedule)
