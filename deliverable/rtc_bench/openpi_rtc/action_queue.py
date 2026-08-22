"""Thread-safe action queue for the async RTC robot loop (numpy-based).

Two streams are tracked:

- ``original_queue``: the raw model-space chunks (normalized delta actions),
  used as ``prev_chunk_left_over`` for the next inference call's RTC guidance;
- ``queue``: the processed actions (robot units) actually sent to the robot.

In RTC mode ``merge`` *replaces* the in-flight queues with the new chunk
skipping the ``real_delay`` already-executed steps; in baseline mode it
appends.
"""

from __future__ import annotations

import logging
from threading import Lock

import numpy as np

from .rtc_config import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    def __init__(self, cfg: RTCConfig):
        self.queue: np.ndarray | None = None
        self.original_queue: np.ndarray | None = None
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def get(self) -> np.ndarray | None:
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index]
            self.last_index += 1
            return action.copy()

    def clear(self) -> None:
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        if self.queue is None:
            return 0
        return len(self.queue) - self.last_index

    def empty(self) -> bool:
        if self.queue is None:
            return True
        return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        return self.last_index

    def get_left_over(self) -> np.ndarray | None:
        """Unconsumed raw model-space actions of the current chunk."""
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].copy()

    def merge(
        self,
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        real_delay: int,
    ) -> None:
        with self.lock:
            delay = max(0, int(real_delay))
            if self.cfg.enabled:
                clamped = max(0, min(delay, len(original_actions), len(processed_actions)))
                self.original_queue = np.asarray(original_actions[clamped:], dtype=np.float32)
                self.queue = np.asarray(processed_actions[clamped:], dtype=np.float32)
                self.last_index = 0
                logger.debug(
                    "RTC merge: delay=%d clamped=%d remaining=%d",
                    delay,
                    clamped,
                    len(self.queue),
                )
                return
            if self.queue is None:
                self.original_queue = np.asarray(original_actions, dtype=np.float32)
                self.queue = np.asarray(processed_actions, dtype=np.float32)
                return
            self.original_queue = np.concatenate(
                [self.original_queue, np.asarray(original_actions, dtype=np.float32)]
            )
            self.original_queue = self.original_queue[self.last_index :]
            self.queue = np.concatenate(
                [self.queue, np.asarray(processed_actions, dtype=np.float32)]
            )
            self.queue = self.queue[self.last_index :]
            self.last_index = 0
