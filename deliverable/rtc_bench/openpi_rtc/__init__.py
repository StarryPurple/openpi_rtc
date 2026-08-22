"""JAX RTC (Real-Time Chunking) module for the openpi pi0/pi0.5 stack.

Implements lerobot-style RTC guidance (arXiv:2506.07339) for the *JAX*
``openpi.models.pi0.Pi0`` policy: guidance inpainting inside the denoising
loop, a numpy action queue, offline HDF5 evaluation, checkpoint probing, a
real-robot adapter, and the train-RTC / πR² training entries. See README.md.
"""

import pathlib
import sys

# This package is also the repository root: the vendored `openpi` /
# `openpi_client` packages live next to it. Make both importable regardless
# of whether they were installed by `uv sync` (the workspace installs
# openpi-client from packages/openpi-client/src) or just on PYTHONPATH.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent
_OPENPI_CLIENT_SRC = _REPO_ROOT / "packages" / "openpi-client" / "src"
for _p in (str(_REPO_ROOT), str(_OPENPI_CLIENT_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# When this package is extracted as <xtrainer>/rtc_control, the platform's
# dobot_control lives in the parent (<xtrainer> root). Append the parent
# *after* the repo root so `import dobot_control` resolves without a
# PYTHONPATH tweak, while this repo's own modules (e.g. scripts/) keep
# priority.
_PARENT_ROOT = _REPO_ROOT.parent
if str(_PARENT_ROOT) not in sys.path:
    sys.path.append(str(_PARENT_ROOT))

from .action_queue import ActionQueue
from .integrate_openpi import (
    RtcPolicy,
    enable_rtc_on_model,
    load_norm_stats,
    rtc_sample_actions,
    wrap_policy_for_rtc,
)
from .rtc_train import (
    patch_pi0_for_train_rtc,
    rtc_compute_loss,
    train_rtc_sample_actions,
    wrap_policy_for_train_rtc,
)
from .pir2_train import (
    Pir2Config,
    patch_pi0_for_pir2,
    pir2_compute_loss,
    pir2_sample_actions,
    wrap_policy_for_pir2,
)
from .rtc_config import RTCAttentionSchedule, RTCConfig
from .rtc_processor import RTCProcessor

__all__ = [
    "ActionQueue",
    "Pir2Config",
    "RTCConfig",
    "RTCProcessor",
    "RTCAttentionSchedule",
    "RtcPolicy",
    "enable_rtc_on_model",
    "load_norm_stats",
    "patch_pi0_for_pir2",
    "patch_pi0_for_train_rtc",
    "pir2_compute_loss",
    "pir2_sample_actions",
    "rtc_compute_loss",
    "rtc_sample_actions",
    "train_rtc_sample_actions",
    "wrap_policy_for_pir2",
    "wrap_policy_for_train_rtc",
    "wrap_policy_for_rtc",
]
