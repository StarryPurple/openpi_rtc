"""Drop-in RTC integration for the openpi JAX pi0/pi0.5 policy.

How it works
------------
openpi's JAX policy flow is ``Policy.infer(obs)`` ->
``model.sample_actions(rng, observation)``, jitted via
``openpi.shared.nnx_utils.module_jit``. ``Pi0.sample_actions`` runs a
``jax.lax.while_loop`` over flow-matching denoising steps. RTC needs to inject
guidance inside that loop, so we:

1. monkey-patch ``Pi0.sample_actions`` with an RTC-aware copy
   (``rtc_sample_actions``) that calls ``RTCProcessor.denoise_step`` when a
   previous chunk tail is available;
2. re-jit the patched method with ``module_jit``;
3. wrap the policy (``RtcPolicy``) so ``infer`` forwards
   ``prev_chunk_left_over / inference_delay / execution_horizon`` to the model
   and captures the raw normalized chunk (``last_raw_chunk``) for the next
   call.

Normalization
-------------
The guidance target must live in the model's action space: quantile-normalized
*delta* actions (12 joints relative to the observation state, 2 gripper dims
absolute). ``Policy.infer`` returns robot-unit absolute actions, so
``last_raw_chunk`` (captured before the output transforms) is what gets fed
back. Because deltas are anchored to the observation state, ``RtcPolicy``
re-anchors a previous chunk to the *current* observation state
(``anchor_correction``) before it is used as guidance target.
"""

from __future__ import annotations

import pathlib
from types import MethodType
from typing import Any

import einops
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy as _policy
from openpi.shared import nnx_utils
from openpi import transforms as _transforms

from .rtc_config import RTCConfig
from .rtc_processor import RTCProcessor


# Default XTrainer delta mask: 12 joints (delta) + 2 gripper dims (absolute).
DELTA_ACTION_MASK = tuple(_transforms.make_bool_mask(6, -1, 6, -1))


def rtc_sample_actions(
    self,
    rng,
    observation,
    *,
    num_steps=10,
    noise=None,
    prev_chunk_left_over=None,
    inference_delay: int | None = None,
    execution_horizon: int | None = None,
):
    """RTC-aware copy of ``openpi.models.pi0.Pi0.sample_actions``.

    The denoising loop is identical to the original; the only difference is
    that each step's velocity is computed through
    ``self.rtc_processor.denoise_step`` when RTC is enabled and a previous
    chunk tail is available.
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    # Same convention as the original: t=1 is noise, t=0 is the target.
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

    # First fill the KV cache with a forward pass of the prefix.
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )

    rtc = getattr(self, "rtc_processor", None)
    enabled = rtc is not None and rtc.rtc_config.enabled and prev_chunk_left_over is not None

    if enabled:
        prev = jnp.asarray(prev_chunk_left_over)
        if prev.ndim == 2:
            prev = prev[None, ...]
        horizon = (
            rtc.rtc_config.execution_horizon
            if execution_horizon is None
            else execution_horizon
        )
        horizon = min(int(horizon), prev.shape[1])
        delay = min(int(inference_delay) if inference_delay is not None else 0, horizon)
        if prev.shape[1] < self.action_horizon or prev.shape[2] < self.action_dim:
            padded = jnp.zeros(
                (batch_size, self.action_horizon, self.action_dim), dtype=prev.dtype
            )
            padded = padded.at[:, : prev.shape[1], : prev.shape[2]].set(prev)
            prev = padded

    def suffix_forward(x_t, time):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_2 = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate([prefix_attn_mask_2, suffix_attn_mask], axis=-1)
        suffix_positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1)
            - 1
        )
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    def step(carry):
        x_t, time = carry
        if enabled:
            v_t = rtc.denoise_step(
                x_t,
                prev,
                delay,
                time,
                lambda x: suffix_forward(x, time),
                execution_horizon=horizon,
            )
        else:
            v_t = suffix_forward(x_t, time)
        return x_t + dt * v_t, time + dt

    def cond(carry):
        _, time = carry
        # Robust to floating-point error.
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0


def enable_rtc_on_model(model, rtc_config: RTCConfig) -> None:
    """Attach an RTCProcessor and swap the model's ``sample_actions``."""
    model.rtc_processor = RTCProcessor(rtc_config)
    model.sample_actions = MethodType(rtc_sample_actions, model)


def load_norm_stats(checkpoint_dir, train_config) -> dict[str, Any] | None:
    """Load the checkpoint's normalization stats (same source as the policy)."""
    from openpi.training import checkpoints as _checkpoints

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        return None
    return _checkpoints.load_norm_stats(
        pathlib.Path(checkpoint_dir) / "assets", data_config.asset_id
    )


class RtcPolicy:
    """JAX ``Policy`` wrapper that forwards RTC kwargs and captures raw chunks.

    The wrapper shares the underlying policy's attributes (duck-typed as a
    ``Policy``); ``infer`` is re-implemented on top of
    ``openpi.policies.policy.Policy.infer`` so that the patched
    ``_sample_actions`` (which lives on the wrapper) is actually invoked.
    """

    def __init__(self, policy, rtc_config: RTCConfig, norm_stats=None):
        self._orig_policy = policy
        self.__dict__.update(policy.__dict__)
        enable_rtc_on_model(self._model, rtc_config)

        # `num_steps` / `inference_delay` / `execution_horizon` must stay
        # Python ints inside the jitted body (they are used to build static
        # prefix weights and Python-level masks).
        jitted = nnx_utils.module_jit(
            self._model.sample_actions,
            static_argnames=("num_steps", "inference_delay", "execution_horizon"),
        )

        def capture(rng_or_device, observation, **kwargs):
            out = jitted(rng_or_device, observation, **kwargs)
            self._last_raw_chunk = np.asarray(out[0])
            return out

        self._sample_actions = capture
        self._rtc_config = rtc_config
        self._norm_stats = norm_stats
        self._delta_mask = np.asarray(DELTA_ACTION_MASK, dtype=bool)
        self._last_raw_chunk = None
        self._warned_no_stats = False

    def infer(
        self,
        obs: dict,
        *,
        noise=None,
        prev_chunk_left_over=None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
    ) -> dict:
        extra = {}
        if prev_chunk_left_over is not None:
            extra["prev_chunk_left_over"] = jnp.asarray(
                prev_chunk_left_over, dtype=jnp.float32
            )
        if inference_delay is not None:
            extra["inference_delay"] = int(inference_delay)
        if execution_horizon is not None:
            extra["execution_horizon"] = int(execution_horizon)
        prev_kwargs = dict(self._sample_kwargs)
        try:
            self._sample_kwargs.update(extra)
            return _policy.Policy.infer(self, obs, noise=noise)
        finally:
            self._sample_kwargs.clear()
            self._sample_kwargs.update(prev_kwargs)

    def prepare_prev_chunk(self, prev_raw, prev_state, cur_state) -> np.ndarray:
        """Re-anchor a previous raw chunk to the current observation state.

        ``prev_raw`` is the model-space chunk (H, A) generated from
        ``prev_state``. XTrainer deltas are ``abs - state``, so a chunk must be
        shifted by ``prev_state - cur_state`` (joint dims only) before it is
        compared against a new chunk generated from ``cur_state``.
        """
        prev_raw = np.asarray(prev_raw)
        if not self._rtc_config.anchor_correction:
            return prev_raw
        if self._norm_stats is None:
            if not self._warned_no_stats:
                import logging

                logging.getLogger(__name__).warning(
                    "anchor_correction enabled but no norm_stats provided; "
                    "falling back to unshifted prev chunk."
                )
                self._warned_no_stats = True
            return prev_raw

        actions_stats = self._norm_stats.get("actions")
        if actions_stats is None or actions_stats.q01 is None or actions_stats.q99 is None:
            return prev_raw

        dim = min(prev_raw.shape[-1], 14)
        q01 = np.asarray(actions_stats.q01, dtype=np.float32)[:dim]
        q99 = np.asarray(actions_stats.q99, dtype=np.float32)[:dim]
        scale = 2.0 / (q99 - q01 + 1e-6)
        shift = scale * (
            np.asarray(prev_state, dtype=np.float32)[:dim]
            - np.asarray(cur_state, dtype=np.float32)[:dim]
        )
        shift = shift * self._delta_mask[:dim]
        out = prev_raw.copy()
        out[..., :dim] = out[..., :dim] + shift
        return out

    @property
    def last_raw_chunk(self) -> np.ndarray | None:
        """Last generated chunk in the model's normalized space, (H, A)."""
        return self._last_raw_chunk

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def wrap_policy_for_rtc(policy, rtc_config: RTCConfig, norm_stats=None):
    """Wrap an openpi JAX policy so its model runs RTC-guided denoising."""
    if not rtc_config.enabled:
        return policy
    return RtcPolicy(policy, rtc_config, norm_stats=norm_stats)
