"""RTC guidance (inpainting) core, JAX implementation.

Reference: "Real-Time Execution of Action Chunking Flow Policies"
(arXiv:2506.07339). The guidance math follows Physical Intelligence's
original Kinetix implementation (``jax.vjp`` over ``x1(x_t)``, see
``references/kinetix_model.py``) and huggingface/lerobot's port
(``references/modeling_rtc.py``), adapted to the openpi JAX time convention.

Time convention (same as ``openpi.models.pi0.Pi0.sample_actions``):
``time`` runs from 1 (noise) down to 0 (clean), the Euler update is
``x_t += dt * v_t`` with ``dt = -1 / num_steps``, and the predicted clean
chunk is ``x1 = x_t - time * v_t``.

The correction term uses ``jax.vjp`` on ``x1(x_t)`` so that the *full*
Jacobian ``d x1 / d x_t`` (including the velocity's own dependence on
``x_t``) is applied, exactly like Kinetix. Note that lerobot's eager-torch
port computes ``v_t`` *before* ``x.requires_grad_(True)``, which silently
drops the ``dv/dx`` term and reduces the correction to the identity part;
this implementation deliberately keeps the full Jacobian (the sign of the
correction is chosen accordingly: ``v' = v - w * J^T err`` with
``err = (prev - x1) * weights``).
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp

from .rtc_config import RTCAttentionSchedule, RTCConfig


class RTCProcessor:
    def __init__(self, rtc_config: RTCConfig):
        self.rtc_config = rtc_config

    # ------------------------------------------------------------------
    # guidance
    # ------------------------------------------------------------------
    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay: int,
        time,
        denoise_fn: Callable,
        execution_horizon: int | None = None,
    ):
        """Wrap one denoising step with RTC prefix guidance.

        Args:
            x_t: current noisy chunk, shape (H, A) or (B, H, A).
            prev_chunk_left_over: unconsumed actions of the previous chunk in
                the model's normalized space, shape (H_prev, A) or (B, H_prev, A);
                ``None`` disables guidance.
            inference_delay: number of prefix steps frozen by the soft mask (d).
            time: scalar timestep in [1, 0].
            denoise_fn: callable ``x_t -> v_t`` (one model denoising step).
            execution_horizon: soft-mask blend horizon; defaults to config.

        Returns:
            The guided velocity (same shape as ``x_t``).
        """
        if prev_chunk_left_over is None:
            return denoise_fn(x_t)

        batched = x_t.ndim == 3
        x = x_t if batched else x_t[None, ...]
        prev = (
            prev_chunk_left_over
            if prev_chunk_left_over.ndim == 3
            else prev_chunk_left_over[None, ...]
        )

        horizon = (
            self.rtc_config.execution_horizon
            if execution_horizon is None
            else execution_horizon
        )
        horizon = min(int(horizon), prev.shape[1])
        delay = min(int(inference_delay), horizon)

        batch_size, chunk_size, action_dim = x.shape
        if prev.shape[1] < chunk_size or prev.shape[2] < action_dim:
            padded = jnp.zeros((batch_size, chunk_size, action_dim), dtype=x.dtype)
            padded = padded.at[:, : prev.shape[1], : prev.shape[2]].set(prev)
            prev = padded

        weights = self.get_prefix_weights(delay, horizon, chunk_size).astype(
            x.dtype
        )[None, :, None]

        def x1_fn(z):
            # Predicted clean chunk: x1 = z - time * v(z).
            return z - time * denoise_fn(z)

        x1, f_vjp = jax.vjp(x1_fn, x)
        v_t = (x - x1) / time
        err = (prev - x1) * weights
        correction = f_vjp(err)[0]

        tau = 1.0 - time  # paper's flow time: 0 -> 1 as the chunk cleans up
        guidance_weight = self._guidance_weight(tau, dtype=x.dtype)
        guided = v_t - guidance_weight * correction
        return guided if batched else guided[0]

    def _guidance_weight(self, tau, max_guidance_weight: float | None = None, dtype=jnp.float32):
        """c * r^{-2} clipped at kappa (lerobot's formula)."""
        kappa = (
            self.rtc_config.max_guidance_weight
            if max_guidance_weight is None
            else max_guidance_weight
        )
        tau = jnp.asarray(tau, jnp.float32)
        kappa = jnp.asarray(kappa, jnp.float32)
        one_minus_tau_sq = (1.0 - tau) ** 2
        inv_r2 = jnp.where(
            one_minus_tau_sq > 1e-8,
            (one_minus_tau_sq + tau**2) / jnp.maximum(one_minus_tau_sq, 1e-8),
            kappa,
        )
        c = jnp.where(tau > 1e-8, (1.0 - tau) / jnp.maximum(tau, 1e-8), kappa)
        return jnp.minimum(c * inv_r2, kappa).astype(dtype)

    # ------------------------------------------------------------------
    # prefix attention weights (soft mask)
    # ------------------------------------------------------------------
    def get_prefix_weights(self, start: int, end: int, total: int):
        start = min(int(start), int(end))
        schedule = self.rtc_config.prefix_attention_schedule
        if schedule == RTCAttentionSchedule.ZEROS:
            weights = jnp.zeros(total)
            return weights.at[:start].set(1.0)
        if schedule == RTCAttentionSchedule.ONES:
            weights = jnp.ones(total)
            return weights.at[end:].set(0.0)
        if schedule in (RTCAttentionSchedule.LINEAR, RTCAttentionSchedule.EXP):
            lin_weights = self._linweights(start, end, total)
            if schedule == RTCAttentionSchedule.EXP:
                lin_weights = lin_weights * jnp.expm1(lin_weights) / (math.e - 1)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            return self._add_leading_ones(weights, start, total)
        raise ValueError(f"unknown schedule: {schedule}")

    def _linweights(self, start: int, end: int, total: int):
        skip = max(total - end, 0)
        steps = total - skip - start
        if end <= start or steps <= 0:
            return jnp.zeros(0)
        return jnp.linspace(1, 0, steps + 2)[1:-1]

    def _add_trailing_zeros(self, weights, total: int, end: int):
        zeros_len = total - end
        if zeros_len <= 0:
            return weights
        return jnp.concatenate([weights, jnp.zeros(zeros_len)])

    def _add_leading_ones(self, weights, start: int, total: int):
        ones_len = min(start, total)
        if ones_len <= 0:
            return weights
        return jnp.concatenate([jnp.ones(ones_len), weights])
