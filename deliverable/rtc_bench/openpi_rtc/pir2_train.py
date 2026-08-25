#!/usr/bin/env python3
"""πR² (Reactive Real-time Flow Policies) for the JAX openpi pi0/pi0.5 stack.

Faithful port of arXiv:2607.26055 (Park & Tulsiani, 2026), adapted to
openpi's pi05 convention (t=1 noise -> t=0 clean, ``u_t = noise - actions``).

v2 (slow channel + single-step streaming)
-----------------------------------------
1. Staircase per-position noise schedule (clean front d / ramp / noise tail d)
   used as the training-time diffusion-forcing schedule. The deployment delay
   ``d`` is sampled per sample in [1, max_delay] during training, so the model
   is latency-adaptive: at inference the staircase reshapes for whatever ``d``
   is measured on the robot.
2. Fast proprioception channel: pi05 embeds the state as discrete language
   tokens in the prefix (stale during denoising). πR² adds a *continuous*
   state token to the suffix at every denoising step via a new ``state_proj``
   parameter (randomly initialized, fine-tuned from 49999).
3. Slow channel (asynchronous vision/language): the prefix (images + prompt,
   including the tokenized state) is recomputed only every ``slow_refresh_every``
   ticks and cached as the LLM KV cache; every stream call runs ONE DiT step
   against the cached prefix + fresh fast state. Training simulates the
   staleness by sampling a per-sample image delay k in [0, image_delay_max]
   (the data pipeline feeds the stale frames) and conditioning the action
   expert on a learned delay embedding indexed by k (zero at k=0, i.e. a
   no-op, so the base-policy behaviour is preserved).

   Deviation from the paper: the paper adds the delay embedding to the slow
   representation itself; we add it (together with the per-position staircase
   time) to the action tokens, and keep the adaRMS cond per-sample (B, D) —
   the base pi0.5 contract — so openpi-main's gemma needs no modification.
   The conditioning is trained and deployed identically (stale prefix +
   embedding of its age), so the model learns the same "how stale is my
   vision" signal.
4. Latency-adaptive inference: instead of re-denosing a full chunk from
   scratch every call, deployment keeps a persistent buffer x_t (H, A) plus
   the staircase times and advances it by ONE Euler substep per call,
   releasing ``d`` clean actions and appending ``d`` fresh noise slots
   (paper Fig. 2 / Eq. 4). The buffer is warm-started with standard flow
   inference (multi-step denoise from pure noise), matching the 20% standard
   flow samples mixed into training.

Training entry (GPU machine, repo root):
    uv run python pir2_train.py --exp-name pir2_v2 \
        --max-delay 8 --image-delay-max 5 --slow-channel \
        --num-train-steps 10000 --fsdp-devices 2
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (_REPO_ROOT / "pyproject.toml").exists() and _REPO_ROOT.parent != _REPO_ROOT:
    _REPO_ROOT = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openpi_rtc.rtc_train import _posemb_sincos_batch  # noqa: E402

# Machine-specific paths via env vars / CLI args (no hardcoded machine paths).
#   export OPENPI05_CHECKPOINT_49999=/path/to/49999
#   export OPENPI05_RAW_TRAIN_DIR=/path/to/raw_hdf5_dir
CHECKPOINT_49999 = os.environ.get("OPENPI05_CHECKPOINT_49999", "")
DEFAULT_CONFIG = "pi05-task_00031_entong-xtrainer"


@dataclass
class Pir2Config:
    """πR² knobs. ``max_delay`` is the training latency budget; at inference
    any ``inference_delay <= max_delay`` works (latency-adaptive staircase)."""

    max_delay: int = 8
    # Ramp occupies [delay, horizon - delay); keep the tail as long as front.
    min_clean_ratio: float = 0.0
    enabled: bool = False
    # Deployment defaults (used by the sampler when not overridden).
    inference_delay: int = 7
    num_steps: int = 10

    # --- v2: async slow channel + single-step streaming ---
    # Slow channel: cached vision/language prefix, refreshed every
    # ``slow_refresh_every`` ticks; ``image_delay_max`` is the training
    # staleness budget (per-sample delay in [0, image_delay_max]).
    slow_channel: bool = False
    image_delay_max: int = 5
    slow_refresh_every: int = 5
    # Symmetric jitter on the staircase times (paper Sec. 3.3).
    time_jitter: float = 0.05
    # Probability of a plain flow-matching sample (no mask, shared time);
    # enables the standard-flow warm start of the stream buffer.
    standard_flow_prob: float = 0.2


_PIR2_STATE: dict[str, Any] = {"config": None}
_ORIGINAL_COMPUTE_LOSS_PIR2 = None


# ---------------------------------------------------------------------------
# staircase schedule
# ---------------------------------------------------------------------------
def staircase_time(delay: int, horizon: int, dtype=jnp.float32) -> jax.Array:
    """Per-position time levels for the latency-adaptive staircase.

    Returns shape (H,): positions < d -> 0 (clean), positions >= H-d -> 1
    (pure noise), the interior ramps linearly 0 -> 1 (pi0 convention: t=1 is
    noise, t=0 is the target). Interior follows the paper's Eq.(3):
    t_p = (p - d) / (H - 2d) for p in [d, H-d), which makes the schedule
    exactly invariant under the single-step slide in ``staircase_deltas``
    (valid for the operating range d <= H/3; our budget d <= max_delay is
    far below that for H=50).
    """
    d = int(min(max(delay, 0), horizon // 2))
    clean = jnp.zeros(d, dtype=dtype)
    tail = jnp.ones(d, dtype=dtype)
    ramp_len = horizon - 2 * d
    if ramp_len > 0:
        ramp = jnp.arange(ramp_len, dtype=dtype) / ramp_len
    else:
        ramp = jnp.zeros(0, dtype=dtype)
    return jnp.concatenate([clean, ramp, tail])


def staircase_matrix(max_delay: int, horizon: int) -> jax.Array:
    """Stack of staircase schedules for d=0..max_delay, shape (D+1, H).

    Static (Python-built) so it can be indexed by per-sample integer delays
    inside a traced function.
    """
    return jnp.stack([staircase_time(d, horizon) for d in range(max_delay + 1)])


def _staircase_for_delays(delays: jax.Array, max_delay: int, horizon: int) -> jax.Array:
    """Per-sample staircase times, shape (*B, H)."""
    return staircase_matrix(max_delay, horizon)[delays]


def staircase_deltas(delay: int, horizon: int) -> jax.Array:
    """Per-position time advance Δt for one πR² stream substep (Fig. 2/Eq. 4).

    After the substep the schedule has effectively slid right by ``delay``
    slots: positions [d, 2d) reach t=0 (clean, released/emitted), the interior
    ramp advances by the constant ramp slope, and the old tail moves onto the
    back of the ramp. The front [0, d) is the in-flight inpaint conditioning
    and does not move (Δt=0). Openpi convention: t decreases toward clean, so
    all advances are negative.
    """
    # Operating assumption: d <= H/3 keeps the three staircase regions
    # disjoint (ramp_len = H-2d >= d), which is what the derivation needs.
    # Our training/deployment budget (d <= max_delay <= 8, H=50) satisfies it.
    d = int(min(max(delay, 0), horizon // 3))
    s = 1.0 / max(horizon - 2 * d, 1)
    idx = jnp.arange(horizon, dtype=jnp.float32)
    dt = jnp.where(
        idx < d,
        0.0,
        jnp.where(
            idx < 2 * d,
            -(idx - d) * s,  # [d, 2d): reach t=0
            jnp.where(
                idx < horizon - d,
                -d * s,  # interior: constant ramp advance
                (idx - horizon) * s,  # tail: onto the back of the ramp
            ),
        ),
    )
    return dt


# ---------------------------------------------------------------------------
# slow-channel delay embedding
# ---------------------------------------------------------------------------
def pir2_delay_embedding(model, delay: jax.Array) -> jax.Array:
    """Learned per-delay embedding indexed by the slow-channel staleness.

    ``delay`` is an integer array (*B,); delay 0 is forced to exactly zero so
    a fresh prefix is a no-op and the base-policy behaviour is preserved.
    Returns (*B, width).
    """
    delay = jnp.asarray(delay)
    if delay.ndim == 0:
        delay = delay[None]
    width = model.action_in_proj.out_features
    dmax = int(getattr(model, "image_delay_max", 0))
    if dmax <= 0 or not hasattr(model, "slow_delay_embed"):
        return jnp.zeros((delay.shape[0], width), dtype=jnp.float32)
    onehot = jax.nn.one_hot(jnp.clip(delay, 0, dmax), dmax + 1, dtype=jnp.float32)
    emb = model.slow_delay_embed(onehot)
    return jnp.where((delay[..., None] == 0), 0.0, emb)


def ensure_slow_delay_embed(model, image_delay_max: int = 5, rngs=None) -> None:
    """Add the slow-channel delay embedding if missing (fine-tune-only param)."""
    image_delay_max = int(image_delay_max)
    model.image_delay_max = image_delay_max
    if image_delay_max <= 0:
        return
    if hasattr(model, "slow_delay_embed"):
        return
    width = model.action_in_proj.out_features
    model.slow_delay_embed = nnx.Linear(
        image_delay_max + 1,
        width,
        kernel_init=nnx.initializers.zeros,
        bias_init=nnx.initializers.zeros,
        rngs=rngs or nnx.Rngs(0),
    )


# ---------------------------------------------------------------------------
# suffix embedding with the fast proprioception state token
# ---------------------------------------------------------------------------
def pir2_embed_suffix(
    model,
    observation,
    x_t: jax.Array,
    timestep: jax.Array,
    state: jax.Array | None = None,
    slow_delay: jax.Array | None = None,
):
    """Per-position-time suffix embedder with a fresh state token prepended.

    Suffix layout: [state_token(1), action_tokens(H)]; the state token attends
    to the prefix only, action tokens attend causally to prefix+state+earlier
    actions. ``timestep`` is shape (*B, H) (per-action time levels); the state
    token gets time 0 (clean). ``slow_delay`` (*B,) optionally adds the learned
    slow-channel staleness embedding to the per-position time embedding.

    Conditioning contract: pi0.5's adaRMS only accepts a per-sample (B, D)
    cond (openpi-main 的 gemma 不支持 (B, T, D) 逐位置 cond，且我们不修改
    rtc_bench 之外的文件)。因此逐位置（阶梯/斜坡）时间**注入 action tokens**，
    adarms_cond 用逐样本代表时间（取末位置，非 stream 共享时间下即样本时间），
    保持 (B, D) —— 训练与推理共用本函数，自洽。
    """
    if not getattr(model, "pi05", True):
        raise NotImplementedError("πR² v1 requires pi05 models")
    if not hasattr(model, "state_proj"):
        raise AttributeError(
            "model has no state_proj; call ensure_state_proj() or patch "
            "Pi0.__init__ before creating the model"
        )
    if state is None:
        if observation is None:
            raise ValueError("pir2_embed_suffix needs either observation or state")
        state = observation.state

    batch = x_t.shape[0]
    action_tokens = model.action_in_proj(x_t)
    state_token = model.state_proj(state)[:, None, :]  # (B, 1, E)
    tokens = jnp.concatenate([state_token, action_tokens], axis=1)

    time_emb = _posemb_sincos_batch(
        timestep,
        model.action_in_proj.out_features,
        min_period=4e-3,
        max_period=4.0,
    )
    time_emb = model.time_mlp_in(time_emb)
    time_emb = nnx.swish(time_emb)
    time_emb = model.time_mlp_out(time_emb)
    time_emb = nnx.swish(time_emb)
    if slow_delay is not None:
        # Learned delay embedding: how stale the cached vision/language prefix
        # is (zero at delay=0). Added to the per-position time embedding.
        time_emb = time_emb + pir2_delay_embedding(model, slow_delay)[:, None, :]
    # 逐位置时间注入 action tokens；state token 不带时间（现在态）。
    action_tokens = action_tokens + time_emb
    tokens = jnp.concatenate([state_token, action_tokens], axis=1)

    # 逐样本代表时间 -> (B, D) cond（原始 pi0.5 契约，gemma 无需改动）。
    sample_time = timestep[..., -1:]
    cond_emb = _posemb_sincos_batch(
        sample_time,
        model.action_in_proj.out_features,
        min_period=4e-3,
        max_period=4.0,
    )
    cond_emb = model.time_mlp_in(cond_emb)
    cond_emb = nnx.swish(cond_emb)
    cond_emb = model.time_mlp_out(cond_emb)
    cond_emb = nnx.swish(cond_emb)
    adarms_cond = cond_emb[..., 0, :]  # (B, D)

    input_mask = jnp.ones((batch, 1 + model.action_horizon), dtype=jnp.bool_)
    ar_mask = jnp.concatenate(
        [
            jnp.ones(1, dtype=jnp.bool_),  # state token: prefix only
            jnp.ones(1, dtype=jnp.bool_),  # first action: prefix+state
            jnp.zeros(model.action_horizon - 1, dtype=jnp.bool_),
        ]
    )
    return tokens, input_mask, ar_mask, adarms_cond


def ensure_state_proj(model, rngs=None) -> None:
    """Add the fast-channel ``state_proj`` if missing (fine-tune-only param)."""
    if hasattr(model, "state_proj"):
        return
    width = model.action_in_proj.out_features
    model.state_proj = nnx.Linear(model.action_dim, width, rngs=rngs or nnx.Rngs(0))


# ---------------------------------------------------------------------------
# training loss (diffusion forcing with the staircase schedule)
# ---------------------------------------------------------------------------
def pir2_compute_loss(
    model,
    rng: jax.Array,
    observation,
    actions: jax.Array,
    max_delay: int,
    *,
    train: bool = True,
    config: Pir2Config | None = None,
) -> jax.Array:
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    cfg = config or Pir2Config(max_delay=max_delay)
    preprocess_rng, noise_rng, delay_rng, flow_rng, jitter_rng = jax.random.split(
        rng, 5
    )
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

    batch_shape = actions.shape[:-2]
    horizon = actions.shape[-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    use_standard = (
        cfg.standard_flow_prob > 0.0
        and jax.random.uniform(flow_rng, batch_shape) < cfg.standard_flow_prob
    )
    if max_delay > 0:
        delays = jax.random.randint(
            delay_rng, batch_shape, 1, max_delay + 1
        )  # latency-adaptive: d in [1, max_delay]
        time_pos = _staircase_for_delays(delays, max_delay, horizon)
        mask = time_pos > 1e-6  # clean front excluded from the loss
    else:
        delays = jnp.zeros(batch_shape, dtype=jnp.int32)
        time_pos = jnp.zeros(actions.shape[:-1], dtype=jnp.float32)
        mask = jnp.ones(actions.shape[:-1], dtype=jnp.bool_)

    if cfg.standard_flow_prob > 0.0:
        # Plain flow matching (shared time, all positions supervised): the same
        # network must be able to denoise a full chunk from pure noise, which
        # is how the stream buffer is warm-started at episode start.
        flow_rng, beta_rng = jax.random.split(flow_rng)
        t_std = jax.random.beta(beta_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_std = jnp.broadcast_to(t_std[..., None], actions.shape[:-1])
        time_pos = jnp.where(use_standard, time_std, time_pos)
        mask = jnp.where(use_standard, True, mask)

    if cfg.time_jitter > 0.0:
        # Symmetric jitter around the central staircase (paper Sec. 3.3) to
        # absorb per-call d variation. Only the supervised positions are
        # jittered: the clean front stays exactly clean (t=0), matching the
        # hard-clamped in-flight conditioning used at inference.
        delta = jax.random.uniform(
            jitter_rng,
            actions.shape[:-1],
            minval=-cfg.time_jitter,
            maxval=cfg.time_jitter,
        )
        time_pos = jnp.where(mask, jnp.clip(time_pos + delta, 0.0, 1.0), time_pos)

    x_t = time_pos[..., None] * noise + (1 - time_pos[..., None]) * actions
    u_t = noise - actions

    slow_delay = getattr(observation, "slow_delay", None)
    if slow_delay is None:
        slow_delay = jnp.zeros(batch_shape, dtype=jnp.int32)

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = pir2_embed_suffix(
        model, observation, x_t, time_pos, slow_delay=slow_delay
    )
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    del prefix_out
    v_t = model.action_out_proj(suffix_out[:, -horizon:])

    loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # (*B, H)
    loss_mask = mask.astype(jnp.float32)
    return jnp.sum(loss * loss_mask, axis=-1) / (jnp.sum(loss_mask, axis=-1) + 1e-8)


# ---------------------------------------------------------------------------
# inference sampler (v1: full multi-step denoising with hard-clamped front)
# ---------------------------------------------------------------------------
def pir2_sample_actions(
    self,
    rng,
    observation,
    *,
    num_steps: int = 10,
    noise=None,
    prev_chunk_left_over=None,
    inference_delay: int | None = None,
    execution_horizon: int | None = None,
):
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
    import einops

    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

    delay = int(inference_delay) if inference_delay is not None else 7
    prefix = None
    if prev_chunk_left_over is not None and delay > 0:
        prev = jnp.asarray(prev_chunk_left_over)
        if prev.ndim == 2:
            prev = prev[None, ...]
        if prev.shape[1] < delay:
            raise ValueError(
                f"prev_chunk_left_over length {prev.shape[1]} < inference_delay {delay}"
            )
        padded = jnp.zeros(
            (batch_size, self.action_horizon, self.action_dim), dtype=prev.dtype
        )
        padded = padded.at[:, : prev.shape[1], : prev.shape[2]].set(prev)
        prefix = padded
        noise = noise.at[:, :delay, :].set(padded[:, :delay, :])

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )

    def step(carry):
        x_t, time = carry
        time_pos = jnp.broadcast_to(time, (batch_size, self.action_horizon))
        if prefix is not None:
            mask = jnp.arange(self.action_horizon)[None, :] < delay
            x_t = jnp.where(mask[..., None], prefix, x_t)
            time_pos = jnp.where(mask, 0.0, time_pos)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = pir2_embed_suffix(
            self, observation, x_t, time_pos, slow_delay=None
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
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return x_t + dt * v_t, time + dt

    def cond(carry):
        _, time = carry
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    if prefix is not None:
        x_0 = x_0.at[:, :delay, :].set(prefix[:, :delay, :])
    return x_0


# ---------------------------------------------------------------------------
# v2: single-step streaming sampler (paper Fig. 2 / Eq. 4)
# ---------------------------------------------------------------------------
def pir2_stream_step(
    self,
    rng,
    state: jax.Array,
    x_t: jax.Array,
    time_pos: jax.Array,
    kv_cache,
    in_flight: jax.Array,
    prefix_mask: jax.Array,
    prefix_len: int,
    inference_delay: int,
    slow_delay: jax.Array,
):
    """One πR² stream substep + slide.

    Inputs (batch=1):
      state       (1, S)  fresh normalized proprioception (fast channel)
      x_t         (1, H, A) persistent denoising buffer
      time_pos    (1, H)   staircase times
      kv_cache    cached slow-channel prefix (vision/language)
      in_flight   (1, H, A) last emitted chunk; only [:d] is used as inpaint
                   conditioning
      prefix_mask (1, L)   cached prefix validity mask (True=valid token)
      prefix_len  Python int: number of valid prefix tokens (static)
      inference_delay  Python int d (static)
      slow_delay  (1,)   int age of the cached prefix in ticks

    Returns:
      emitted   (1, d, A) clean actions for the next d ticks
      x_t       (1, H, A) advanced + slid buffer (front d = emitted)
      time_pos  (1, H)    reproduced staircase
      kv_cache  unchanged
    """
    from openpi.models.pi0 import make_attn_mask
    import einops

    d = int(inference_delay)
    batch_size = state.shape[0]
    horizon = self.action_horizon
    dt = staircase_deltas(d, horizon)  # (H,)

    # Inpaint conditioning: the front d in-flight actions are fixed and clean.
    front_mask = jnp.arange(horizon)[None, :] < d
    x_t = jnp.where(front_mask[..., None], in_flight, x_t)
    time_pos = jnp.where(front_mask, 0.0, time_pos)

    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = pir2_embed_suffix(
        self, None, x_t, time_pos, state=state, slow_delay=slow_delay
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask_2 = einops.repeat(
        prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
    )
    full_attn_mask = jnp.concatenate([prefix_attn_mask_2, suffix_attn_mask], axis=-1)
    suffix_positions = (
        jnp.full((batch_size, 1), prefix_len, dtype=jnp.int32)
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
    v_t = self.action_out_proj(suffix_out[:, -horizon:])

    # One Euler substep with per-position advances (Eq. 4).
    x_t = x_t + dt[None, :, None] * v_t
    time_pos = jnp.clip(time_pos + dt, 0.0, 1.0)

    # Released: positions [d, 2d) reached t=0 (clean) -> emit.
    emitted = x_t[:, d : 2 * d]
    # Slide left by d and append d fresh noise slots at the back.
    fresh_noise = jax.random.normal(rng, (batch_size, d, self.action_dim))
    x_t = jnp.concatenate([x_t[:, d:], fresh_noise], axis=1)
    time_pos = jnp.broadcast_to(staircase_time(d, horizon), (batch_size, horizon))
    return emitted, x_t, time_pos, kv_cache


def pir2_refresh_prefix(self, observation):
    """Slow channel refresh: embed vision+language prefix and fill the KV cache.

    Returns ``(kv_cache, prefix_mask)``; the wrapper computes the concrete
    prefix length from the mask (it is a Python-level value, not traced).
    """
    from openpi.models.pi0 import make_attn_mask

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )
    return kv_cache, prefix_mask


# ---------------------------------------------------------------------------
# monkey-patching
# ---------------------------------------------------------------------------
def _pir2_compute_loss_patched(self, rng, observation, actions, *, train: bool = False):
    config = _PIR2_STATE["config"]
    if config is None or not config.enabled:
        return _ORIGINAL_COMPUTE_LOSS_PIR2(self, rng, observation, actions, train=train)
    return pir2_compute_loss(
        self, rng, observation, actions, config.max_delay, train=train, config=config
    )


def patch_pi0_for_pir2(config: Pir2Config) -> None:
    """Class-level patch: Pi0 gets a fast-channel state_proj + πR² loss."""
    from openpi.models.pi0 import Pi0

    global _ORIGINAL_COMPUTE_LOSS_PIR2
    if _ORIGINAL_COMPUTE_LOSS_PIR2 is None:
        _ORIGINAL_COMPUTE_LOSS_PIR2 = Pi0.compute_loss

    original_init = Pi0.__init__

    def _patched_init(self, config_, *, rngs):
        original_init(self, config_, rngs=rngs)
        if getattr(config_, "pi05", False) and not hasattr(self, "state_proj"):
            width = self.action_in_proj.out_features
            self.state_proj = nnx.Linear(self.action_dim, width, rngs=rngs)
        ensure_slow_delay_embed(
            self, config.image_delay_max if config.slow_channel else 0, rngs=rngs
        )

    Pi0.__init__ = _patched_init
    _PIR2_STATE["config"] = config
    Pi0.compute_loss = _pir2_compute_loss_patched
    Pi0.sample_actions = pir2_sample_actions


# ---------------------------------------------------------------------------
# deployment wrapper
# ---------------------------------------------------------------------------
def wrap_policy_for_pir2(
    policy,
    inference_delay: int = 7,
    norm_stats=None,
    *,
    slow_channel: bool = False,
    image_delay_max: int = 5,
    slow_refresh_every: int = 5,
    num_steps: int = 10,
):
    """Wrap a JAX policy with the πR² sampler.

    ``slow_channel=True`` additionally enables the asynchronous vision/language
    prefix cache: ``infer_stream`` runs ONE DiT step per call against the
    cached prefix + fresh fast state (paper's single-step emission).
    """
    from types import MethodType

    from openpi.shared import nnx_utils

    from openpi_rtc.integrate_openpi import RtcPolicy
    from openpi_rtc.rtc_config import RTCConfig

    wrapped = RtcPolicy(
        policy,
        RTCConfig(enabled=False, anchor_correction=True),
        norm_stats=norm_stats,
    )
    model = wrapped._model
    if slow_channel and not hasattr(model, "slow_delay_embed"):
        print(
            "[WARN] 当前 checkpoint 没有 slow_delay_embed（πR² v2 参数）；"
            "将以零初始化代替并随推理使用，慢通道行为会退化。"
            "请使用 pir2_v2 微调产物（--checkpoint 指向 .../pir2_v2/<step>）。"
        )
    if not hasattr(model, "state_proj"):
        print(
            "[WARN] 当前 checkpoint 没有 state_proj（πR² 快速本体通道参数，"
            "49999 基础模型没有）；已随机初始化，推理结果不可信。"
            "请先完成 pir2 微调。"
        )
    ensure_state_proj(model)
    ensure_slow_delay_embed(model, image_delay_max if slow_channel else 0)
    model.sample_actions = MethodType(pir2_sample_actions, model)
    jitted = nnx_utils.module_jit(
        model.sample_actions,
        static_argnames=("num_steps", "inference_delay"),
    )

    def capture(rng_or_device, observation, **kwargs):
        out = jitted(rng_or_device, observation, **kwargs)
        wrapped._last_raw_chunk = np.asarray(out[0])
        return out

    wrapped._sample_actions = capture
    wrapped._default_delay = int(inference_delay)
    original_infer = wrapped.infer

    def infer(obs, *, noise=None, prev_chunk_left_over=None, inference_delay=None, **kwargs):
        if inference_delay is None:
            inference_delay = wrapped._default_delay
        return original_infer(
            obs,
            noise=noise,
            prev_chunk_left_over=prev_chunk_left_over,
            inference_delay=inference_delay,
            **kwargs,
        )

    wrapped.infer = infer

    if not slow_channel:
        wrapped._slow_channel = False
        return wrapped

    # --- v2: async slow channel + single-step stream ----------------------
    wrapped._slow_channel = True
    wrapped._slow_refresh_every = max(1, int(slow_refresh_every))
    model.image_delay_max = max(0, int(image_delay_max))

    model.stream_step = MethodType(pir2_stream_step, model)
    model.refresh_prefix = MethodType(pir2_refresh_prefix, model)
    jit_step = nnx_utils.module_jit(
        model.stream_step, static_argnames=("prefix_len", "inference_delay")
    )
    jit_refresh = nnx_utils.module_jit(model.refresh_prefix)
    wrapped._jit_step = jit_step
    wrapped._jit_refresh = jit_refresh
    wrapped._slow = {
        "kv_cache": None,
        "age": 0,
        "x_t": None,
        "time": None,
        "in_flight": None,
        "warm_chunk": None,
        "cache_inputs": None,
        "prefix_len": 0,
        "cache_len": 0,
        "prefix_mask": None,
        "last_d": None,
    }
    wrapped._slow_rng = jax.random.key(0)
    wrapped._num_steps = int(num_steps)

    def _transform_inputs(obs):
        """Apply the policy's input transforms (normalize/pad/tokenize)."""
        import openpi.models.model as _model

        inputs = wrapped._input_transform({k: v for k, v in obs.items()})
        observation = _model.Observation.from_dict(inputs)
        return inputs, _model.preprocess_observation(None, observation, train=False)

    def refresh_slow(obs):
        """Recompute the cached slow-channel prefix from the latest obs."""
        inputs, observation = _transform_inputs(obs)
        kv_cache, prefix_mask = wrapped._jit_refresh(observation)
        wrapped._slow["kv_cache"] = kv_cache
        wrapped._slow["prefix_len"] = int(
            np.asarray(prefix_mask).sum(axis=-1).max()
        )
        wrapped._slow["cache_len"] = int(np.asarray(prefix_mask).shape[-1])
        wrapped._slow["prefix_mask"] = np.asarray(prefix_mask[0], dtype=bool)
        wrapped._slow["age"] = 0
        wrapped._slow["cache_inputs"] = inputs
        return inputs

    wrapped.refresh_slow = refresh_slow

    def warm_start(obs, *, d: int | None = None, num_steps: int | None = None):
        """Standard-flow warm start of the stream buffer (paper Sec. 3.3)."""
        d = int(d if d is not None else wrapped._default_delay)
        d = max(1, min(d, model.action_horizon // 3))
        num_steps = int(num_steps or wrapped._num_steps)
        inputs, observation = _transform_inputs(obs)
        wrapped._slow_rng, rng = jax.random.split(wrapped._slow_rng)
        out = wrapped._sample_actions(
            rng, observation, num_steps=num_steps, inference_delay=0
        )
        chunk = np.asarray(out[0])  # (H, A) model space
        H, A = chunk.shape
        t = np.asarray(staircase_time(d, H), dtype=np.float32)
        eps = np.random.default_rng(0).standard_normal((H, A)).astype(np.float32)
        x_t = (t[:, None] * eps + (1.0 - t[:, None]) * chunk).astype(np.float32)
        x_t[:d] = chunk[:d]
        t[:d] = 0.0
        wrapped._slow["warm_chunk"] = chunk
        wrapped._slow["x_t"] = x_t
        wrapped._slow["time"] = t
        in_flight = np.zeros((H, A), dtype=np.float32)
        in_flight[:d] = chunk[:d]
        wrapped._slow["in_flight"] = in_flight
        wrapped._slow["last_d"] = d
        wrapped._slow["age"] = 0
        return chunk

    wrapped.warm_start = warm_start

    def infer_stream(obs, *, inference_delay: int | None = None, warm: bool = False):
        """One πR² stream call: refresh if stale, then a single DiT step.

        ``warm=True`` (episode start) refreshes the slow channel, warm-starts
        the buffer with standard flow inference and returns the full warm
        chunk; subsequent calls emit ``d`` actions per single DiT step.

        Returns a dict with robot-unit actions (H, A; the first ``d`` are new)
        and the model-space raw chunk.
        """
        d = int(inference_delay if inference_delay is not None else wrapped._default_delay)
        d = max(1, min(d, model.action_horizon // 3))
        if warm or wrapped._slow.get("x_t") is None:
            refresh_slow(obs)
            chunk = warm_start(obs, d=d)
            wrapped._last_raw_chunk = chunk
            outputs = {"state": wrapped._slow["cache_inputs"]["state"], "actions": chunk}
            outputs = wrapped._output_transform(outputs)
            return {
                "actions": np.asarray(outputs["actions"], dtype=np.float32),
                "raw_actions": chunk,
                "inference_delay": d,
                "slow_age": 0,
                "warm": True,
                "refreshed": True,
            }
        refreshed = False
        if wrapped._slow.get("kv_cache") is None or wrapped._slow["age"] >= wrapped._slow_refresh_every:
            refresh_slow(obs)
            refreshed = True
        age = min(int(wrapped._slow["age"]), max(0, int(model.image_delay_max)))
        wrapped._slow["age"] += 1

        last_d = wrapped._slow.get("last_d")
        if last_d is not None and last_d != d:
            # Latency changed: pull the buffer toward the new staircase
            # (paper: schedule adapts when d changes between calls).
            wrapped._slow_rng, rng = jax.random.split(wrapped._slow_rng)
            x_cur = np.asarray(wrapped._slow["x_t"], dtype=np.float32)
            t_new = np.asarray(staircase_time(d, model.action_horizon), dtype=np.float32)
            eps = np.asarray(jax.random.normal(
                rng, (model.action_horizon, model.action_dim)
            ))
            x_new = (t_new[:, None] * eps + (1.0 - t_new[:, None]) * x_cur).astype(np.float32)
            x_new[:d] = np.asarray(wrapped._slow["in_flight"], dtype=np.float32)[:d]
            t_new[:d] = 0.0
            wrapped._slow["x_t"] = x_new
            wrapped._slow["time"] = t_new
            wrapped._slow["last_d"] = d

        inputs, _ = _transform_inputs(obs)
        state = np.asarray(inputs["state"], dtype=np.float32)[None, :]
        x_t = np.asarray(wrapped._slow["x_t"], dtype=np.float32)[None, ...]
        time_pos = np.asarray(wrapped._slow["time"], dtype=np.float32)[None, :]
        in_flight = np.asarray(wrapped._slow["in_flight"], dtype=np.float32)[None, ...]
        prefix_mask = np.asarray(wrapped._slow["prefix_mask"], dtype=bool)[None, :]
        wrapped._slow_rng, rng = jax.random.split(wrapped._slow_rng)
        emitted, x_new, t_new, kv_new = wrapped._jit_step(
            rng,
            jnp.asarray(state),
            jnp.asarray(x_t),
            jnp.asarray(time_pos),
            wrapped._slow["kv_cache"],
            jnp.asarray(in_flight),
            jnp.asarray(prefix_mask),
            prefix_len=int(wrapped._slow["prefix_len"]),
            inference_delay=d,
            slow_delay=jnp.asarray([age], dtype=jnp.int32),
        )
        emitted = np.asarray(emitted[0])  # (d, A)
        wrapped._slow["x_t"] = np.asarray(x_new[0])
        wrapped._slow["time"] = np.asarray(t_new[0])
        wrapped._slow["kv_cache"] = kv_new
        wrapped._slow["in_flight"] = np.concatenate(
            [emitted, np.repeat(emitted[-1:], model.action_horizon - d, axis=0)],
            axis=0,
        )
        chunk = wrapped._slow["in_flight"].copy()
        wrapped._last_raw_chunk = chunk
        outputs = {"state": inputs["state"], "actions": chunk}
        outputs = wrapped._output_transform(outputs)
        return {
            "actions": np.asarray(outputs["actions"], dtype=np.float32),
            "raw_actions": chunk,
            "inference_delay": d,
            "slow_age": age,
            "warm": False,
            "refreshed": refreshed,
        }

    wrapped.infer_stream = infer_stream
    return wrapped


# ---------------------------------------------------------------------------
# training entry
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--exp-name", default="pir2_v2")
    ap.add_argument("--num-train-steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--fsdp-devices", type=int, default=None)
    ap.add_argument("--save-interval", type=int, default=10000)
    ap.add_argument("--keep-period", type=int, default=30000)
    ap.add_argument("--checkpoint", default=CHECKPOINT_49999 or None)
    ap.add_argument("--raw-dir",
                    default=os.environ.get("OPENPI05_RAW_TRAIN_DIR", ""),
                    help="raw XTrainer HDF5 dir (for fresh-machine convert)")
    ap.add_argument("--dataset-repo-id", default="task_00031_entong_train")
    ap.add_argument("--prompt",
                    default="Transfer the test tube from the right rack to the left rack.")
    ap.add_argument(
        "--max-delay",
        type=int,
        default=8,
        help="πR² latency budget: training samples d in [1, max_delay]; "
             "deploy with any inference_delay <= max_delay",
    )
    ap.add_argument(
        "--slow-channel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="v2 async vision/language slow channel + single-step stream "
             "(paper πR²; default ON)",
    )
    ap.add_argument(
        "--image-delay-max",
        type=int,
        default=5,
        help="slow-channel staleness budget in ticks; training samples the "
             "image delay uniformly in [0, image_delay_max]",
    )
    ap.add_argument(
        "--time-jitter",
        type=float,
        default=0.05,
        help="symmetric jitter on staircase times (paper Sec. 3.3)",
    )
    ap.add_argument(
        "--standard-flow-prob",
        type=float,
        default=0.2,
        help="probability of a plain flow-matching sample (warm start)",
    )
    ap.add_argument("--wandb-enabled", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.max_delay < 2:
        raise ValueError("--max-delay must be >= 2")
    if args.slow_channel and args.image_delay_max < 0:
        raise ValueError("--image-delay-max must be >= 0")
    if not 0.0 <= args.standard_flow_prob <= 1.0:
        raise ValueError("--standard-flow-prob must be in [0, 1]")
    # Fail fast on missing/misconfigured data paths before any heavy work.
    from openpi_rtc.paths import require_checkpoint, require_hdf5_dir

    require_checkpoint(args.checkpoint)
    require_hdf5_dir(args.raw_dir, "原始数据目录")

    from openpi_rtc.rtc_train import ensure_dataset_and_norm_stats

    ensure_dataset_and_norm_stats(
        args.config,
        repo_id=args.dataset_repo_id,
        raw_dir=args.raw_dir,
        prompt=args.prompt,
        dry_run=args.dry_run,
    )

    print("=" * 70)
    print(f"πR²: config={args.config} exp={args.exp_name} "
          f"max_delay={args.max_delay} steps={args.num_train_steps}")
    print(f"  slow_channel={args.slow_channel} image_delay_max={args.image_delay_max} "
          f"time_jitter={args.time_jitter} standard_flow_prob={args.standard_flow_prob}")
    print(f"  checkpoint init: {args.checkpoint}")
    print("=" * 70)

    if args.dry_run:
        print("[DRY RUN] patching Pi0 + invoking scripts/train.py skipped")
        return 0

    patch_pi0_for_pir2(
        Pir2Config(
            max_delay=args.max_delay,
            enabled=True,
            slow_channel=args.slow_channel,
            image_delay_max=args.image_delay_max if args.slow_channel else 0,
            time_jitter=args.time_jitter,
            standard_flow_prob=args.standard_flow_prob,
        )
    )

    # Build the TrainConfig by hand: get_config() does not apply the tyro CLI
    # overrides, so replicate the argv from v1 as explicit replaces.
    import dataclasses
    from openpi.training import config as _config
    from openpi.training import weight_loaders as _weight_loaders

    cfg = _config.get_config(args.config)
    max_d_ok = cfg.model.action_horizon // 3
    if args.max_delay > max_d_ok:
        raise ValueError(
            f"--max-delay {args.max_delay} 超过单步流阶梯的数学假设 "
            f"H//3 = {max_d_ok}（H={cfg.model.action_horizon}）；请调小 "
            f"--max-delay 或增大 action_horizon"
        )
    cfg = dataclasses.replace(
        cfg,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        wandb_enabled=args.wandb_enabled,
        weight_loader=_weight_loaders.CheckpointWeightLoader(
            os.path.join(args.checkpoint, "params")),
    )
    if args.fsdp_devices is not None:
        cfg = dataclasses.replace(cfg, fsdp_devices=args.fsdp_devices)
    if args.slow_channel:
        cfg = dataclasses.replace(
            cfg,
            data=dataclasses.replace(
                cfg.data, slow_channel_delay_max=args.image_delay_max
            ),
        )

    from scripts import train as _train

    _train.main(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
