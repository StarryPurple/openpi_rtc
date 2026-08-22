#!/usr/bin/env python3
"""πR² (Reactive Real-time Flow Policies) for the JAX openpi pi0/pi0.5 stack.

Faithful-enough v1 of arXiv:2607.26055 (Park & Tulsiani, 2026), adapted to
openpi's pi05 convention (t=1 noise -> t=0 clean, ``u_t = noise - actions``):

1. Staircase per-position noise schedule (clean front d / ramp / noise tail d)
   used as the training-time diffusion-forcing schedule. The deployment delay
   ``d`` is sampled per sample in [1, max_delay] during training, so the model
   becomes latency-adaptive: at inference the staircase reshapes for whatever
   ``d`` is measured on the robot.
2. Fast proprioception channel: pi05 embeds the state as discrete language
   tokens in the prefix (stale during denoising). πR² adds a *continuous*
   state token to the suffix at every denoising step via a new ``state_proj``
   parameter (randomly initialized, fine-tuned from 49999).
3. The loss is masked over the clean front (already-in-flight actions), like
   train-RTC, but with the staircase time levels instead of a binary clamp.

The slow channel (async vision/language KV reuse + learned delay embedding) is
deployment wiring and is NOT implemented in v1; the prefix is recomputed per
inference call, which is correct but not yet "asynchronous".

Training entry (GPU machine, repo root):
    uv run python pir2_train.py --exp-name pir2_v1 \
        --max-delay 8 --num-train-steps 10000 --fsdp-devices 2
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass, field
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


_PIR2_STATE: dict[str, Any] = {"config": None}
_ORIGINAL_COMPUTE_LOSS_PIR2 = None


# ---------------------------------------------------------------------------
# staircase schedule
# ---------------------------------------------------------------------------
def staircase_time(delay: int, horizon: int, dtype=jnp.float32) -> jax.Array:
    """Per-position time levels for the latency-adaptive staircase.

    Returns shape (H,): positions < d -> 0 (clean), positions >= H-d -> 1
    (pure noise), the interior ramps linearly 0 -> 1 (pi0 convention: t=1 is
    noise, t=0 is the target).
    """
    d = int(min(max(delay, 0), horizon // 2))
    clean = jnp.zeros(d, dtype=dtype)
    tail = jnp.ones(d, dtype=dtype)
    ramp_len = horizon - 2 * d
    if ramp_len > 0:
        ramp = jnp.linspace(0.0, 1.0, ramp_len + 2, dtype=dtype)[1:-1]
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


# ---------------------------------------------------------------------------
# suffix embedding with the fast proprioception state token
# ---------------------------------------------------------------------------
def pir2_embed_suffix(model, observation, x_t: jax.Array, timestep: jax.Array, state: jax.Array):
    """Per-position-time suffix embedder with a fresh state token prepended.

    Suffix layout: [state_token(1), action_tokens(H)]; the state token attends
    to the prefix only, action tokens attend causally to prefix+state+earlier
    actions. ``timestep`` is shape (*B, H) (per-action time levels); the state
    token gets time 0 (clean).
    """
    if not getattr(model, "pi05", True):
        raise NotImplementedError("πR² v1 requires pi05 models")
    if not hasattr(model, "state_proj"):
        raise AttributeError(
            "model has no state_proj; call ensure_state_proj() or patch "
            "Pi0.__init__ before creating the model"
        )

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
    # The state token carries no denoising time (it is always "now").
    adarms_cond = jnp.concatenate(
        [jnp.zeros((batch, 1, time_emb.shape[-1]), dtype=time_emb.dtype), time_emb],
        axis=1,
    )

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
) -> jax.Array:
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    preprocess_rng, noise_rng, delay_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

    batch_shape = actions.shape[:-2]
    horizon = actions.shape[-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    if max_delay > 0:
        delays = jax.random.randint(
            delay_rng, batch_shape, 1, max_delay + 1
        )  # latency-adaptive: d in [1, max_delay]
        time_pos = _staircase_for_delays(delays, max_delay, horizon)
        mask = time_pos > 1e-6  # clean front excluded from the loss
    else:
        time = jax.random.beta(delay_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_pos = jnp.broadcast_to(time[..., None], actions.shape[:-1])
        mask = jnp.ones(actions.shape[:-1], dtype=jnp.bool_)

    x_t = time_pos[..., None] * noise + (1 - time_pos[..., None]) * actions
    u_t = noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = pir2_embed_suffix(
        model, observation, x_t, time_pos, observation.state
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
# inference sampler (fast proprio channel + optional hard-clamped front)
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
            self, observation, x_t, time_pos, observation.state
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
# monkey-patching
# ---------------------------------------------------------------------------
def _pir2_compute_loss_patched(self, rng, observation, actions, *, train: bool = False):
    config = _PIR2_STATE["config"]
    if config is None or not config.enabled:
        return _ORIGINAL_COMPUTE_LOSS_PIR2(self, rng, observation, actions, train=train)
    return pir2_compute_loss(self, rng, observation, actions, config.max_delay, train=train)


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

    Pi0.__init__ = _patched_init
    _PIR2_STATE["config"] = config
    Pi0.compute_loss = _pir2_compute_loss_patched
    Pi0.sample_actions = pir2_sample_actions


# ---------------------------------------------------------------------------
# deployment wrapper
# ---------------------------------------------------------------------------
def wrap_policy_for_pir2(policy, inference_delay: int = 7, norm_stats=None):
    """Wrap a JAX policy with the πR² sampler (fast state token + clamp)."""
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
    ensure_state_proj(model)
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
    return wrapped


# ---------------------------------------------------------------------------
# training entry
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--exp-name", default="pir2_v1")
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
    ap.add_argument("--wandb-enabled", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.max_delay < 2:
        raise ValueError("--max-delay must be >= 2")
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
    print(f"  checkpoint init: {args.checkpoint}")
    print("=" * 70)

    if args.dry_run:
        print("[DRY RUN] patching Pi0 + invoking scripts/train.py skipped")
        return 0

    patch_pi0_for_pir2(Pir2Config(max_delay=args.max_delay, enabled=True))

    argv = [
        "train.py",
        args.config,
        "--exp_name", args.exp_name,
        "--batch_size", str(args.batch_size),
        "--num_train_steps", str(args.num_train_steps),
        "--num_workers", str(args.num_workers),
        "--save_interval", str(args.save_interval),
        "--keep_period", str(args.keep_period),
        "--weight_loader.checkpoint_path", args.checkpoint,
        "--wandb_enabled", "true" if args.wandb_enabled else "false",
    ]
    if args.fsdp_devices is not None:
        argv += ["--fsdp_devices", str(args.fsdp_devices)]
    sys.argv = argv

    from openpi.training import config as _config
    from scripts import train as _train

    _train.main(_config.cli())
    return 0


if __name__ == "__main__":
    sys.exit(main())
