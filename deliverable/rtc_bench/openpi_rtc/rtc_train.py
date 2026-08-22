#!/usr/bin/env python3
"""Training-time RTC (simulated delay) for the JAX openpi pi0/pi0.5 stack.

Faithful port of Kinetix's train-time RTC (arXiv:2506.07339), adapted to
openpi's time convention (t=1 noise -> t=0 clean, ``u_t = noise - actions``).

Training loss (Kinetix ``model.py:FlowPolicy.loss`` with ``simulated_delay``):
    * a per-sample delay is drawn from [0, simulated_delay) with exponential
      weights ``w = exp([d-1, ..., 0])`` normalized (Kinetix exactly; note
      this puts the most mass on delay=0, i.e. most samples are standard flow
      matching, with a decaying chance of a larger in-flight prefix);
    * the first ``delay`` positions of the chunk are clamped to *clean*
      actions and get time=0 (the "already in flight" prefix);
    * the standard flow-matching loss is computed only on the remaining
      positions, normalized by the number of supervised positions.

Inference (Kinetix ``model.py:realtime_action``): at every denoising step the
first ``inference_delay`` positions are hard-clamped to the previous chunk's
in-flight actions (re-anchored to the current observation state) and their
time is fixed at 0 (clean), so only the tail is denoised.

The implementation monkey-patches ``Pi0.compute_loss`` / ``Pi0.sample_actions``
in-process, so the vendored openpi code stays untouched. The repository is
standalone (``uv sync`` at the repo root sets up the environment).

Usage (GPU machine, repo root):
    uv run python rtc_train.py --exp-name rtc_train_d6 \
        --simulated-delay 6 --num-train-steps 10000 --fsdp-devices 2
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
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

# Machine-specific paths are configurable via env vars or CLI args (no
# hardcoded machine paths). Set these on the GPU machine before training:
#   export OPENPI05_CHECKPOINT_49999=/path/to/49999
#   export OPENPI05_RAW_TRAIN_DIR=/path/to/raw_hdf5_dir
CHECKPOINT_49999 = os.environ.get("OPENPI05_CHECKPOINT_49999", "")
RAW_DATA_DIR = os.environ.get("OPENPI05_RAW_TRAIN_DIR", "")
DEFAULT_CONFIG = "pi05-task_00031_entong-xtrainer"
DATASET_REPO_ID = "task_00031_entong_train"
TASK_DESC = "Transfer the test tube from the right rack to the left rack."

# In-process patch state. The training entry sets this *before* the parent
# pipeline creates the model, so jax traces it as a Python-int constant.
_RTC_STATE: dict[str, Any] = {"simulated_delay": None}


# ---------------------------------------------------------------------------
# per-position time embedding (pi05 branch of Pi0.embed_suffix)
# ---------------------------------------------------------------------------
def _posemb_sincos_batch(
    pos: jax.Array,
    embedding_dim: int,
    min_period: float,
    max_period: float,
) -> jax.Array:
    """pi0 ``posemb_sincos`` generalized to leading batch dims."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "...i,j->...ij",
        pos,
        2 * jnp.pi / period,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def rtc_embed_suffix(model, observation, x_t: jax.Array, timestep: jax.Array):
    """Per-position-time suffix embedder, matching ``Pi0.embed_suffix`` (pi05).

    ``timestep`` has shape ``(*batch, H)`` (one noise level per action
    position). Returns the same tuple as ``Pi0.embed_suffix``.
    """
    if not getattr(model, "pi05", True):
        raise NotImplementedError("train-RTC currently requires pi05 models")

    action_tokens = model.action_in_proj(x_t)
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

    input_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
    ar_mask = jnp.concatenate(
        [
            jnp.ones(1, dtype=jnp.bool_),
            jnp.zeros(model.action_horizon - 1, dtype=jnp.bool_),
        ]
    )
    return action_tokens, input_mask, ar_mask, time_emb


# ---------------------------------------------------------------------------
# training loss
# ---------------------------------------------------------------------------
def sample_delay(rng: jax.Array, simulated_delay: int, batch_shape) -> jax.Array:
    """Per-sample delay in [0, simulated_delay), exponential bias to large d."""
    weights = jnp.exp(jnp.arange(simulated_delay - 1, -1, -1, dtype=jnp.float32))
    weights = weights / jnp.sum(weights)
    return jax.random.choice(rng, simulated_delay, batch_shape, p=weights)


def _prefix_mask(delay: jax.Array, horizon: int, batch_shape) -> jax.Array:
    """Boolean mask of the clamped (already in-flight) positions."""
    idx = jnp.arange(horizon)
    if len(batch_shape) == 0:
        return idx < delay
    return idx[(None,) * len(batch_shape) + (slice(None),)] < delay[..., None]


def rtc_compute_loss(
    model,
    rng: jax.Array,
    observation,
    actions: jax.Array,
    simulated_delay: int | None,
    *,
    train: bool = True,
) -> jax.Array:
    """Flow-matching loss with a simulated in-flight prefix (Kinetix train-RTC).

    Returns per-sample supervised loss (shape ``(*batch,)``); positions inside
    the clamped prefix are excluded and the loss is normalized by the number
    of supervised positions. ``simulated_delay=None`` reproduces the standard
    pi0 loss (per-sample scalar time, no clamping).
    """
    from openpi.models import model as _model

    preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

    batch_shape = actions.shape[:-2]
    horizon = actions.shape[-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_pos = jnp.broadcast_to(time[..., None], actions.shape[:-1])

    if simulated_delay is not None and simulated_delay > 0:
        delay = sample_delay(delay_rng, simulated_delay, batch_shape)
        mask = _prefix_mask(delay, horizon, batch_shape)
        # pi0 convention: t=0 is the clean target, so the in-flight prefix is
        # clamped to clean actions at time 0 (opposite of Kinetix t=1).
        time_pos = jnp.where(mask, 0.0, time_pos)
    else:
        mask = jnp.zeros(actions.shape[:-1], dtype=jnp.bool_)

    x_t = time_pos[..., None] * noise + (1 - time_pos[..., None]) * actions
    u_t = noise - actions

    from openpi.models.pi0 import make_attn_mask

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = rtc_embed_suffix(
        model, observation, x_t, time_pos
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

    loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # (*batch, H)
    if simulated_delay is not None and simulated_delay > 0:
        loss_mask = (~mask).astype(jnp.float32)
        return jnp.sum(loss * loss_mask, axis=-1) / (
            jnp.sum(loss_mask, axis=-1) + 1e-8
        )
    return jnp.mean(loss, axis=-1)


# ---------------------------------------------------------------------------
# inference sampler (hard clamp, train-RTC deployment)
# ---------------------------------------------------------------------------
def train_rtc_sample_actions(
    self,
    rng,
    observation,
    *,
    num_steps: int = 10,
    noise=None,
    prev_chunk_left_over=None,
    inference_delay: int | None = None,
):
    """``Pi0.sample_actions`` with train-RTC hard prefix clamping.

    ``prev_chunk_left_over`` is the re-anchored tail of the previous chunk
    (same convention as the guidance path): its first ``inference_delay``
    positions are the in-flight actions and are clamped into the new chunk at
    every denoising step, with time fixed at 0 (clean). With no prefix this is
    numerically identical to the original sampler.
    """
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
    import einops

    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

    delay = int(inference_delay) if inference_delay is not None else 0
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
        prefix = padded  # (B, H, A); only [:delay] is selected by the mask
        noise = noise.at[:, :delay, :].set(padded[:, :delay, :])

    # First fill the KV cache with a forward pass of the prefix.
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
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = rtc_embed_suffix(
            self, observation, x_t, time_pos
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
        # Robust to floating-point error.
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    if prefix is not None:
        # Re-clamp once more so the emitted chunk's prefix exactly matches the
        # in-flight actions (Kinetix never reads output[:d], but we do).
        x_0 = x_0.at[:, :delay, :].set(prefix[:, :delay, :])
    return x_0


# ---------------------------------------------------------------------------
# monkey-patching
# ---------------------------------------------------------------------------
def _rtc_compute_loss_patched(self, rng, observation, actions, *, train: bool = False):
    simulated_delay = _RTC_STATE["simulated_delay"]
    if simulated_delay is None or simulated_delay <= 0:
        return _ORIGINAL_COMPUTE_LOSS(self, rng, observation, actions, train=train)
    return rtc_compute_loss(
        self, rng, observation, actions, int(simulated_delay), train=train
    )


def patch_pi0_for_train_rtc(simulated_delay: int) -> None:
    """Class-level patch: every Pi0 created afterwards uses the RTC loss."""
    from openpi.models.pi0 import Pi0

    global _ORIGINAL_COMPUTE_LOSS
    if _ORIGINAL_COMPUTE_LOSS is None:
        _ORIGINAL_COMPUTE_LOSS = Pi0.compute_loss
    _RTC_STATE["simulated_delay"] = int(simulated_delay)
    Pi0.compute_loss = _rtc_compute_loss_patched
    Pi0.sample_actions = train_rtc_sample_actions


_ORIGINAL_COMPUTE_LOSS = None


# ---------------------------------------------------------------------------
# deployment wrapper (mirrors RtcPolicy but with the hard-clamp sampler)
# ---------------------------------------------------------------------------
def wrap_policy_for_train_rtc(policy, inference_delay: int, norm_stats=None):
    """Wrap a JAX policy with the train-RTC hard-clamp sampler.

    Returns an ``RtcPolicy``-compatible object: captures the raw chunk,
    re-anchors the previous chunk to the current observation state, and
    forwards ``prev_chunk_left_over`` / ``inference_delay`` to the sampler.
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
    model.sample_actions = MethodType(train_rtc_sample_actions, model)
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
def _run_command(cmd: list[str], description: str, dry_run: bool) -> None:
    import subprocess

    print(f"[{description}] {' '.join(cmd)}")
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with code {result.returncode}")


def ensure_dataset_and_norm_stats(
    config_name: str,
    repo_id: str = DATASET_REPO_ID,
    raw_dir: str = RAW_DATA_DIR,
    prompt: str = TASK_DESC,
    *,
    dry_run: bool = False,
) -> None:
    """Convert raw HDF5 -> LeRobot and compute norm stats if missing (fresh
    training machine with no shared filesystem)."""
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    if not raw_dir:
        raise SystemExit(
            "ERROR: 需要原始 HDF5 目录: 传 --raw-dir 或设 OPENPI05_RAW_TRAIN_DIR"
        )
    dataset_dir = HF_LEROBOT_HOME / repo_id
    if dataset_dir.exists():
        print(f"dataset {repo_id} already exists; skipping convert.")
    else:
        _run_command(
            [
                "uv", "run",
                "examples/xtrainer_real/convert_xtrainer_data_to_lerobot.py",
                "--raw-dir", raw_dir,
                "--repo-id", repo_id,
                "--task", prompt,
            ],
            "convert raw hdf5 -> lerobot",
            dry_run,
        )

    task_name = config_name.split("-", 1)[1].rsplit("-xtrainer", 1)[0]
    norm_stats_path = (
        _REPO_ROOT / "assets" / f"pi0-{task_name}-xtrainer" / repo_id / "norm_stats.json"
    )
    if norm_stats_path.exists():
        print(f"norm_stats already exists: {norm_stats_path}; skipping.")
    else:
        _run_command(
            ["uv", "run", "scripts/compute_norm_stats.py", "--config-name", config_name],
            "compute norm stats",
            dry_run,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--exp-name", default="rtc_train")
    ap.add_argument("--num-train-steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--fsdp-devices", type=int, default=None,
                    help="FSDP device count (49999 was trained on a 2-device mesh)")
    ap.add_argument("--save-interval", type=int, default=10000)
    ap.add_argument("--keep-period", type=int, default=30000)
    ap.add_argument("--checkpoint", default=CHECKPOINT_49999 or None)
    ap.add_argument("--raw-dir", default=RAW_DATA_DIR or None,
                    help="raw XTrainer HDF5 dir (for fresh-machine convert)")
    ap.add_argument("--dataset-repo-id", default=DATASET_REPO_ID)
    ap.add_argument("--prompt", default=TASK_DESC)
    ap.add_argument(
        "--simulated-delay",
        type=int,
        default=8,
        help="train-time RTC delay budget; covers deployment d up to "
             "simulated_delay - 1 (default 8 -> deploy d<=7)",
    )
    ap.add_argument("--wandb-enabled", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.simulated_delay < 2:
        raise ValueError("--simulated-delay must be >= 2 (covers d=0..d-1)")
    # Fail fast on missing/misconfigured data paths before any heavy work.
    from openpi_rtc.paths import require_checkpoint, require_hdf5_dir

    require_checkpoint(args.checkpoint)
    require_hdf5_dir(args.raw_dir, "原始数据目录")

    ensure_dataset_and_norm_stats(
        args.config,
        repo_id=args.dataset_repo_id,
        raw_dir=args.raw_dir,
        prompt=args.prompt,
        dry_run=args.dry_run,
    )

    print("=" * 70)
    print(f"train-RTC: config={args.config} exp={args.exp_name} "
          f"simulated_delay={args.simulated_delay} steps={args.num_train_steps}")
    print(f"  checkpoint init: {args.checkpoint}")
    print("=" * 70)

    if args.dry_run:
        print("[DRY RUN] patching Pi0 + invoking scripts/train.py skipped")
        return 0

    patch_pi0_for_train_rtc(args.simulated_delay)

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
