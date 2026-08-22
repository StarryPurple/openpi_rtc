#!/usr/bin/env python3
"""Benchmark the JAX pi0.5 inference prefix and every flow-matching step.

The benchmark mirrors ``Pi0.sample_actions`` but deliberately keeps the prefix
and denoising step as two separately-jitted functions.  This lets us synchronize
the GPU at each boundary and report device-complete wall time for every step.
JIT compilation and checkpoint loading are reported separately and are never
included in steady-state latency statistics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import statistics
import time
from typing import Any

import einops
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models import pi0 as pi0_lib
from openpi.models import pi0_config

DEFAULT_CHECKPOINT = pathlib.Path(
    "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi/"
    ".cache/openpi-assets/checkpoints/pi05_base/params"
)


@dataclasses.dataclass(frozen=True)
class SimulatedInputConfig:
    """Shape/content metadata for the simulated robot observation."""

    batch_size: int
    cameras: int
    image_height: int
    image_width: int
    state_dim: int
    prompt_array_length: int
    prompt_valid_tokens: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-valid-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts/pi05_inference_timing.json"))
    return parser.parse_args()


def _block_until_ready(tree: Any) -> Any:
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return tree


def _timed_call(fn, *args):
    start_ns = time.perf_counter_ns()
    result = fn(*args)
    _block_until_ready(result)
    return result, (time.perf_counter_ns() - start_ns) / 1e6


def _summary(values: list[float]) -> dict[str, float]:
    values_sorted = sorted(values)
    percentile_95_index = max(0, int(np.ceil(0.95 * len(values_sorted))) - 1)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "p95_ms": values_sorted[percentile_95_index],
        "std_ms": statistics.pstdev(values),
    }


def _make_simulated_observation(config: pi0_config.Pi0Config, args: argparse.Namespace):
    """Create three normalized RGB camera streams plus state/prompt tensors."""
    rng = np.random.default_rng(args.seed)
    image_shape = (args.batch_size, *model_lib.IMAGE_RESOLUTION, 3)
    images = {
        key: jnp.asarray(rng.uniform(-1.0, 1.0, image_shape).astype(np.float32))
        for key in model_lib.IMAGE_KEYS
    }
    image_masks = {key: jnp.ones((args.batch_size,), dtype=jnp.bool_) for key in model_lib.IMAGE_KEYS}
    state = jnp.asarray(rng.normal(size=(args.batch_size, config.action_dim)).astype(np.float32))

    # The production tokenizer pads pi0.5 prompts to max_token_len.  Arbitrary
    # in-vocabulary IDs are sufficient for a latency benchmark; the tensor shape
    # and valid-token mask match the real model interface.
    valid_tokens = min(args.prompt_valid_tokens, config.max_token_len)
    prompt = np.zeros((args.batch_size, config.max_token_len), dtype=np.int32)
    prompt[:, :valid_tokens] = rng.integers(1, 256_000, size=(args.batch_size, valid_tokens), dtype=np.int32)
    prompt_mask = np.zeros_like(prompt, dtype=np.bool_)
    prompt_mask[:, :valid_tokens] = True

    observation = model_lib.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=jnp.asarray(prompt),
        tokenized_prompt_mask=jnp.asarray(prompt_mask),
    )
    metadata = SimulatedInputConfig(
        batch_size=args.batch_size,
        cameras=len(images),
        image_height=model_lib.IMAGE_RESOLUTION[0],
        image_width=model_lib.IMAGE_RESOLUTION[1],
        state_dim=config.action_dim,
        prompt_array_length=config.max_token_len,
        prompt_valid_tokens=valid_tokens,
    )
    return _block_until_ready(observation), metadata


def _make_phase_functions(model):
    """Build phase-level JIT functions with the same math as Pi0.sample_actions."""
    graphdef, state = nnx.split(model)

    def preprocess_impl(observation):
        return model_lib.preprocess_observation(None, observation, train=False)

    def prefix_impl(model_state, observation):
        phase_model = nnx.merge(graphdef, model_state)
        prefix_tokens, prefix_mask, prefix_ar_mask = phase_model.embed_prefix(observation)
        prefix_attn_mask = pi0_lib.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = phase_model.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        return prefix_mask, kv_cache

    def denoise_step_core(phase_model, observation, prefix_mask, kv_cache, x_t, timestep, dt):
        batch_size = observation.state.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = phase_model.embed_suffix(
            observation, x_t, jnp.broadcast_to(timestep, batch_size)
        )
        suffix_attn_mask = pi0_lib.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = phase_model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        velocity = phase_model.action_out_proj(suffix_out[:, -phase_model.action_horizon :])
        return x_t + dt * velocity

    def denoise_step_impl(model_state, observation, prefix_mask, kv_cache, x_t, timestep, dt):
        phase_model = nnx.merge(graphdef, model_state)
        return denoise_step_core(phase_model, observation, prefix_mask, kv_cache, x_t, timestep, dt)

    def flow_loop_impl(model_state, observation, prefix_mask, kv_cache, noise, num_steps):
        phase_model = nnx.merge(graphdef, model_state)
        dt = jnp.asarray(-1.0 / num_steps, dtype=jnp.float32)

        def body(step_index, x_t):
            timestep = jnp.asarray(1.0, dtype=jnp.float32) + step_index * dt
            return denoise_step_core(phase_model, observation, prefix_mask, kv_cache, x_t, timestep, dt)

        return jax.lax.fori_loop(0, num_steps, body, noise)

    return (
        state,
        jax.jit(preprocess_impl),
        jax.jit(prefix_impl),
        jax.jit(denoise_step_impl),
        jax.jit(flow_loop_impl, static_argnums=(5,)),
    )


def _run_once(state, preprocess_fn, prefix_fn, denoise_step_fn, flow_loop_fn, observation, noise, num_steps):
    processed, preprocess_ms = _timed_call(preprocess_fn, observation)
    (prefix_mask, kv_cache), vlm_prefix_ms = _timed_call(prefix_fn, state, processed)

    dt = jnp.asarray(-1.0 / num_steps, dtype=jnp.float32)
    x_t = noise
    step_ms = []
    for step_index in range(num_steps):
        timestep = jnp.asarray(1.0 - step_index / num_steps, dtype=jnp.float32)
        x_t, elapsed_ms = _timed_call(
            denoise_step_fn, state, processed, prefix_mask, kv_cache, x_t, timestep, dt
        )
        step_ms.append(elapsed_ms)

    flow_actions, flow_loop_ms = _timed_call(
        flow_loop_fn, state, processed, prefix_mask, kv_cache, noise, num_steps
    )

    return flow_actions, {
        "preprocess_ms": preprocess_ms,
        "vlm_prefix_ms": vlm_prefix_ms,
        "denoise_step_ms": step_ms,
        "instrumented_step_sum_ms": sum(step_ms),
        "flow_matching_loop_ms": flow_loop_ms,
        "phased_model_total_ms": vlm_prefix_ms + flow_loop_ms,
        "phased_with_preprocess_total_ms": preprocess_ms + vlm_prefix_ms + flow_loop_ms,
        "instrumented_vs_loop_max_abs_diff": float(np.asarray(jnp.max(jnp.abs(x_t - flow_actions)))),
    }


def main() -> None:
    args = _parse_args()
    if args.num_steps <= 0 or args.warmup < 1 or args.repeats < 1 or args.batch_size < 1:
        raise ValueError("num-steps, warmup, repeats, and batch-size must all be positive")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {args.checkpoint}")

    devices = jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"A CUDA GPU is required for meaningful inference timing; found {devices}")

    config = pi0_config.Pi0Config(pi05=True)
    load_start = time.perf_counter()
    params = model_lib.restore_params(args.checkpoint, dtype=jnp.bfloat16)
    model = config.load(params)
    _block_until_ready(model)
    checkpoint_load_s = time.perf_counter() - load_start

    observation, input_metadata = _make_simulated_observation(config, args)
    noise_rng = jax.random.key(args.seed + 1)
    noise = jax.random.normal(
        noise_rng, (args.batch_size, config.action_horizon, config.action_dim), dtype=jnp.float32
    )
    _block_until_ready(noise)
    state, preprocess_fn, prefix_fn, denoise_step_fn, flow_loop_fn = _make_phase_functions(model)

    # The first call includes XLA compilation.  Keep its time visible but exclude
    # all warmups from the steady-state statistics below.
    _, compile_timing = _run_once(
        state, preprocess_fn, prefix_fn, denoise_step_fn, flow_loop_fn, observation, noise, args.num_steps
    )
    for _ in range(args.warmup - 1):
        _run_once(
            state, preprocess_fn, prefix_fn, denoise_step_fn, flow_loop_fn, observation, noise, args.num_steps
        )

    runs = []
    final_actions = None
    for repeat_index in range(args.repeats):
        final_actions, timing = _run_once(
            state, preprocess_fn, prefix_fn, denoise_step_fn, flow_loop_fn, observation, noise, args.num_steps
        )
        timing["repeat"] = repeat_index
        runs.append(timing)

    step_summaries = [
        _summary([run["denoise_step_ms"][step] for run in runs]) for step in range(args.num_steps)
    ]
    summary = {
        "preprocess": _summary([run["preprocess_ms"] for run in runs]),
        "vlm_prefix": _summary([run["vlm_prefix_ms"] for run in runs]),
        "flow_matching_loop": _summary([run["flow_matching_loop_ms"] for run in runs]),
        "instrumented_step_sum": _summary([run["instrumented_step_sum_ms"] for run in runs]),
        "phased_model_total": _summary([run["phased_model_total_ms"] for run in runs]),
        "phased_with_preprocess_total": _summary(
            [run["phased_with_preprocess_total_ms"] for run in runs]
        ),
        "denoise_steps": step_summaries,
        "denoise_step_all": _summary(
            [value for run in runs for value in run["denoise_step_ms"]]
        ),
    }
    result = {
        "benchmark": "pi0.5 JAX phased inference latency",
        "timing_semantics": (
            "perf_counter wall time after each JAX result is block_until_ready; flow_matching_loop is the "
            "production-equivalent continuous loop; denoise_step_ms synchronizes every step for observability; "
            "steady-state summary excludes checkpoint loading, compilation, and warmup"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_load_s": checkpoint_load_s,
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "model": {
            "pi05": config.pi05,
            "dtype": config.dtype,
            "action_horizon": config.action_horizon,
            "action_dim": config.action_dim,
            "num_flow_steps": args.num_steps,
        },
        "simulated_input": dataclasses.asdict(input_metadata),
        "warmup_calls": args.warmup,
        "measured_repeats": args.repeats,
        "first_call_including_compilation_ms": compile_timing,
        "summary": summary,
        "runs": runs,
        "output_sanity": {
            "shape": list(final_actions.shape),
            "all_finite": bool(np.asarray(jnp.all(jnp.isfinite(final_actions)))),
            "mean": float(np.asarray(jnp.mean(final_actions))),
            "std": float(np.asarray(jnp.std(final_actions))),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"device: {', '.join(result['devices'])}")
    print(f"checkpoint load: {checkpoint_load_s:.2f} s (excluded)")
    print(f"VLM prefix: {summary['vlm_prefix']['mean_ms']:.3f} ms mean")
    print(
        f"flow matching continuous loop ({args.num_steps} steps): "
        f"{summary['flow_matching_loop']['mean_ms']:.3f} ms mean"
    )
    for index, step_summary in enumerate(step_summaries, start=1):
        timestep = 1.0 - (index - 1) / args.num_steps
        print(
            f"  step {index:02d}/{args.num_steps:02d}, t={timestep:.3f}: "
            f"{step_summary['mean_ms']:.3f} ms mean, {step_summary['median_ms']:.3f} ms median"
        )
    print(f"phased model total: {summary['phased_model_total']['mean_ms']:.3f} ms mean")
    print(f"JSON: {args.output.resolve()}")


if __name__ == "__main__":
    main()
