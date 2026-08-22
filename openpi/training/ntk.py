"""Empirical NTK utilities for JAX pi0/pi05 training states."""

from __future__ import annotations

from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.model as _model
from openpi.models.pi0 import make_attn_mask
import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.training.utils as training_utils


class NTKResult(NamedTuple):
    """Empirical NTK result for selected samples."""

    gram: at.Float[at.Array, "n n"]
    cosine: at.Float[at.Array, "n n"]
    grad_norms: at.Float[at.Array, " n"]


def _leaf_value(x):
    return x.value if hasattr(x, "value") else x


def _tree_dot(a, b) -> at.Array:
    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    return sum(jnp.vdot(_leaf_value(x), _leaf_value(y)) for x, y in zip(leaves_a, leaves_b, strict=True))


def _take_sample(batch: tuple[_model.Observation, _model.Actions], index: int) -> tuple[_model.Observation, _model.Actions]:
    return jax.tree.map(lambda x: x[index : index + 1], batch)


def _rademacher(key: at.KeyArrayLike, shape: tuple[int, ...]) -> at.Array:
    return jnp.where(jax.random.bernoulli(key, shape=shape), 1.0, -1.0).astype(jnp.float32)


def _make_noise(noise_mode: str, rng: at.KeyArrayLike, action_shape: tuple[int, ...]) -> at.Array:
    if noise_mode == "zeros":
        return jnp.zeros(action_shape, dtype=jnp.float32)
    if noise_mode == "ones":
        return jnp.ones(action_shape, dtype=jnp.float32)
    if noise_mode == "shared_normal":
        return jax.random.normal(rng, action_shape, dtype=jnp.float32)
    raise ValueError(f"Unsupported noise_mode: {noise_mode!r}. Use zeros, ones, or shared_normal.")


def _pi0_flow_prediction(
    model: _model.BaseModel,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
    *,
    train: bool,
    noise: _model.Actions,
    time: at.Float[at.Array, " b"],
) -> _model.Actions:
    """Standalone pi0/pi05 flow-prediction forward pass for NTK analysis."""

    required_attrs = ("embed_prefix", "embed_suffix", "PaliGemma", "action_out_proj", "action_horizon")
    if not all(hasattr(model, attr) for attr in required_attrs):
        raise TypeError("NTK flow prediction currently supports pi0/pi05-style models only.")

    observation = _model.preprocess_observation(rng, observation, train=train)
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)  # pyright: ignore[reportAttributeAccessIssue]
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(  # pyright: ignore[reportAttributeAccessIssue]
        observation,
        x_t,
        time,
    )
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(  # pyright: ignore[reportAttributeAccessIssue]
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    return model.action_out_proj(suffix_out[:, -model.action_horizon :])  # pyright: ignore[reportAttributeAccessIssue]


def output_projection_grad(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    index: int,
    rng: at.KeyArrayLike,
    projection: at.Float[at.Array, "ah ad"],
    noise: at.Float[at.Array, "ah ad"],
    time: at.Float[at.Array, ""],
    *,
    train: bool = False,
) -> nnx.State:
    """Gradient of a scalar random projection of the model output.

    For pi0/pi05, the network output is the flow prediction ``v_t``. This is an
    output-Jacobian gradient, not a loss gradient.
    """

    model = nnx.merge(state.model_def, state.params)
    if train:
        model.train()
    else:
        model.eval()

    observation, actions = _take_sample(batch, index)
    noise = noise[None, ...]
    time = jnp.broadcast_to(time, actions.shape[:-2])

    def scalar_output(
        model: _model.BaseModel,
        preprocess_rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        noise: _model.Actions,
        time: at.Float[at.Array, " b"],
        projection: at.Float[at.Array, "ah ad"],
    ):
        flow = _pi0_flow_prediction(
            model,
            preprocess_rng,
            observation,
            actions,
            train=train,
            noise=noise,
            time=time,
        )
        return jnp.sum(flow[0] * projection)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    _, grads = nnx.value_and_grad(scalar_output, argnums=diff_state)(
        model,
        rng,
        observation,
        actions,
        noise,
        time,
        projection,
    )
    return grads


def empirical_ntk_matrix(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    rng: at.KeyArrayLike,
    *,
    indices: tuple[int, ...] | None = None,
    num_projections: int = 1,
    fixed_t: float = 0.5,
    noise_mode: str = "zeros",
    cache_grads: bool = False,
    train: bool = False,
    eps: float = 1e-12,
) -> NTKResult:
    """Estimate the empirical NTK trace between selected samples.

    ``gram[i, j]`` estimates ``trace(J_i J_j^T)``, where ``J_i`` is the
    Jacobian of the model output ``v_t`` for sample ``i`` with respect to the
    trainable parameters. Random Rademacher projections avoid materializing the
    full output Jacobian. The flow-matching noise and timestep are fixed across
    all samples so they do not become an extra source of sample variation.
    """

    if num_projections < 1:
        raise ValueError("--num_projections must be >= 1.")
    if not 0.0 <= fixed_t <= 1.0:
        raise ValueError("--fixed_t must be in [0, 1].")

    num_samples = batch[1].shape[0]
    indices = tuple(range(num_samples)) if indices is None else indices
    action_shape = batch[1].shape[-2:]
    time = jnp.asarray(fixed_t, dtype=jnp.float32)
    noise = _make_noise(noise_mode, jax.random.fold_in(rng, 50_000), action_shape)

    gram = None
    for projection_index in range(num_projections):
        projection_rng = jax.random.fold_in(rng, 100_000 + projection_index)
        projection = _rademacher(projection_rng, action_shape)

        if cache_grads:
            grads = []
            for position, index in enumerate(indices):
                grad_rng = jax.random.fold_in(rng, projection_index * 10_000 + position)
                grads.append(
                    output_projection_grad(
                        config,
                        state,
                        batch,
                        index,
                        grad_rng,
                        projection,
                        noise,
                        time,
                        train=train,
                    )
                )
            projection_gram = jnp.stack([jnp.stack([_tree_dot(left, right) for right in grads]) for left in grads])
        else:
            n = len(indices)
            projection_gram = jnp.zeros((n, n), dtype=jnp.float32)
            for left_position, left_index in enumerate(indices):
                left_rng = jax.random.fold_in(rng, projection_index * 10_000 + left_position)
                left_grad = output_projection_grad(
                    config,
                    state,
                    batch,
                    left_index,
                    left_rng,
                    projection,
                    noise,
                    time,
                    train=train,
                )
                for right_position in range(left_position, n):
                    if right_position == left_position:
                        right_grad = left_grad
                    else:
                        right_rng = jax.random.fold_in(rng, projection_index * 10_000 + right_position)
                        right_grad = output_projection_grad(
                            config,
                            state,
                            batch,
                            indices[right_position],
                            right_rng,
                            projection,
                            noise,
                            time,
                            train=train,
                        )
                    value = _tree_dot(left_grad, right_grad)
                    projection_gram = projection_gram.at[left_position, right_position].set(value)
                    projection_gram = projection_gram.at[right_position, left_position].set(value)
        gram = projection_gram if gram is None else gram + projection_gram

    assert gram is not None
    gram = gram / num_projections
    grad_norms = jnp.sqrt(jnp.clip(jnp.real(jnp.diag(gram)), a_min=0.0))
    cosine = gram / jnp.clip(grad_norms[:, None] * grad_norms[None, :], eps)
    return NTKResult(gram=gram, cosine=cosine, grad_norms=grad_norms)


def ntk_info_dict(prefix: str, result: NTKResult) -> dict[str, at.Array]:
    """Small summary dict suitable for logging."""

    off_diag_mask = ~jnp.eye(result.gram.shape[0], dtype=jnp.bool_)
    offdiag_mean = jnp.where(
        result.gram.shape[0] > 1,
        jnp.mean(result.cosine[off_diag_mask]),
        jnp.asarray(0.0, dtype=result.cosine.dtype),
    )
    return {
        f"{prefix}/gram_mean": jnp.mean(result.gram),
        f"{prefix}/gram_diag_mean": jnp.mean(jnp.diag(result.gram)),
        f"{prefix}/cosine_mean": jnp.mean(result.cosine),
        f"{prefix}/cosine_offdiag_mean": offdiag_mean,
        f"{prefix}/grad_norm_mean": jnp.mean(result.grad_norms),
    }
