"""CPU tests for the train-RTC (simulated delay) implementation.

Covers: per-position time suffix embedding (equivalence with the scalar-time
pi05 path), delay sampling bounds, masked loss accounting, and the hard-clamp
inference sampler (prefix exactly preserved, no-prefix == original).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.policies.policy import Policy
from openpi.models import model as _model
from openpi_rtc import rtc_train
from openpi_rtc.rtc_train import (
    rtc_compute_loss,
    rtc_embed_suffix,
    sample_delay,
    train_rtc_sample_actions,
    wrap_policy_for_train_rtc,
)


def toy_posemb(pos, embedding_dim, min_period, max_period):
    """Scalar/batch-time positional embedding, same formula as pi0."""
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij", pos, 2 * jnp.pi / period, precision=jax.lax.Precision.HIGHEST
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class ToyLLM(nnx.Module):
    """Toy gemma stand-in: prefix pass stores a KV, suffix pass projects."""

    def __init__(self, rngs):
        self.kv_proj = nnx.Linear(8, 8, rngs=rngs)
        self.suffix_proj = nnx.Linear(8, 8, rngs=rngs)
        self.pool_proj = nnx.Linear(8, 8, rngs=rngs)  # cross-token mixing

    def __call__(self, inputs, *, mask, positions, kv_cache=None, adarms_cond=None):
        del mask, positions, adarms_cond
        if inputs[0] is not None and inputs[1] is not None:
            # One-pass training forward (prefix + suffix together).
            x = jnp.concatenate([inputs[0], inputs[1]], axis=1)
            # Per-token projection + a mean-pool broadcast so every output
            # token depends on every input token (mimics attention mixing).
            out = self.suffix_proj(x) + self.pool_proj(jnp.mean(x, axis=1, keepdims=True))
            return (
                out[:, : inputs[0].shape[1]],
                out[:, inputs[0].shape[1] :],
            ), None
        if inputs[0] is not None:
            kv = self.kv_proj(inputs[0])
            return (None, kv), kv
        out = self.suffix_proj(inputs[1])
        return (None, out), kv_cache


class ToyPi0(nnx.Module):
    """Mini pi0.5: pi05-style scalar-time path plus the RTC-train API surface."""

    def __init__(self, rngs):
        self.pi05 = True
        self.action_horizon = 16
        self.action_dim = 4
        self.PaliGemma = nnx.Dict(llm=ToyLLM(rngs))
        self.action_in_proj = nnx.Linear(self.action_dim, 8, rngs=rngs)
        self.action_out_proj = nnx.Linear(8, self.action_dim, rngs=rngs)
        self.time_mlp_in = nnx.Linear(8, 8, rngs=rngs)
        self.time_mlp_out = nnx.Linear(8, 8, rngs=rngs)

    def embed_prefix(self, observation):
        b = observation.state.shape[0]
        tokens = jnp.zeros((b, 4, 8), jnp.float32)
        mask = jnp.ones((b, 4), jnp.bool_)
        ar = jnp.zeros(4, jnp.bool_)
        return tokens, mask, ar

    def embed_suffix(self, observation, x_t, timestep):
        """Scalar-time pi05 path, independent of ``rtc_embed_suffix``."""
        action_tokens = self.action_in_proj(x_t)
        time_emb = toy_posemb(timestep, self.action_in_proj.out_features, 4e-3, 4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        input_mask = jnp.ones(action_tokens.shape[:2], jnp.bool_)
        ar_mask = jnp.concatenate(
            [
                jnp.ones(1, jnp.bool_),
                jnp.zeros(self.action_horizon - 1, jnp.bool_),
            ]
        )
        return action_tokens, input_mask, ar_mask, time_emb

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        """Original-style sampler (scalar time) for equivalence checks."""
        b = observation.state.shape[0]
        dt = -1.0 / num_steps
        if noise is None:
            noise = jax.random.normal(
                rng, (b, self.action_horizon, self.action_dim)
            )
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        _, kv = self.PaliGemma.llm(
            [prefix_tokens, None], mask=None, positions=None
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, b)
            )
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=None,
                positions=None,
                kv_cache=kv,
                adarms_cond=[None, cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0


def make_obs():
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    return {
        "image": {k: np.zeros((3, 224, 224), dtype=np.uint8) for k in keys},
        "image_mask": {k: True for k in keys},
        "state": np.zeros((32,), dtype=np.float32),
    }


def make_obs_obj():
    """Mimic ``Policy.infer``: batch the dict, then build the Observation."""
    return _model.Observation.from_dict(
        jax.tree.map(lambda x: np.asarray(x)[np.newaxis, ...], make_obs())
    )


def test_embed_suffix_equivalence():
    model = ToyPi0(nnx.Rngs(0))
    obs = make_obs_obj()
    x_t = jax.random.normal(jax.random.key(1), (1, model.action_horizon, model.action_dim))
    time = jax.random.uniform(jax.random.key(2), (1,))
    time_pos = jnp.broadcast_to(time[:, None], (1, model.action_horizon))

    toks_rtc, mask_rtc, ar_rtc, cond_rtc = rtc_embed_suffix(model, obs, x_t, time_pos)
    toks_ref, mask_ref, ar_ref, cond_ref = model.embed_suffix(obs, x_t, time)

    assert jnp.allclose(toks_rtc, toks_ref), "action token path drifted"
    assert jnp.array_equal(mask_rtc, mask_ref)
    assert jnp.array_equal(ar_rtc, ar_ref)
    assert cond_rtc.shape == (1, model.action_horizon, 8)
    assert jnp.allclose(cond_rtc, jnp.repeat(cond_ref[:, None, :], model.action_horizon, axis=1), atol=1e-6)
    print("embed_suffix equivalence OK")


def test_sample_delay_bounds():
    d = 6
    rng = jax.random.key(3)
    delays = sample_delay(rng, d, (1024,))
    assert delays.shape == (1024,)
    vals = np.asarray(delays)
    assert vals.min() >= 0 and vals.max() < d
    # Kinetix weights w = exp([d-1, ..., 0]): most mass on delay=0, decaying
    # probability of larger prefixes. The empirical mean should be well below
    # the uniform mean (d-1)/2 and delay=0 must dominate.
    assert vals.mean() < (d - 1) / 2
    assert np.bincount(vals, minlength=d)[0] > np.bincount(vals, minlength=d)[-1]
    print(f"sample_delay OK (mean={vals.mean():.2f} of [0,{d}), p(0)="
          f"{np.bincount(vals, minlength=d)[0] / len(vals):.2f})")


def test_loss_runs_and_masking():
    model = ToyPi0(nnx.Rngs(0))
    obs = make_obs_obj()
    actions = jax.random.normal(jax.random.key(4), (1, model.action_horizon, model.action_dim))
    rng = jax.random.key(5)

    loss_rtc = rtc_compute_loss(model, rng, obs, actions, 6, train=False)
    assert loss_rtc.shape == (1,), loss_rtc.shape
    assert np.all(np.isfinite(np.asarray(loss_rtc)))

    loss_plain = rtc_compute_loss(model, jax.random.key(5), obs, actions, None, train=False)
    assert loss_plain.shape == (1,)
    assert np.all(np.isfinite(np.asarray(loss_plain)))

    # Masked loss must be sensitive to the clamped prefix (conditioning flows
    # through the transformer), otherwise the prefix could never be learned.
    grads = jax.grad(lambda a: jnp.sum(rtc_compute_loss(model, rng, obs, a, 6, train=False)))(
        actions
    )
    assert grads.shape == actions.shape
    assert float(jnp.abs(grads).max()) > 1e-8
    print("loss runs + conditioning gradient OK")


def test_inference_hard_clamp_and_equivalence():
    model = ToyPi0(nnx.Rngs(0))
    policy = Policy(
        model,
        transforms=[],
        output_transforms=[],
        sample_kwargs={},
        is_pytorch=False,
        pytorch_device="cpu",
    )
    obs = make_obs()

    # no-prefix: RTC sampler must match the original sampler.
    rng = jax.random.key(6)
    a_orig = model.sample_actions(rng, make_obs_obj())
    a_rtc = train_rtc_sample_actions(model, rng, make_obs_obj())
    assert np.allclose(np.asarray(a_orig), np.asarray(a_rtc), atol=1e-6)

    # wrapped policy: hard clamp preserves the in-flight prefix exactly.
    wrapped = wrap_policy_for_train_rtc(policy, inference_delay=4, norm_stats=None)
    out = wrapped.infer(obs)
    act0 = np.asarray(out["actions"])
    prev = wrapped.last_raw_chunk
    assert prev is not None and prev.shape == act0.shape

    out2 = wrapped.infer(obs, prev_chunk_left_over=prev[4:])
    act2 = np.asarray(out2["actions"])
    assert np.all(np.isfinite(act2))
    assert np.allclose(act2[:4], prev[4:8], atol=1e-6), "hard clamp prefix drifted"
    print("inference hard clamp + equivalence OK")


if __name__ == "__main__":
    test_embed_suffix_equivalence()
    test_sample_delay_bounds()
    test_loss_runs_and_masking()
    test_inference_hard_clamp_and_equivalence()
    print("train-RTC jax tests passed")
