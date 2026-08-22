"""End-to-end smoke test of the JAX RTC integration plumbing.

Uses the *real* ``openpi.policies.policy.Policy`` and the *real* patched
``rtc_sample_actions`` (``jax.lax.while_loop`` + guidance ``jax.vjp`` inside),
but with a tiny toy ``nnx`` model that mimics ``Pi0``'s API surface
(``embed_prefix`` / ``PaliGemma.llm`` / ``embed_suffix`` /
``action_out_proj``). This keeps the test CPU/CI-friendly while exercising
the exact integration code: model patching, ``module_jit`` re-jit, RTC kwarg
forwarding through ``Policy.infer``, raw-chunk capture and anchor fallback.

The full-size ``Pi0`` (with the real SigLIP tower) cannot fit in this
container's 4GB cgroup; that path is validated by ``probe_checkpoint.py`` on
the GPU machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.policies.policy import Policy
from openpi_rtc import wrap_policy_for_rtc
from openpi_rtc.rtc_config import RTCConfig


class MiniLLM(nnx.Module):
    """Toy gemma stand-in: prefix pass stores a KV, suffix pass projects."""

    def __init__(self, rngs):
        self.kv_proj = nnx.Linear(8, 8, rngs=rngs)
        self.suffix_proj = nnx.Linear(8, 8, rngs=rngs)

    def __call__(self, inputs, *, mask, positions, kv_cache=None, adarms_cond=None):
        if inputs[0] is not None:
            kv = self.kv_proj(inputs[0])
            return (None, kv), kv
        out = self.suffix_proj(inputs[1])
        return (None, out), kv_cache


class MiniImg(nnx.Module):
    def __init__(self, rngs):
        self.proj = nnx.Linear(3, 8, rngs=rngs)

    def __call__(self, images, train=False):
        b = images.shape[0]
        return jnp.zeros((b, 4, 8), dtype=jnp.float32), None


class MiniPi0(nnx.Module):
    """Toy model exposing the same API surface used by rtc_sample_actions."""

    def __init__(self, rngs):
        self.action_horizon = 16
        self.action_dim = 4
        self.PaliGemma = nnx.Dict(llm=MiniLLM(rngs), img=MiniImg(rngs))
        self.action_in_proj = nnx.Linear(self.action_dim, 8, rngs=rngs)
        self.action_out_proj = nnx.Linear(8, self.action_dim, rngs=rngs)

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        b = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (b, self.action_horizon, self.action_dim))
        dt = -1.0 / num_steps

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, b)
            )
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=suffix_mask, positions=None,
                adarms_cond=[None, cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def embed_prefix(self, observation):
        b = observation.state.shape[0]
        tokens = jnp.zeros((b, 4, 8), jnp.float32)
        mask = jnp.ones((b, 4), jnp.bool_)
        ar = jnp.zeros(4, jnp.bool_)
        return tokens, mask, ar

    def embed_suffix(self, observation, x_t, timestep):
        b, h, a = x_t.shape
        toks = self.action_in_proj(x_t)
        mask = jnp.ones((b, h), jnp.bool_)
        ar = jnp.concatenate([jnp.ones(1, jnp.bool_), jnp.zeros(h - 1, jnp.bool_)])
        cond = jnp.zeros((b, 8), jnp.float32)
        return toks, mask, ar, cond


def make_obs():
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    return {
        "image": {k: np.zeros((3, 224, 224), dtype=np.uint8) for k in keys},
        "image_mask": {k: True for k in keys},
        "state": np.zeros((32,), dtype=np.float32),
    }


def main():
    model = MiniPi0(nnx.Rngs(0))
    policy = Policy(
        model,
        transforms=[],
        output_transforms=[],
        sample_kwargs={},
        is_pytorch=False,
        pytorch_device="cpu",
    )

    obs = make_obs()
    out = policy.infer(obs)
    act = np.asarray(out["actions"])
    print(f"baseline infer OK: actions shape={act.shape}")
    assert act.shape == (model.action_horizon, model.action_dim)

    rtc = wrap_policy_for_rtc(
        policy,
        RTCConfig(enabled=True, execution_horizon=4,
                  max_guidance_weight=5.0, prefix_attention_schedule="exp"),
        norm_stats=None,
    )
    out2 = rtc.infer(obs)
    print(f"rtc infer (no prev) OK: actions shape={np.asarray(out2['actions']).shape}")
    prev = rtc.last_raw_chunk
    assert prev is not None and prev.shape == act.shape, "raw chunk capture failed"

    out3 = rtc.infer(obs, prev_chunk_left_over=prev[4:],
                     inference_delay=4, execution_horizon=4)
    act3 = np.asarray(out3["actions"])
    print(f"rtc infer (guidance) OK: actions shape={act3.shape}")
    assert np.all(np.isfinite(act3))

    # prepare_prev_chunk without norm_stats must be a no-op fallback.
    prepared = rtc.prepare_prev_chunk(
        prev, np.zeros(32, np.float32), np.zeros(32, np.float32)
    )
    assert np.allclose(prepared, prev)

    # A second guidance call must work too (stateful wrapper path).
    prev2 = rtc.last_raw_chunk
    out4 = rtc.infer(obs, prev_chunk_left_over=prev2[4:],
                     inference_delay=4, execution_horizon=4)
    assert np.all(np.isfinite(np.asarray(out4["actions"])))
    print("integration smoke test passed")


if __name__ == "__main__":
    main()
