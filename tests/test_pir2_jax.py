"""CPU tests for the πR² v1 scaffold (staircase schedule, fast channel)."""

from __future__ import annotations

import dataclasses
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
from openpi_rtc.pir2_train import (
    Pir2Config,
    pir2_compute_loss,
    pir2_sample_actions,
    staircase_matrix,
    staircase_time,
    wrap_policy_for_pir2,
)
from test_rtc_train_jax import ToyPi0, make_obs, make_obs_obj


class Pir2ToyPi0(ToyPi0):
    """Toy pi0.5 with the fast-channel ``state_proj``."""

    def __init__(self, rngs):
        super().__init__(rngs)
        # Real pi0.5: state dim == config.action_dim (32). The toy's action
        # dim is 4 for chunk shapes, so the state projector matches the
        # observation state used by make_obs (32).
        self.state_proj = nnx.Linear(32, 8, rngs=rngs)


def test_staircase_schedule():
    H = 12
    t = staircase_time(3, H)
    assert t.shape == (H,)
    assert jnp.all(t[:3] == 0.0)
    assert jnp.all(t[-3:] == 1.0)
    mid = np.asarray(t[3:-3])
    assert np.all(np.diff(mid) > 0), "ramp must be monotonic"

    m = staircase_matrix(4, H)
    assert m.shape == (5, H)
    # d=0 -> no clean front/tail: the whole chunk is a 0->1 ramp (endpoints
    # excluded, so all values lie strictly inside (0, 1) and increase).
    m0 = np.asarray(m[0])
    assert np.all(m0 > 0.0) and np.all(m0 < 1.0)
    assert np.all(np.diff(m0) > 0)
    print("staircase schedule OK")


def test_loss_runs_and_state_channel():
    model = Pir2ToyPi0(nnx.Rngs(0))
    obs = make_obs_obj()
    actions = jax.random.normal(jax.random.key(7), (1, model.action_horizon, model.action_dim))
    rng = jax.random.key(8)

    loss = pir2_compute_loss(model, rng, obs, actions, 4, train=False)
    assert loss.shape == (1,)
    assert np.all(np.isfinite(np.asarray(loss)))

    # The fast proprio channel must actually influence the loss.
    g_state = jax.grad(
        lambda s: jnp.sum(
            pir2_compute_loss(
                model,
                rng,
                make_obs_obj_state(s),
                actions,
                4,
                train=False,
            )
        )
    )(np.asarray(obs.state, dtype=np.float32))
    assert float(jnp.abs(g_state).max()) > 1e-8, "state token has no effect"
    print("πR² loss + fast channel OK")


def make_obs_obj_state(state):
    obs = make_obs_obj()
    return dataclasses.replace(obs, state=jnp.asarray(state, dtype=jnp.float32))


def test_inference_clamp_and_wrapper():
    model = Pir2ToyPi0(nnx.Rngs(0))
    policy = Policy(
        model,
        transforms=[],
        output_transforms=[],
        sample_kwargs={},
        is_pytorch=False,
        pytorch_device="cpu",
    )
    obs = make_obs()

    wrapped = wrap_policy_for_pir2(policy, inference_delay=3, norm_stats=None)
    out = wrapped.infer(obs)
    act0 = np.asarray(out["actions"])
    prev = wrapped.last_raw_chunk
    assert prev is not None and prev.shape == act0.shape

    out2 = wrapped.infer(obs, prev_chunk_left_over=prev[3:])
    act2 = np.asarray(out2["actions"])
    assert np.all(np.isfinite(act2))
    assert np.allclose(act2[:3], prev[3:6], atol=1e-6), "clean front drifted"
    print("πR² inference clamp + wrapper OK")


if __name__ == "__main__":
    test_staircase_schedule()
    test_loss_runs_and_state_channel()
    test_inference_clamp_and_wrapper()
    print("πR² jax tests passed")
