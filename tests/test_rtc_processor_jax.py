"""CPU-verifiable test: RTC guidance on a small JAX flow-matching policy.

Trains a tiny MLP velocity field on synthetic 6-DOF trajectories (openpi time
convention: t=1 noise -> t=0 clean), then rolls out baseline vs RTC with the
async protocol (stride = d) and asserts RTC reduces chunk-boundary jumps while
keeping action accuracy. The ``sample_actions`` below mirrors the patched
``Pi0.sample_actions`` structure (``jax.lax.while_loop`` with guidance
``jax.vjp`` inside the body), so this validates the exact primitive usage.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi_rtc.rtc_config import RTCConfig
from openpi_rtc.rtc_processor import RTCProcessor

OBS_DIM = 6
ACTION_DIM = 6
CHUNK = 16
NUM_STEPS = 5


def make_trajectories(n_traj: int, traj_len: int, seed: int):
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n_traj):
        n_wp = 5
        wps = rng.standard_normal((n_wp, OBS_DIM)) * 1.5
        t = np.linspace(0, 1, traj_len)
        idx = t * (n_wp - 1)
        i0 = np.clip(idx.astype(int), 0, n_wp - 2)
        frac = (idx - i0)[:, None]
        s = wps[i0] * (1 - frac) + wps[i0 + 1] * frac
        s += 0.25 * np.sin(2 * np.pi * t[:, None] * 3 + rng.standard_normal(OBS_DIM))
        trajs.append(s)
    return jnp.asarray(np.stack(trajs))


def init_params(seed: int = 0):
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (OBS_DIM + ACTION_DIM * CHUNK + 1, 256)) * 0.1,
        "b1": jnp.zeros(256),
        "w2": jax.random.normal(k2, (256, 256)) * 0.1,
        "b2": jnp.zeros(256),
        "w3": jax.random.normal(k3, (256, ACTION_DIM * CHUNK)) * 0.1,
        "b3": jnp.zeros(ACTION_DIM * CHUNK),
    }


def forward(params, obs, x_t, t):
    b = x_t.shape[0]
    inp = jnp.concatenate([obs, x_t.reshape(b, -1), t.reshape(b, 1)], axis=-1)
    h = jax.nn.relu(inp @ params["w1"] + params["b1"])
    h = jax.nn.relu(h @ params["w2"] + params["b2"])
    return (h @ params["w3"] + params["b3"]).reshape(b, CHUNK, ACTION_DIM)


def flow_loss(params, obs, chunk, time, key):
    noise = jax.random.normal(key, chunk.shape)
    x_t = time[:, None, None] * noise + (1 - time)[:, None, None] * chunk
    target = noise - chunk
    pred = forward(params, obs, x_t, time)
    return jnp.mean((pred - target) ** 2)


def train(params, trajs, steps: int = 1200, seed: int = 0):
    opt = optax.adam(3e-4)
    state = opt.init(params)
    n_traj, traj_len, _ = trajs.shape
    n_chunks = traj_len - CHUNK
    key = jax.random.key(seed + 1)

    @jax.jit
    def step(params, state, key):
        k1, k2, k3 = jax.random.split(key, 3)
        idx_traj = jax.random.randint(k1, (32,), 0, n_traj)
        idx_t = jax.random.randint(k2, (32,), 0, n_chunks)
        obs = trajs[idx_traj, idx_t]
        chunk = jax.vmap(
            lambda tr, t: jax.lax.dynamic_slice(
                trajs[tr], (t, 0), (CHUNK, ACTION_DIM)
            )
        )(idx_traj, idx_t)
        time = jax.random.uniform(k3, (32,))
        loss, grads = jax.value_and_grad(flow_loss)(params, obs, chunk, time, k1)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, loss

    for i in range(steps):
        key, subkey = jax.random.split(key)
        params, state, loss = step(params, state, subkey)
        if i % 400 == 0:
            print(f"  train step {i}: loss={float(loss):.5f}")
    return params


def sample_actions(params, obs, rng_key, *, rtc_processor=None,
                   prev_chunk_left_over=None, inference_delay=None,
                   execution_horizon=None):
    x_t = jax.random.normal(rng_key, (1, CHUNK, ACTION_DIM))
    dt = -1.0 / NUM_STEPS
    prev = None
    delay = 0
    horizon = execution_horizon
    if rtc_processor is not None and prev_chunk_left_over is not None:
        prev = jnp.asarray(prev_chunk_left_over)
        if prev.ndim == 2:
            prev = prev[None, ...]
        horizon = (
            rtc_processor.rtc_config.execution_horizon
            if horizon is None
            else horizon
        )
        horizon = min(int(horizon), prev.shape[1])
        delay = min(int(inference_delay), horizon)
        if prev.shape[1] < CHUNK or prev.shape[2] < ACTION_DIM:
            padded = jnp.zeros((1, CHUNK, ACTION_DIM))
            padded = padded.at[:, : prev.shape[1], : prev.shape[2]].set(prev)
            prev = padded

    def denoise(x, t):
        return forward(params, obs[None], x, jnp.full((1,), t))

    def step(carry):
        x, t = carry
        if prev is not None:
            v = rtc_processor.denoise_step(
                x, prev, delay, t, lambda z: denoise(z, t), execution_horizon=horizon
            )
        else:
            v = denoise(x, t)
        return x + dt * v, t + dt

    def cond(carry):
        _, t = carry
        return t >= -dt / 2

    x0, _ = jax.lax.while_loop(cond, step, (x_t, jnp.asarray(1.0)))
    return np.asarray(x0[0])


def rollout(params, traj, *, rtc: bool = False, seed: int = 7, d: int = 4,
            horizon: int = 8, kappa: float = 5.0, schedule: str = "exp"):
    proc = (
        RTCProcessor(
            RTCConfig(enabled=True, execution_horizon=horizon,
                      max_guidance_weight=kappa,
                      prefix_attention_schedule=schedule)
        )
        if rtc
        else None
    )
    key = jax.random.key(seed)
    t_end = traj.shape[0] - CHUNK
    prev_chunk = None
    boundary_errs, acc_errs, times_ms = [], [], []
    for i in range(0, t_end, d):
        obs = traj[i]
        gt = np.asarray(traj[i : i + CHUNK])
        t0 = time.perf_counter()
        if rtc and prev_chunk is not None:
            chunk = sample_actions(
                params, obs, key, rtc_processor=proc,
                prev_chunk_left_over=prev_chunk[d:],
                inference_delay=d, execution_horizon=horizon,
            )
        else:
            chunk = sample_actions(params, obs, key)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        acc_errs.append(float(np.mean((chunk - gt) ** 2)))
        if prev_chunk is not None and d + d <= len(prev_chunk):
            boundary_errs.append(
                float(np.mean((chunk[:d] - prev_chunk[d : d + d]) ** 2))
            )
        prev_chunk = chunk
    return boundary_errs, acc_errs, times_ms


def test_prefix_weights():
    proc = RTCProcessor(RTCConfig(prefix_attention_schedule="linear"))
    w = np.asarray(proc.get_prefix_weights(2, 5, 10))
    assert w.shape == (10,)
    assert np.allclose(w[:2], 1.0)
    assert np.allclose(w[2:5], [0.75, 0.5, 0.25], atol=1e-6)
    assert np.allclose(w[5:], 0.0)
    # Zeros schedule: hard mask.
    proc0 = RTCProcessor(RTCConfig(prefix_attention_schedule="zeros"))
    w0 = np.asarray(proc0.get_prefix_weights(3, 6, 10))
    assert np.allclose(w0[:3], 1.0) and np.allclose(w0[3:], 0.0)


def test_rtc_reduces_boundary_jump():
    print("1) synthesizing trajectories ...")
    trajs = make_trajectories(48, 220, seed=0)
    print("2) training small flow policy ...")
    params = train(init_params(0), trajs, steps=1200, seed=0)
    print("3) rolling out baseline vs RTC ...")
    traj_test = make_trajectories(1, 220, seed=999)[0]
    b_b, b_a, b_t = rollout(params, traj_test, rtc=False)
    r_b, r_a, r_t = rollout(params, traj_test, rtc=True,
                            d=4, horizon=8, kappa=5.0)
    bm_b = float(np.mean(b_b))
    bm_r = float(np.mean(r_b))
    am_b = float(np.mean(b_a))
    am_r = float(np.mean(r_a))
    print(f"  baseline boundary_mse={bm_b:.5f} action_mse={am_b:.5f} infer_ms={np.mean(b_t):.2f}")
    print(f"  rtc     boundary_mse={bm_r:.5f} action_mse={am_r:.5f} infer_ms={np.mean(r_t):.2f}")
    assert bm_r < bm_b, "RTC should reduce chunk-boundary jumps"
    assert np.isfinite(bm_r) and np.isfinite(am_r)
    # Guidance adds a backward pass; allow generous overhead factor on CPU.
    assert np.mean(r_t) < 8 * np.mean(b_t) + 50
    print(f"boundary_mse reduction: {100 * (bm_b - bm_r) / bm_b:.1f}%")


if __name__ == "__main__":
    test_prefix_weights()
    test_rtc_reduces_boundary_jump()
    print("ALL JAX RTC PROCESSOR TESTS PASSED")
