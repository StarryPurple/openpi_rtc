# openpi_rtc — JAX Real-Time Chunking for openpi pi0/pi0.5

Real-time chunking (RTC) for the **JAX** `openpi.models.pi0.Pi0` policy used
by the pi0.5 XTrainer checkpoints (task_00031_yulong tube-transfer). Three
methods, one codebase, all entry points in this directory:

| Method | Train-free? | Entry |
| --- | --- | --- |
| **inference-RTC** | yes | `eval_offline_rtc.py --mode rtc` / `run_robot.py --mode rtc` |
| **train-RTC** | no (fine-tune from 49999) | `rtc_train.py` |
| **πR² v1** | no (fine-tune from 49999) | `pir2_train.py` |

## Contents

| File | Purpose |
| --- | --- |
| `rtc_config.py` | RTC knobs (horizon, kappa schedule, anchor correction). |
| `rtc_processor.py` | Pure-JAX guidance core (`jax.vjp` full-Jacobian correction). |
| `action_queue.py` | Numpy action queue for the async robot loop. |
| `integrate_openpi.py` | Wraps the JAX policy, captures raw chunks, re-anchors deltas. |
| `eval_offline_rtc.py` | Offline HDF5 replay: baseline vs inference-RTC metrics. |
| `eval_offline_train_rtc.py` | Offline replay for train-RTC checkpoints (hard-clamp sampler). |
| `pir2_eval.py` | Offline replay for πR² checkpoints. |
| `sweep_d.py` | Offline sweep over inference delay d (boundary continuity vs d). |
| `measure_latency.py` | End-to-end latency on the target machine + d recommendation. |
| `probe_checkpoint.py` | One-shot checkpoint readiness probe (GPU machine). |
| `run_robot.py` | Async real-robot loop (baseline / inference-RTC). |
| `robot_xtrainer.py` | Dobot XTrainer real-robot adapter (left-first, verified against data). |
| `safety.py` | Safety checks: finite values, J3 zone, servo delta, FK pose (default on). |
| `check_robot_pc.py` | One-shot robot-PC environment check (writes a report). |
| `rtc_train.py` | train-RTC: simulated-delay loss + sampler + training entry. |
| `pir2_train.py` | πR² v1: staircase diffusion-forcing loss + fast proprio channel + training entry. |
| `paths.py` | Absolute-path validation helpers (fail fast before model load). |
| `bos_transfer.py` | BOS upload/download helper for the training / robot bundles. |
| `requirements-bos.txt` | Dependency for `bos_transfer.py` (`bce-python-sdk`). |
| `tests/` | CPU-runnable unit/smoke tests (no checkpoint needed). |

Machine-specific paths are never hardcoded and are never derived from the
current working directory. All data paths must be **absolute** and are checked
for existence *before* any model load / dataset conversion starts. Use these
env vars (or pass the corresponding CLI flags):

```bash
export OPENPI05_CHECKPOINT_49999=/path/to/checkpoints/pi05-task_00031_yulong-xtrainer/49999
export OPENPI05_RAW_TRAIN_DIR=/path/to/80_hdf5_files
```

`OPENPI05_CHECKPOINT_49999` must point at a directory containing `params/` and
`assets/`; `OPENPI05_RAW_TRAIN_DIR` must contain the `.hdf5` files. The
checkpoints passed to `--checkpoint` in the eval / robot / latency entries go
through the same checks (`paths.py`).

## How RTC works here

`Pi0.sample_actions` runs `jax.lax.while_loop` over flow-matching steps
(t: 1 → 0, `x_t += dt * v_t`). The wrapped policy steps each denoising step
with `RTCProcessor.denoise_step`, which:

1. predicts the clean chunk `x1 = x_t - t * v(x_t)`;
2. computes the full-Jacobian correction `J^T ((prev - x1) * weights)` with
   `jax.vjp` (this also recovers `v_t` without an extra forward pass);
3. applies the lerobot guidance weight with a `kappa`-clipped schedule.

The guidance target lives in the model's action space: quantile-normalized
*delta* actions (12 joints relative to the observation state, 2 gripper dims
absolute). `RtcPolicy` captures `last_raw_chunk` (before output transforms),
and re-anchors a previous chunk from `prev_state` to the current observation
state (`anchor_correction`, `RTCConfig`), which is required for
whole-chunk-shift delta policies.

## Offline evaluation (GPU machine, repo root)

```bash
uv run python openpi_rtc/eval_offline_rtc.py --mode baseline \
  --checkpoint "$OPENPI05_CHECKPOINT_49999" \
  --dataset "$OPENPI05_RAW_TRAIN_DIR" \
  --prompt "Transfer the test tube from the right rack to the left rack."

# inference-RTC (stride must equal d)
uv run python openpi_rtc/eval_offline_rtc.py --mode rtc \
  --inference_delay 4 --execution_horizon 10 --max_guidance_weight 10.0 \
  --checkpoint "$OPENPI05_CHECKPOINT_49999" \
  --dataset "$OPENPI05_RAW_TRAIN_DIR" \
  --prompt "Transfer the test tube from the right rack to the left rack."
```

Reported metrics: action MSE/L1 vs GT, boundary MSE/L1 (absolute action
space), mean `infer_ms`. Results are written under `eval_logs/`.

## Checkpoint probe and latency

```bash
# readiness probe (config registration, JAX backend, first-frame infer)
uv run python openpi_rtc/probe_checkpoint.py \
  --checkpoint "$OPENPI05_CHECKPOINT_49999" \
  --dataset "$OPENPI05_RAW_TRAIN_DIR"

# steady-state latency + d recommendation on the machine that will drive
# the robot (RTX 4090 on the robot PC, ~150-300 ms for pi0.5 10 steps)
uv run python openpi_rtc/measure_latency.py \
  --checkpoint "$OPENPI05_CHECKPOINT_49999" \
  --hdf5 "$OPENPI05_RAW_TRAIN_DIR"
```

Control period is 25 Hz (40 ms); the deployment delay is
`d = ceil(latency_ms / 40)` (+1 safety tick), measured on the robot PC, not
inferred from the data. Default budget: **d = 7** (280 ms).

## Real robot (robot PC)

The robot PC loads the trained checkpoint itself and drives the arms locally
(RTX 4090 / 24 GB); it does not need the training data. The adapter
`openpi_rtc.robot_xtrainer.XtrainerRobot` implements the observation/action
API for the Dobot Nova dual-arm platform:

- **left-first** action order: [left 6 joints, left gripper, right 6 joints,
  right gripper] (14,), joints in radians, gripper 1 = open / 0 = closed;
- arms on 192.168.5.1 (left) / 192.168.5.2 (right); three RealSense cameras,
  top crop [150:420, 220:480] → 640×480, BGR.

```bash
uv run python openpi_rtc/run_robot.py --mode rtc \
  --checkpoint <model dir> --robot openpi_rtc.robot_xtrainer:XtrainerRobot \
  --episodes 10 [--robot-type "Nova 2"|"Nova 5"]
```

`safety.py` runs before every action and is **on by default**: finite values,
J3 safe zones, per-step servo delta (0.9 rad), and FK working-zone / TCP
Z-speed protection. `--robot-type` must match the real unit (Nova 2 / Nova 5).
`run_robot.py` supports `--mode baseline` and `--mode rtc` on the same
checkpoint; the async worker estimates the in-flight delay from measured
latency.

## train-RTC (`rtc_train.py`)

Faithful port of Kinetix's train-RTC (arXiv:2506.07339): during training a
per-sample delay is drawn with exponential weights
(`w = exp([d-1, ..., 0])`, most mass on delay=0), the first `delay` positions
are clamped to clean actions (time=0 in pi0 convention), and the flow-matching
loss is masked to the remaining positions. The inference sampler
(`train_rtc_sample_actions`) hard-clamps the first `inference_delay` positions
to the previous chunk's in-flight actions at every denoising step.

```bash
# dry run first (prints the planned invocation)
uv run python openpi_rtc/rtc_train.py --exp-name rtc_train_d6 \
  --simulated-delay 6 --num-train-steps 10000 --fsdp-devices 2 --dry-run

# real run (GPU machine, 2xH200; ~1h for 10k steps, wandb off)
uv run python openpi_rtc/rtc_train.py --exp-name rtc_train_d6 \
  --simulated-delay 6 --num-train-steps 10000 --fsdp-devices 2
```

`--simulated-delay` is the training delay budget; deploy with
`inference_delay <= simulated_delay - 1`. The raw HDF5 is converted to
LeRobot and norm stats computed automatically on a fresh machine
(`--raw-dir` / `OPENPI05_RAW_TRAIN_DIR`). The new checkpoint lands under
`checkpoints/pi05-task_00031_yulong-xtrainer/<exp_name>/`.

## πR² (`pir2_train.py`)

v1 of arXiv:2607.26055 for the pi0.5 stack:

- **staircase per-position noise schedule** (clean front d / ramp / noise tail
  d) as the training-time diffusion-forcing schedule; the deployment delay is
  sampled per sample in [1, max_delay], making the model latency-adaptive
  (any `inference_delay <= max_delay` works at deployment);
- **fast proprioception channel**: a continuous `state_proj` token in the
  suffix at every denoising step (new parameter, fine-tuned from 49999), since
  pi0.5 embeds state as discrete language tokens in the prefix (stale during
  denoising).

The slow channel (async vision/language KV reuse + learned delay embedding)
is deployment wiring and is not implemented in v1; the prefix is recomputed
per inference call.

```bash
uv run python openpi_rtc/pir2_train.py --exp-name pir2_v1 \
  --max-delay 8 --num-train-steps 10000 --fsdp-devices 2 --dry-run
uv run python openpi_rtc/pir2_train.py --exp-name pir2_v1 \
  --max-delay 8 --num-train-steps 10000 --fsdp-devices 2
```

Evaluate with `openpi_rtc/pir2_eval.py` (default `--inference-delay 7`,
`--num-steps 10`; try `--num-steps 1` for the fast single-step mode).

## Transfer between machines (BOS)

The training machine and the robot PC are not on the same network; bundles
move through the `handzero-research` BOS bucket (Beijing). Fill in the AK/SK
at the top of `bos_transfer.py`, then:

```bash
uv pip install -r openpi_rtc/requirements-bos.txt

uv run python openpi_rtc/bos_transfer.py upload /tmp/openpi_training_bundle.tar.gz openpi05/training_bundle.tar.gz
uv run python openpi_rtc/bos_transfer.py upload /tmp/openpi_training_bundle.data.tar.gz openpi05/raw_data.tar.gz
uv run python openpi_rtc/bos_transfer.py upload /tmp/openpi_inference_bundle.tar.gz openpi05/inference_bundle.tar.gz
```

Bundle contents are defined by `package_for_training.sh` (code + starting
checkpoint + optional raw data) and `package_for_inference.sh` (code + params
+ assets only, no training data).

## Tests (CPU, no checkpoint needed)

```bash
uv run python openpi_rtc/tests/test_rtc_processor_jax.py     # guidance math + while_loop/vjp
uv run python openpi_rtc/tests/test_eval_offline_smoke.py    # offline eval data flow
uv run python openpi_rtc/tests/test_integrate_smoke.py       # dummy pi0.5 end-to-end
uv run python openpi_rtc/tests/test_rtc_train_jax.py         # train-RTC loss + hard clamp
uv run python openpi_rtc/tests/test_pir2_jax.py              # πR² staircase + fast channel
uv run python openpi_rtc/tests/test_safety.py                # robot safety checks
```

## Current assumptions

- The 49999 checkpoint is 2-device FSDP-sharded; validate loading on the GPU
  machine with `probe_checkpoint.py` before training.
- Guidance math is verified against the upstream sources vendored in
  `references/`: `modeling_rtc.py` (lerobot) and `kinetix_model.py`
  (Physical Intelligence). Our JAX version keeps Kinetix's full-Jacobian
  `jax.vjp` correction; lerobot's torch port computes only the identity part
  (see `rtc_processor.py` docstring).
- πR²'s slow channel (async VLM + delay embedding) is the remaining extension.
- `safety.py` pose protection needs the correct `--robot-type` (Nova 2 /
  Nova 5) for the working-zone FK to be valid.
