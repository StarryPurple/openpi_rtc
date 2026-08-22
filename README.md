# openpi_rtc — JAX Real-Time Chunking for openpi pi0/pi0.5

Real-time chunking (RTC) for the **JAX** `openpi.models.pi0.Pi0` policy used
by the pi0.5 XTrainer checkpoints. Three methods, one codebase, all entry
points at this repository root:

| Method | Train-free? | Entry |
| --- | --- | --- |
| **inference-RTC** | yes | `eval_offline_rtc.py --mode rtc` / `run_robot.py --mode rtc` |
| **train-RTC** | no (fine-tune from 49999) | `rtc_train.py` |
| **πR² v1** | no (fine-tune from 49999) | `pir2_train.py` |

## Standalone layout (no parent repository)

This repo used to live inside the `openpi05` parent repository and imported
`openpi.*` / `scripts.train` from it. It is now self-contained: the required
openpi code is vendored in, so the repo can be moved, bundled, and `uv sync`ed
on any machine without a parent checkout.

| Path | Origin |
| --- | --- |
| `openpi/` | vendored from `openpi05/src/openpi` |
| `packages/openpi-client/` | vendored workspace member (needed by `openpi.transforms` / `openpi.policies`) |
| `scripts/train.py` | vendored training entry (used by `rtc_train.py` / `pir2_train.py`) |
| `scripts/compute_norm_stats.py` | vendored norm-stats computation |
| `scripts/benchmark_pi05_inference.py` | vendored inference benchmark |
| `examples/xtrainer_real/convert_xtrainer_data_to_lerobot.py` | vendored raw-HDF5 → LeRobot converter |
| `references/` | vendored upstream RTC references (lerobot / Kinetix) |
| `pyproject.toml`, `uv.lock`, `.python-version` | environment definition (uv sync at repo root) |

The **inference bundle** (`package_for_inference.sh`) additionally embeds the
native XTrainer stack (`dobot_control/`, `ModelTrain/`, `experiments/`,
`scripts/`, `third_party/`, `ckpt/`) inside `rtc_control/`, so the robot PC
is fully self-contained: the openpi RTC stack, the openpi baseline, and the
native `run_inference.py` baseline all run from the extracted
`rtc_control/` without relying on anything pre-installed. The control repo's
`README.md` / `LICENSE` are kept as `README-xtrainer.md` / `LICENSE-xtrainer`
so they do not clobber this repo's. Two Python environments coexist on the
robot PC: the native ACT stack runs under its Python 3.8.10 (the shipped
`ModelTrain/module/*.pyc` are 3.8 bytecode), and the openpi stack runs in
the uv-managed Python 3.11 venv.

## Data and checkpoints

Reference checkpoint (baseline + fine-tune init; trained on the yulong data):

```bash
export OPENPI05_CHECKPOINT_49999=/inspire/qb-ilm/project/robot-reasoning/xuyue-p-xuyue/ziyu/checkpoints/g100_pi/pi05-task_00031_yulong-xtrainer/default_pi05/49999
```

Fine-tune data (new HDF5, conventions unchanged):

```bash
export OPENPI05_RAW_TRAIN_DIR=/inspire/hdd/project/robot-reasoning/public/RHOS/dobot/task_00031_entong/train
```

`OPENPI05_CHECKPOINT_49999` must point at a directory containing `params/` and
`assets/`; `OPENPI05_RAW_TRAIN_DIR` must contain the `.hdf5` files directly.
All paths passed to `--checkpoint` / `--dataset` / `--raw-dir` are validated
by `paths.py` (absolute path + existence + expected layout) before any model
load / dataset conversion starts.

Two task configs are registered in the vendored `openpi/training/config.py`:

- `pi05-task_00031_yulong-xtrainer` — the reference 49999 checkpoint and
  baseline / inference-RTC comparisons;
- `pi05-task_00031_entong-xtrainer` — fine-tuning and evaluation of the
  train-RTC / πR² checkpoints (LeRobot repo id `task_00031_entong_train`,
  norm stats under `assets/pi0-task_00031_entong-xtrainer/`).

The eval / latency / probe / robot entries default to the entong config, so
pass `--config pi05-task_00031_yulong-xtrainer` explicitly when evaluating
the 49999 baseline.

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
uv run python eval_offline_rtc.py --mode baseline \
  --config pi05-task_00031_yulong-xtrainer \
  --checkpoint "$OPENPI05_CHECKPOINT_49999" \
  --dataset "$OPENPI05_RAW_TRAIN_DIR" \
  --prompt "Transfer the test tube from the right rack to the left rack."

# inference-RTC (stride must equal d)
uv run python eval_offline_rtc.py --mode rtc \
  --config pi05-task_00031_yulong-xtrainer \
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
uv run python probe_checkpoint.py \
  --config pi05-task_00031_yulong-xtrainer \
  --checkpoint "$OPENPI05_CHECKPOINT_49999" \
  --dataset "$OPENPI05_RAW_TRAIN_DIR"

# steady-state latency + d recommendation on the machine that will drive
# the robot (RTX 4090 on the robot PC, ~150-300 ms for pi0.5 10 steps)
uv run python measure_latency.py \
  --config pi05-task_00031_yulong-xtrainer \
  --checkpoint "$OPENPI05_CHECKPOINT_49999" \
  --hdf5 "$OPENPI05_RAW_TRAIN_DIR"
```

Control period is 25 Hz (40 ms); the deployment delay is
`d = ceil(latency_ms / 40)` (+1 safety tick), measured on the robot PC, not
inferred from the data. Default budget: **d = 7** (280 ms).

## Real robot (robot PC)

The robot PC loads the trained checkpoint itself and drives the arms locally
(RTX 4090 / 24 GB); it does not need the training data. The inference bundle
extracts as a self-contained `rtc_control/` (unpacked anywhere): it carries
the openpi RTC stack, the checkpoint, and the native XTrainer stack, so
`run_inference.py` works with its original `./ckpt/...` paths with nothing
pre-installed on the machine. The adapter `robot_xtrainer.XtrainerRobot`
implements the observation/action API for the Dobot Nova dual-arm platform:

- **left-first** action order: [left 6 joints, left gripper, right 6 joints,
  right gripper] (14,), joints in radians, gripper 1 = open / 0 = closed;
- arms on 192.168.5.1 (left) / 192.168.5.2 (right); three RealSense cameras,
  top crop [150:420, 220:480] → 640×480, BGR.

```bash
# 工控机 python 是 3.8.10 且没有 uv：openpi 栈需要 Python >=3.11，先装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd <rtc_control> && uv sync
uv run python run_robot.py --mode rtc \
  --config pi05-task_00031_entong-xtrainer \
  --checkpoint . --robot robot_xtrainer:XtrainerRobot \
  --episodes 10 [--robot-type "Nova 2"|"Nova 5"]
```

`safety.py` runs before every action and is **on by default**: finite values,
J3 safe zones, per-step servo delta (0.9 rad), and FK working-zone / TCP
Z-speed protection. `--robot-type` must match the real unit (Nova 2 / Nova 5).
`run_robot.py` supports `--mode baseline` and `--mode rtc` on the same
checkpoint; the async worker estimates the in-flight delay from measured
latency. `dobot_control` resolves automatically (it lives inside
`rtc_control/`; the parent directory is appended as a fallback for older
layouts), so no PYTHONPATH tweak is needed.

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
uv run python rtc_train.py --exp-name rtc_train_d6 \
  --simulated-delay 6 --num-train-steps 10000 --fsdp-devices 2 --dry-run

# real run (GPU machine, 2xH200; ~1h for 10k steps, wandb off)
uv run python rtc_train.py --exp-name rtc_train_d6 \
  --simulated-delay 6 --num-train-steps 10000 --fsdp-devices 2
```

Defaults: config `pi05-task_00031_entong-xtrainer`, LeRobot repo id
`task_00031_entong_train`, starting checkpoint `$OPENPI05_CHECKPOINT_49999`
(yulong 49999). `--simulated-delay` is the training delay budget; deploy with
`inference_delay <= simulated_delay - 1`. The raw HDF5 is converted to
LeRobot and norm stats computed automatically on a fresh machine
(`--raw-dir` / `OPENPI05_RAW_TRAIN_DIR`). The new checkpoint lands under
`checkpoints/qb-ilm-ckpts/g100_pi/pi05-task_00031_entong-xtrainer/<exp_name>/`.

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
uv run python pir2_train.py --exp-name pir2_v1 \
  --max-delay 8 --num-train-steps 10000 --fsdp-devices 2 --dry-run
uv run python pir2_train.py --exp-name pir2_v1 \
  --max-delay 8 --num-train-steps 10000 --fsdp-devices 2
```

Evaluate with `pir2_eval.py` (default `--inference-delay 7`,
`--num-steps 10`; try `--num-steps 1` for the fast single-step mode).

## Transfer between machines (BOS for training, OBS for the robot PC)

The training machine and the robot PC are not on the same network. Two
object-storage channels are used:

- **Training GPU machine ← Baidu BOS** (`bos_transfer.py`, bucket
  `handzero-research`, endpoint `bj.bcebos.com`, SDK `bce-python-sdk`).
- **Industrial PC ← Huawei Cloud OBS** (`obs_transfer.py`, bucket
  `handzero-research`, endpoint `https://obs.cn-north-4.myhuaweicloud.com`
  (Beijing), SDK `esdk-obs-python`; obsutil CLI works as an alternative with
  the same object keys).

Bundle contents are defined by `package_for_training.sh` (standalone repo +
starting checkpoint + optional raw data) and `package_for_inference.sh`
(standalone repo + params + assets only, no training data).

### One-command flows

```bash
# dev machine -> BOS (training machine channel); add --with-data for the raw HDF5
export OPENPI05_CHECKPOINT_49999=...
export OPENPI05_RAW_TRAIN_DIR=.../task_00031_entong/train
export BOS_AK=... BOS_SK=...        # or fill bos_transfer.py top
bash transfer_training_bundle.sh [--with-data]

# dev machine -> OBS (industrial PC channel)
export OBS_BUCKET=handzero-research OBS_ENDPOINT=https://obs.cn-north-4.myhuaweicloud.com
bash transfer_inference_bundle.sh [checkpoint_dir]
```

Object keys (default prefix `openpi05/`):

```text
openpi05/training_bundle.tar.gz        # BOS: code + 49999 checkpoint
openpi05/raw_data.tar.gz               # BOS (optional): entong raw HDF5
openpi05/inference_bundle.tar.gz       # OBS: code + trained checkpoint params/assets
```

### Manual transfers (same object keys)

```bash
# training machine (pull from BOS)
uv pip install -r requirements-bos.txt
python3 bos_transfer.py download openpi05/training_bundle.tar.gz /tmp/training_bundle.tar.gz
python3 bos_transfer.py download openpi05/raw_data.tar.gz /tmp/raw_data.tar.gz

# dev machine (push to BOS)
uv run --with bce-python-sdk==0.9.76 python bos_transfer.py upload \
  /tmp/openpi_training_bundle.tar.gz openpi05/training_bundle.tar.gz
uv run --with bce-python-sdk==0.9.76 python bos_transfer.py upload \
  /tmp/openpi_training_bundle.data.tar.gz openpi05/raw_data.tar.gz

# industrial PC (pull from OBS) -- obsutil (single binary, no Python deps)
obsutil config -i <AK> -k <SK> -e https://obs.cn-north-4.myhuaweicloud.com
obsutil cp obs://handzero-research/openpi05/inference_bundle.tar.gz /tmp/ -f

# or with the Python SDK
pip install -r requirements-obs.txt
python3 obs_transfer.py download openpi05/inference_bundle.tar.gz /tmp/inference_bundle.tar.gz \
  --bucket handzero-research --endpoint https://obs.cn-north-4.myhuaweicloud.com

# extract anywhere -> rtc_control/ (fully self-contained)
tar -xzf /tmp/inference_bundle.tar.gz -C /path/to/extract
cd /path/to/extract/rtc_control && uv sync
```

Both transfer scripts support `ls` / `upload` / `download` / `rm`, verify
size after transfer, and resume large files (BOS: multipart manifest;
OBS: SDK checkpoint files). Credentials are read from the environment
(`BOS_AK`/`BOS_SK`, `OBS_AK`/`OBS_SK`) or the placeholders at the top of
each script and are never printed.

The 工控机 obsutil 下载/解压/清理的模式参考数据目录旁的
`download.sh`（`obsutil cp obs://<bucket>/<prefix>/ ./ -r -f` 递归下载；
注意该脚本下载的是 RHOS 共享桶 `sai.liyl/ziyu/dobot/task_00031_entong/`，
咱们自己上传的 bundle 使用上面的 `handzero-research` 桶和 `openpi05/` 前缀）。

## Tests (CPU, no checkpoint needed)

```bash
uv run python tests/test_rtc_processor_jax.py     # guidance math + while_loop/vjp
uv run python tests/test_eval_offline_smoke.py    # offline eval data flow
uv run python tests/test_integrate_smoke.py       # dummy pi0.5 end-to-end
uv run python tests/test_rtc_train_jax.py         # train-RTC loss + hard clamp
uv run python tests/test_pir2_jax.py              # πR² staircase + fast channel
uv run python tests/test_safety.py                # robot safety checks
```

## Current assumptions

- The repo is standalone; `uv sync` at the repo root installs the vendored
  `openpi` package and the `openpi-client` workspace member.
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
- The vendored `openpi/training/config.py` still contains a few hardcoded
  pretrain-checkpoint paths for unrelated legacy configs; the yulong / entong
  XTrainer flows are machine-independent.
