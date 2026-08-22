#!/usr/bin/env bash
# Package the complete robot-PC runtime into a single `rtc_control/` bundle.
#
# The bundle is self-contained: it includes BOTH the openpi RTC stack (code +
# checkpoint) AND the native XTrainer baseline stack (dobot_control,
# ModelTrain, experiments/run_inference.py, scripts, third_party, ckpt), so
# it does not depend on anything pre-installed on the robot PC.
#
#   <xtrainer>/                     (optional; nothing here is required)
#   └── rtc_control/
#       ├── openpi/  packages/  pyproject.toml  uv.lock ...   <- openpi RTC
#       ├── params/  assets/                                 <- checkpoint
#       ├── dobot_control/  experiments/  scripts/            <- native stack
#       ├── ModelTrain/  ckpt/  third_party/  robomimic-r2d2/
#       ├── README.md  LICENSE                               <- ours
#       ├── README-xtrainer.md  LICENSE-xtrainer             <- control repo's
#       └── run_robot.py  rtc_train.py  measure_latency.py ...
#
# Control sources come from reference/xtrainer/ (extracted from
# reference/xtrainer.tar.gz automatically if missing). The control repo's
# README.md/LICENSE are kept under *-xtrainer names so they do not clobber
# ours.
#
# Usage (from this repository root):
#   bash package_for_inference.sh [output.tar.gz] [checkpoint_dir]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

OUT="${1:-/tmp/openpi_inference_bundle.tar.gz}"
CKPT="${2:-${OPENPI05_CHECKPOINT_49999:-}}"
CTRL_SRC="$REPO_ROOT/reference/xtrainer"
CTRL_TAR="$REPO_ROOT/reference/xtrainer.tar.gz"

if [ -z "$CKPT" ]; then
  echo "ERROR: 需要 checkpoint 目录: 传第二个参数或设 OPENPI05_CHECKPOINT_49999" >&2
  exit 1
fi
case "$CKPT" in
  /*) ;;
  *) echo "ERROR: checkpoint 必须是绝对路径: $CKPT" >&2; exit 1 ;;
esac
if [ ! -d "$CKPT/params" ] || [ ! -d "$CKPT/assets" ]; then
  echo "ERROR: checkpoint 缺少 params/ 或 assets/: $CKPT" >&2
  exit 1
fi
if [ ! -f "pyproject.toml" ] || [ ! -f "uv.lock" ]; then
  echo "ERROR: pyproject.toml / uv.lock not found at repo root" >&2
  exit 1
fi

# Ensure the native control stack source is available.
if [ ! -d "$CTRL_SRC/dobot_control" ]; then
  if [ -f "$CTRL_TAR" ]; then
    echo "reference/xtrainer/ 缺失，从 $CTRL_TAR 解压..."
    mkdir -p "$CTRL_SRC"
    tar -xzf "$CTRL_TAR" -C "$CTRL_SRC" 2>/dev/null || true
  fi
fi
for _need in dobot_control ModelTrain experiments scripts third_party; do
  if [ ! -d "$CTRL_SRC/$_need" ]; then
    echo "ERROR: 缺少控制代码目录 reference/xtrainer/$_need（先解压 xtrainer.tar.gz）" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUT")"

REN="$(mktemp -d)"
trap 'rm -rf "$REN"' EXIT
cp "$CTRL_SRC/README.md" "$REN/README-xtrainer.md"
cp "$CTRL_SRC/LICENSE" "$REN/LICENSE-xtrainer"

# Everything (repo + control stack + checkpoint) is wrapped under a single
# top-level `rtc_control/`.
TAR_ARGS=(
  --exclude='./.git'
  --exclude='./.codex'
  --exclude='./.agents'
  --exclude='./.venv'
  --exclude='./wandb'
  --exclude='./checkpoints'
  --exclude='./eval_logs'
  --exclude='./reference'
  --exclude='__pycache__'
  --transform='s,^,rtc_control/,'
  -C "$REPO_ROOT" .
  -C "$CTRL_SRC" dobot_control
  -C "$CTRL_SRC" ModelTrain
  -C "$CTRL_SRC" experiments
  -C "$CTRL_SRC" scripts
  -C "$CTRL_SRC" examples
  -C "$CTRL_SRC" third_party
  -C "$CTRL_SRC" sh
  -C "$CTRL_SRC" robomimic-r2d2
  -C "$CTRL_SRC" requirements.txt
  -C "$CTRL_SRC" version.txt
  -C "$CTRL_SRC" THIRD-PARTY-LICENSES
  -C "$CTRL_SRC" detect.py
  -C "$CTRL_SRC" detect.sh
  -C "$CTRL_SRC" detect.txt
)
TAR_ARGS+=(
  -C "$REN" README-xtrainer.md
  -C "$REN" LICENSE-xtrainer
  -C "$CKPT" params
  -C "$CKPT" assets
)

tar -czf "$OUT" "${TAR_ARGS[@]}"

if [[ "$OUT" == /* ]]; then
  echo "Bundle written to: $OUT"
else
  echo "Bundle written to: $REPO_ROOT/$OUT"
fi
echo "Size: $(du -h "$OUT" | cut -f1)"
echo
echo "On the robot PC (no pre-installed structure needed):"
echo "  tar -xzf $(basename "$OUT") -C <任意目录>"
echo "  cd <任意目录>/rtc_control && uv sync"
echo "  # openpi latency (no robot API):"
echo "  uv run scripts/benchmark_pi05_inference.py --checkpoint params --num-steps 10 --repeats 20"
echo "  uv run python measure_latency.py --config pi05-task_00031_yulong-xtrainer --checkpoint . --hdf5 <one_hdf5_file>"
echo "  # openpi baseline / train-free RTC:"
echo "  uv run python run_robot.py --mode baseline --config pi05-task_00031_yulong-xtrainer --checkpoint . \\"
echo "    --robot robot_xtrainer:XtrainerRobot --robot-type \"Nova 2\" --episodes 1"
echo "  uv run python run_robot.py --mode rtc --config pi05-task_00031_yulong-xtrainer --checkpoint . \\"
echo "    --robot robot_xtrainer:XtrainerRobot --robot-type \"Nova 2\" --episodes 1"
echo "  # native baseline (self-contained; ckpt path unchanged):"
echo "  #   ACT 栈用工控机 Python 3.8.10（不要用 uv 的 3.11 venv 跑 pyc）"
echo "  python3.8 experiments/run_inference.py"
