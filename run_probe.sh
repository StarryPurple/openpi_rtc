#!/usr/bin/env bash
# One-shot probe runner for the GPU machine. No robot API needed: it only
# reads the checkpoint + HDF5 and runs JAX inference, then writes a report
# you can send back as-is.
#
# Usage (run on the GPU machine, from anywhere):
#   bash run_probe.sh [checkpoint_dir] [hdf5_dir]
#
# Paths must be absolute; defaults come from env vars (or positional args):
#   checkpoint = ${OPENPI05_CHECKPOINT_49999}
#   dataset    = ${OPENPI05_RAW_TRAIN_DIR}
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CKPT="${1:-${OPENPI05_CHECKPOINT_49999:-}}"
DATA="${2:-${OPENPI05_RAW_TRAIN_DIR:-}}"
REPORT="probe_report_$(date +%Y%m%d_%H%M%S).txt"

if [ -z "$CKPT" ]; then
  echo "ERROR: 需要 checkpoint 目录: 传第一个参数或设 OPENPI05_CHECKPOINT_49999" >&2
  exit 1
fi
if [ -z "$DATA" ]; then
  echo "ERROR: 需要数据集目录: 传第二个参数或设 OPENPI05_RAW_TRAIN_DIR" >&2
  exit 1
fi
case "$CKPT" in
  /*) ;;
  *) echo "ERROR: checkpoint 必须是绝对路径: $CKPT" >&2; exit 1 ;;
esac
case "$DATA" in
  /*) ;;
  *) echo "ERROR: 数据集目录必须是绝对路径: $DATA" >&2; exit 1 ;;
esac
if [ ! -d "$CKPT" ]; then
  echo "ERROR: checkpoint dir not found: $CKPT" >&2
  exit 1
fi
if [ ! -d "$CKPT/params" ] || [ ! -d "$CKPT/assets" ]; then
  echo "ERROR: checkpoint 缺少 params/ 或 assets/: $CKPT" >&2
  exit 1
fi
if [ ! -d "$DATA" ]; then
  echo "ERROR: dataset dir not found: $DATA" >&2
  exit 1
fi

if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  PY="uv run python"
else
  PY="python"
fi

{
  echo "=== machine ==="
  echo "host: $(hostname)  date: $(date -u)"
  nvidia-smi || echo "(nvidia-smi unavailable)"

  echo "=== environment ==="
  $PY -c "import sys, jax; print('python', sys.version.split()[0]); print('jax', jax.__version__); print('devices', jax.devices())" \
    || echo "ERROR: jax import failed (wrong venv? missing deps?)"

  echo "=== probe ==="
  $PY probe_checkpoint.py --checkpoint "$CKPT" --dataset "$DATA"
} 2>&1 | tee "$REPORT"

echo
echo "Report saved to: $REPO_ROOT/$REPORT"
