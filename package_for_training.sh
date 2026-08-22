#!/usr/bin/env bash
# Package everything the training machine needs (standalone repo, no parent).
#
# The repository root is self-contained: it bundles the vendored `openpi`
# package, the `openpi-client` workspace member, the training scripts, the
# HDF5->LeRobot converter, pyproject.toml + uv.lock, and the openpi_rtc code.
#
# Bundle 1 (code + starting checkpoint):
#   the whole repo (minus .git / checkpoints / eval_logs / venv) plus the
#   starting checkpoint's params + assets lifted to the bundle root.
#
# Bundle 2 (optional, raw data):
#   the raw XTrainer HDF5 (task_00031_entong), transformed into
#   datasets/task_00031_entong/train/. The converted LeRobot dataset is
#   large -- do NOT transfer it; rtc_train.py / pir2_train.py convert and
#   compute norm stats automatically on the target.
#
# Usage (from this repository root):
#   bash package_for_training.sh [/tmp/openpi_training_bundle] [checkpoint]
#   bash package_for_training.sh /tmp/tb \
#       ${OPENPI05_CHECKPOINT_49999:-<absolute 49999 dir>} \
#       --with-data ${OPENPI05_RAW_TRAIN_DIR:-<raw_hdf5_dir>}
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

OUT_PREFIX="${1:-/tmp/openpi_training_bundle}"
CKPT="${2:-${OPENPI05_CHECKPOINT_49999:-}}"
RAW_DIR=""
if [[ "${3:-}" == "--with-data" ]]; then
  RAW_DIR="${4:-}"
elif [ $# -ge 4 ]; then
  echo "ERROR: use --with-data <raw_hdf5_dir>" >&2
  exit 1
fi

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
if [ -n "$RAW_DIR" ]; then
  case "$RAW_DIR" in
    /*) ;;
    *) echo "ERROR: 数据目录必须是绝对路径: $RAW_DIR" >&2; exit 1 ;;
  esac
  if [ ! -d "$RAW_DIR" ]; then
    echo "ERROR: raw data dir not found: $RAW_DIR" >&2
    exit 1
  fi
  if [ -z "$(ls "$RAW_DIR"/*.hdf5 2>/dev/null)" ]; then
    echo "ERROR: raw data dir 下没有 .hdf5 文件: $RAW_DIR" >&2
    exit 1
  fi
fi

mkdir -p "$(dirname "$OUT_PREFIX")"

tar -czf "$OUT_PREFIX.tar.gz" \
  --exclude='./.git' \
  --exclude='./.codex' \
  --exclude='./.agents' \
  --exclude='./.venv' \
  --exclude='./wandb' \
  --exclude='./checkpoints' \
  --exclude='./eval_logs' \
  --exclude='./reference' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  -C "$REPO_ROOT" . \
  -C "$CKPT" \
  params \
  assets

echo "Bundle 1: $OUT_PREFIX.tar.gz ($(du -h "$OUT_PREFIX.tar.gz" | cut -f1))"

if [ -n "$RAW_DIR" ]; then
  tar -czf "$OUT_PREFIX.data.tar.gz" \
    --transform "s,^,datasets/task_00031_entong/train/," \
    -C "$RAW_DIR" .
  echo "Bundle 2 (data): $OUT_PREFIX.data.tar.gz ($(du -h "$OUT_PREFIX.data.tar.gz" | cut -f1))"
fi

echo
echo "On the training machine:"
echo "  mkdir -p openpi && tar -xzf $(basename "$OUT_PREFIX.tar.gz") -C openpi"
if [ -n "$RAW_DIR" ]; then
  echo "  tar -xzf $(basename "$OUT_PREFIX.data.tar.gz") -C openpi"
fi
echo "  cd openpi && uv sync"
echo "  # one-command training (auto-converts data + norm stats if missing):"
echo "  uv run python rtc_train.py --checkpoint \$PWD --raw-dir datasets/task_00031_entong/train"
