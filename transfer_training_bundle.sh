#!/usr/bin/env bash
# Package the training bundle (code + 49999 checkpoint, optionally + raw
# data) and upload it to Baidu BOS -- the training GPU machine channel.
#
# Depends on package_for_training.sh and bos_transfer.py (bce-python-sdk).
#
# Env:
#   OPENPI05_CHECKPOINT_49999   starting checkpoint (yulong 49999)
#   OPENPI05_RAW_TRAIN_DIR      raw HDF5 dir (only with --with-data)
#   BOS_AK / BOS_SK             or fill bos_transfer.py top
#   BOS_BUCKET                  optional (default handzero-research)
#   BOS_ENDPOINT                optional (default bj.bcebos.com)
#
# Usage:
#   bash transfer_training_bundle.sh [--with-data]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

WITH_DATA=""
if [[ "${1:-}" == "--with-data" ]]; then
  WITH_DATA="--with-data"
fi

OUT_PREFIX="/tmp/openpi_training_bundle"
CKPT="${OPENPI05_CHECKPOINT_49999:-}"
RAW="${OPENPI05_RAW_TRAIN_DIR:-}"
BUCKET="${BOS_BUCKET:-handzero-research}"
ENDPOINT="${BOS_ENDPOINT:-bj.bcebos.com}"

if [ -z "$CKPT" ]; then
  echo "ERROR: 需要 OPENPI05_CHECKPOINT_49999（或传 checkpoint 给 package_for_training.sh）" >&2
  exit 1
fi
if [ -n "$WITH_DATA" ] && [ -z "$RAW" ]; then
  echo "ERROR: --with-data 需要 OPENPI05_RAW_TRAIN_DIR" >&2
  exit 1
fi

PKG_ARGS=("$OUT_PREFIX" "$CKPT")
if [ -n "$WITH_DATA" ]; then
  PKG_ARGS+=("$WITH_DATA" "$RAW")
fi
bash package_for_training.sh "${PKG_ARGS[@]}"

echo "===== BOS 上传 (bucket=$BUCKET endpoint=$ENDPOINT) ====="
uv run --with bce-python-sdk==0.9.76 python bos_transfer.py upload \
  "$OUT_PREFIX.tar.gz" openpi05/training_bundle.tar.gz \
  --bucket "$BUCKET" --endpoint "$ENDPOINT"
if [ -n "$WITH_DATA" ]; then
  uv run --with bce-python-sdk==0.9.76 python bos_transfer.py upload \
    "$OUT_PREFIX.data.tar.gz" openpi05/raw_data.tar.gz \
    --bucket "$BUCKET" --endpoint "$ENDPOINT"
fi

echo
echo "训练 GPU 机器上执行:"
echo "  # 一次性安装传输依赖: uv pip install -r requirements-bos.txt"
echo "  python3 bos_transfer.py download openpi05/training_bundle.tar.gz /tmp/training_bundle.tar.gz \\"
echo "    --bucket $BUCKET --endpoint $ENDPOINT"
if [ -n "$WITH_DATA" ]; then
  echo "  python3 bos_transfer.py download openpi05/raw_data.tar.gz /tmp/raw_data.tar.gz \\"
  echo "    --bucket $BUCKET --endpoint $ENDPOINT"
fi
echo "  mkdir -p openpi && tar -xzf /tmp/training_bundle.tar.gz -C openpi"
if [ -n "$WITH_DATA" ]; then
  echo "  tar -xzf /tmp/raw_data.tar.gz -C openpi"
fi
echo "  cd openpi && uv sync"
