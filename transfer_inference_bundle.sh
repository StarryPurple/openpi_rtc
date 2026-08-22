#!/usr/bin/env bash
# Package the inference bundle (standalone repo + checkpoint params/assets,
# no training data) and upload it to Huawei Cloud OBS -- the industrial PC
# channel.
#
# Upload side prefers the obsutil CLI (single binary, configured on the dev
# machine); falls back to obs_transfer.py (esdk-obs-python).
#
# Env:
#   OPENPI05_CHECKPOINT_49999   or pass the checkpoint dir as $1
#   OBS_BUCKET                  optional (default openpi-rtc)
#   OBS_ENDPOINT                optional (default cn-north-4)
#   OBS_AK / OBS_SK             only needed for the obs_transfer.py fallback
#
# Usage:
#   bash transfer_inference_bundle.sh [checkpoint_dir]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

OUT="/tmp/openpi_inference_bundle.tar.gz"
CKPT="${1:-${OPENPI05_CHECKPOINT_49999:-}}"
BUCKET="${OBS_BUCKET:-handzero-research}"
ENDPOINT="${OBS_ENDPOINT:-https://obs.cn-north-4.myhuaweicloud.com}"
KEY="openpi05/inference_bundle.tar.gz"

if [ -z "$CKPT" ]; then
  echo "ERROR: 需要 checkpoint 目录: 传第一个参数或设 OPENPI05_CHECKPOINT_49999" >&2
  exit 1
fi

bash package_for_inference.sh "$OUT" "$CKPT"

echo "===== OBS 上传 (bucket=$BUCKET key=$KEY) ====="
if command -v obsutil >/dev/null 2>&1; then
  echo "使用 obsutil (请确认本机 obsutil 已配置 endpoint: obsutil config -i <AK> -k <SK> -e $ENDPOINT)"
  obsutil cp "$OUT" "obs://$BUCKET/$KEY" -f
else
  echo "使用 obs_transfer.py (esdk-obs-python)"
  uv run --with esdk-obs-python python obs_transfer.py upload \
    "$OUT" "$KEY" --bucket "$BUCKET" --endpoint "$ENDPOINT"
fi

echo
echo "工控机上执行 (二选一):"
echo "  # 方式 A: obsutil（推荐，单二进制，装一次）"
echo "  obsutil config -i <AK> -k <SK> -e $ENDPOINT"
echo "  obsutil cp obs://$BUCKET/$KEY /tmp/ -f"
echo "  # 方式 B: obs_transfer.py (需要 python + esdk-obs-python)"
echo "  pip install -r requirements-obs.txt"
echo "  python3 obs_transfer.py download $KEY /tmp/inference_bundle.tar.gz \\"
echo "    --bucket $BUCKET --endpoint $ENDPOINT"
echo "  # 解压到任意目录 -> rtc_control/ (自包含: openpi 栈 + 控制代码 + ModelTrain + ckpt)"
echo "  tar -xzf /tmp/inference_bundle.tar.gz -C /path/to/extract"
echo "  cd /path/to/extract/rtc_control && uv sync"
