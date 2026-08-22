#!/usr/bin/env bash
# Package the minimal set needed to run openpi_rtc on the inference machine
# (which is NOT connected to the GPU/training machine; transfer via BOS).
#
# Contents:
#   - openpi_rtc/                  (probe, latency, eval, train-RTC code)
#   - src/openpi/                  (the JAX openpi package, incl. configs)
#   - scripts/benchmark_pi05_inference.py
#   - pyproject.toml + uv.lock     (recreate the venv on the target)
#   - checkpoint params + assets   (inference-only; train_state is omitted)
#
# Usage (from the repo root):
#   bash openpi_rtc/package_for_inference.sh [output.tar.gz] [checkpoint_dir]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT="${1:-/tmp/openpi_inference_bundle.tar.gz}"
CKPT="${2:-${OPENPI05_CHECKPOINT_49999:-}}"

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

mkdir -p "$(dirname "$OUT")"

# Repo files keep their relative layout; the checkpoint's params/assets are
# lifted to the bundle root so the extract dir *is* the checkpoint dir.
tar -czf "$OUT" \
  -C "$REPO_ROOT" \
  openpi_rtc \
  src/openpi \
  scripts/benchmark_pi05_inference.py \
  pyproject.toml \
  uv.lock \
  -C "$CKPT" \
  params \
  assets

if [[ "$OUT" == /* ]]; then
  echo "Bundle written to: $OUT"
else
  echo "Bundle written to: $REPO_ROOT/$OUT"
fi
echo "Size: $(du -h "$OUT" | cut -f1)"
echo
echo "Upload to BOS, then on the inference machine:"
echo "  tar -xzf $(basename "$OUT")"
echo "  uv sync            # or reuse a transferred .venv"
echo "  # GPU + model latency (no robot API needed):"
echo "  uv run scripts/benchmark_pi05_inference.py --checkpoint params --num-steps 10 --repeats 20"
echo "  # or, with one HDF5 file transferred, HDF5+decode latency:"
echo "  uv run python openpi_rtc/measure_latency.py --checkpoint . --hdf5 <one_hdf5_file>"
