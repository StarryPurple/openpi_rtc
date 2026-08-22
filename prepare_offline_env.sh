#!/usr/bin/env bash
# Prepare offline install artifacts for the robot PC (which has no internet
# and no uv). Run this on a networked Linux x86_64 machine -- the GPU /
# training machine is ideal (it already has uv and access to PyPI/GitHub).
#
# Produces (in this repository root):
#   requirements-offline.txt   full dependency list exported from uv.lock
#                              (editable local packages openpi / openpi-client
#                              are stripped: they are imported straight from
#                              the bundle, no pip install needed)
#   wheels/                    all wheels for linux x86_64 / cp311
#                              (incl. lerobot built from its git source)
#   python311/                 standalone Python 3.11 (uv-managed build)
#
# Then package_for_inference.sh picks wheels/ python311/ and
# requirements-offline.txt up automatically.
#
# Usage (repo root, on the networked machine):
#   bash prepare_offline_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

command -v uv >/dev/null 2>&1 || { echo "ERROR: 本机需要 uv（工控机不需要）" >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

echo "==> [1/5] 导出依赖清单 (uv.lock -> requirements-offline.txt)"
uv export --no-hashes -o requirements-offline.full.txt
grep -v '^-e ' requirements-offline.full.txt > requirements-offline.txt
rm -f requirements-offline.full.txt
echo "    $(wc -l < requirements-offline.txt) 行依赖"

echo "==> [2/5] 安装独立 Python 3.11"
uv python install 3.11

echo "==> [3/5] 下载所有 wheel (linux x86_64 / cp311，含 lerobot git 源)"
rm -rf wheels
mkdir -p wheels
uv pip download --python 3.11 -r requirements-offline.txt -d wheels/
echo "    wheels 数量: $(find wheels -name '*.whl' | wc -l)"

echo "==> [4/5] 复制独立 Python 3.11 -> python311/"
PY="$(uv python find 3.11)"
PYROOT="$(cd "$(dirname "$PY")/.." && pwd)"
rm -rf python311
cp -a "$PYROOT" python311
echo "    $(python311/bin/python3.11 -V)"

echo "==> [5/5] 完成"
echo
echo "下一步（本机）:"
echo "  bash package_for_inference.sh /tmp/openpi_inference_bundle.tar.gz \$OPENPI05_CHECKPOINT_49999"
echo
echo "工控机离线安装（bundle 内, 无网无 uv）:"
echo "  python311/bin/python3.11 -m venv .venv"
echo "  .venv/bin/pip install --no-index --find-links wheels -r requirements-offline.txt"
echo "  # openpi/openpi-client 无需 pip 安装，脚本从 rtc_control 根直接 import"
echo "  .venv/bin/python run_robot.py --config pi05-task_00031_yulong-xtrainer --checkpoint . \\"
echo "    --robot robot_xtrainer:XtrainerRobot --robot-type \"Nova 2\" --episodes 1"
