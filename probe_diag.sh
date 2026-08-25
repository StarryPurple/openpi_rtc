#!/usr/bin/env bash
# openpi_rtc 探测包入口。
# 用法（在 openpi-main 根目录）:
#   bash probe_diag.sh                  # 只读探测
#   bash probe_diag.sh --move-test      # 加右臂 J0 +3° 微动探针（人守急停）
#   PROBE_PY=<venv python> bash probe_diag.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PROBE_PY:-python3}"
exec "$PY" probe_diag.py "$@"
