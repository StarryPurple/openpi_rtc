#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PROBE_PY:-python3}"
exec "$PY" gripper_test.py "$@"
