#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

if [ -d ".venv-rknn" ]; then
    echo "[INFO] Activating .venv-rknn"
    # shellcheck source=/dev/null
    source ".venv-rknn/bin/activate"
elif [ -d ".venv" ]; then
    echo "[INFO] Activating .venv"
    # shellcheck source=/dev/null
    source ".venv/bin/activate"
else
    echo "[WARN] No .venv-rknn or .venv found; using current Python environment"
fi

python apps/axon_rknn_probe.py
