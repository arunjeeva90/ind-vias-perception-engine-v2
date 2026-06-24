#!/usr/bin/env bash
# setup_axon_ubuntu.sh
# Prepare Vicharak AXON Ubuntu 22.04 ARM64 environment for DualSight DMS v0.2.9
# Target: CPU/OpenCV Stage 1 runtime
#
# Usage:
#   bash scripts/axon/setup_axon_ubuntu.sh
#
# This script is idempotent and safe to run multiple times.

set -e

echo "========================================"
echo " AXON Ubuntu 22.04 DMS Setup"
echo "========================================"
echo ""

# Print OS information
echo "[INFO] Detecting system..."
uname -a
echo ""
if [ -f /etc/os-release ]; then
    echo "[INFO] OS Release:"
    cat /etc/os-release | grep -E "^(NAME|VERSION|ID|PRETTY_NAME)="
    echo ""
fi

# Update package list
echo "[INFO] Updating apt package list..."
sudo apt update -y

# Install required system packages
echo "[INFO] Installing system packages..."
sudo apt install -y \
    python3 \
    python3-venv \
    python3-pip \
    git \
    v4l-utils \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential

echo ""
echo "[INFO] System packages installed."

# Determine project root (script is at scripts/axon/setup_axon_ubuntu.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "[INFO] Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Create virtual environment if not present
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
    echo "[INFO] Virtual environment created."
else
    echo "[INFO] Virtual environment .venv already exists."
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip/setuptools/wheel
echo "[INFO] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install AXON CPU requirements
# mediapipe may not have ARM64 wheels available -- install it separately so a
# failure does not abort the rest of the setup (the DMS degrades gracefully
# without it).
echo "[INFO] Installing AXON CPU requirements (excluding mediapipe)..."
pip install $(grep -v '^\s*#' requirements/requirements-axon-cpu.txt | grep -vi 'mediapipe' | tr '\n' ' ')

echo "[INFO] Attempting mediapipe install (optional on ARM64)..."
if pip install "mediapipe>=0.10,<1.0"; then
    echo "[INFO] mediapipe installed successfully."
else
    echo "[WARN] mediapipe install failed -- this is expected on some ARM64 systems."
    echo "       DMS will still run but face-detection quality may be reduced."
fi

echo ""
echo "========================================"
echo " Setup Complete"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "  source .venv/bin/activate"
echo "  bash scripts/axon/check_axon_env.sh"
echo "  bash scripts/axon/list_cameras.sh"
echo "  python apps/axon_camera_probe.py"
echo "  bash scripts/axon/run_dms_webcam_axon.sh 0"
echo ""
