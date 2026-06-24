#!/usr/bin/env bash
# check_axon_env.sh
# Check whether AXON environment is ready to run DualSight DMS v0.2.9
#
# Usage:
#   bash scripts/axon/check_axon_env.sh
#
# Prints PASS/WARN/FAIL style messages for each check.

echo "========================================"
echo " AXON Environment Check"
echo "========================================"
echo ""

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

pass_msg() {
    echo "[PASS] $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

warn_msg() {
    echo "[WARN] $1"
    WARN_COUNT=$((WARN_COUNT + 1))
}

fail_msg() {
    echo "[FAIL] $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

# System info
echo "--- System Info ---"
echo "Kernel: $(uname -a)"
echo ""

if [ -f /etc/os-release ]; then
    echo "OS:"
    cat /etc/os-release | grep -E "^(PRETTY_NAME|VERSION)=" || true
    echo ""
fi

if command -v lscpu &> /dev/null; then
    echo "CPU:"
    lscpu | grep -E "^(Architecture|Model name|CPU\(s\)|CPU max MHz)" || true
    echo ""
else
    warn_msg "lscpu not available"
fi

# Python
echo "--- Python Environment ---"
if command -v python3 &> /dev/null; then
    PYVER=$(python3 --version 2>&1)
    pass_msg "Python3 found: $PYVER"
else
    fail_msg "python3 not found"
fi

if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
    PIPVER=$(pip3 --version 2>&1 || pip --version 2>&1)
    pass_msg "pip found: $PIPVER"
else
    fail_msg "pip not found"
fi

# Virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    pass_msg "Virtual environment active: $VIRTUAL_ENV"
elif [ -d ".venv" ]; then
    warn_msg ".venv exists but is not activated. Run: source .venv/bin/activate"
else
    warn_msg "No .venv found. Run: bash scripts/axon/setup_axon_ubuntu.sh"
fi

echo ""
echo "--- Python Packages ---"

# numpy
if python3 -c "import numpy; print(f'numpy {numpy.__version__}')" 2>/dev/null; then
    pass_msg "numpy importable"
else
    fail_msg "numpy import failed"
fi

# cv2
CV2_VER=$(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null)
if [ -n "$CV2_VER" ]; then
    pass_msg "cv2 importable: version $CV2_VER"
else
    fail_msg "cv2 (OpenCV) import failed"
fi

# mediapipe
if python3 -c "import mediapipe; print(f'mediapipe {mediapipe.__version__}')" 2>/dev/null; then
    pass_msg "mediapipe importable"
else
    warn_msg "mediapipe import failed (DMS face detection may not work)"
fi

# onnxruntime
if python3 -c "import onnxruntime; print(f'onnxruntime {onnxruntime.__version__}')" 2>/dev/null; then
    pass_msg "onnxruntime importable"
else
    warn_msg "onnxruntime import failed (cabin ONNX evidence will not work)"
fi

echo ""
echo "--- Camera Devices ---"

DEVS=$(ls /dev/video* 2>/dev/null)
if [ -n "$DEVS" ]; then
    pass_msg "Camera devices found:"
    echo "$DEVS"
else
    warn_msg "No /dev/video* devices found"
fi

if command -v v4l2-ctl &> /dev/null; then
    echo ""
    echo "v4l2-ctl --list-devices:"
    v4l2-ctl --list-devices 2>/dev/null || warn_msg "v4l2-ctl --list-devices failed"
else
    warn_msg "v4l2-ctl not available (install v4l-utils)"
fi

echo ""
echo "--- System Resources ---"
echo "Disk:"
df -h / 2>/dev/null || true
echo ""
echo "Memory:"
free -h 2>/dev/null || true

echo ""
echo "--- Required Files ---"

# Model file
if [ -f "models/dms/cabin_objects.onnx" ]; then
    pass_msg "models/dms/cabin_objects.onnx exists"
else
    warn_msg "models/dms/cabin_objects.onnx NOT found (cabin ONNX evidence will use dummy backend)"
fi

# Class map
if [ -f "configs/dms/cabin_object_class_map_coco_phone.json" ]; then
    pass_msg "configs/dms/cabin_object_class_map_coco_phone.json exists"
else
    warn_msg "configs/dms/cabin_object_class_map_coco_phone.json NOT found"
fi

# DMS app
if [ -f "apps/run_dms_demo.py" ]; then
    pass_msg "apps/run_dms_demo.py exists"
else
    fail_msg "apps/run_dms_demo.py NOT found"
fi

# AXON config
if [ -f "configs/dms/dualsight_dms_axon.yaml" ]; then
    pass_msg "configs/dms/dualsight_dms_axon.yaml exists"
else
    warn_msg "configs/dms/dualsight_dms_axon.yaml NOT found"
fi

echo ""
echo "========================================"
echo " Summary: PASS=$PASS_COUNT  WARN=$WARN_COUNT  FAIL=$FAIL_COUNT"
echo "========================================"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "Some checks FAILED. Please resolve before running DMS."
    exit 1
else
    echo ""
    echo "Environment looks ready. Proceed with camera probe and DMS run."
    exit 0
fi
