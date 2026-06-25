#!/usr/bin/env bash
# run_dms_webcam_axon.sh
# Run DualSight DMS v0.2.9 on live webcam on AXON board
#
# Usage:
#   bash scripts/axon/run_dms_webcam_axon.sh [CAMERA_INDEX] [OUTPUT_DIR] [DMS_ARGS...]
#
# Arguments:
#   CAMERA_INDEX  - camera device index (default: 0)
#   OUTPUT_DIR    - output directory (default: outputs/axon_webcam_live)
#   DMS_ARGS      - optional extra args forwarded to apps/run_dms_demo.py
#
# Environment variables:
#   HEADLESS=1    - run without display (no --display, no --status-window)
#
# Examples:
#   bash scripts/axon/run_dms_webcam_axon.sh 0
#   bash scripts/axon/run_dms_webcam_axon.sh 1 outputs/axon_cam1
#   HEADLESS=1 bash scripts/axon/run_dms_webcam_axon.sh 0 outputs/axon_headless_test
#
# Low-lag AXON mode:
#   bash scripts/axon/run_dms_webcam_axon.sh 1 --camera-fps 20 --inference-fps 12 --width 640 --height 480 --fourcc MJPG --show-perf

set -e

# Arguments
CAMERA_INDEX="${1:-0}"
if [ "$#" -gt 0 ]; then
    shift
fi
OUTPUT_DIR="outputs/axon_webcam_live"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    OUTPUT_DIR="$1"
    shift
fi
DMS_EXTRA_ARGS=("$@")


# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================"
echo " AXON DMS Webcam Run"
echo "========================================"
echo "  Camera index: $CAMERA_INDEX"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Headless:     ${HEADLESS:-0}"
echo "  Extra args:   ${DMS_EXTRA_ARGS[*]:-(none)}"
echo "========================================"
echo ""

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    echo "[INFO] Activating .venv..."
    source .venv/bin/activate
else
    echo "[WARN] No .venv found. Using system Python."
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check for cabin ONNX model and class map
CABIN_ONNX_ARGS=()
MODEL_PATH="models/dms/cabin_objects.onnx"
CLASS_MAP_PATH="configs/dms/cabin_object_class_map_coco_phone.json"

if [ -f "$MODEL_PATH" ] && [ -f "$CLASS_MAP_PATH" ]; then
    echo "[INFO] Cabin ONNX model found. Enabling ONNX cabin evidence."
    CABIN_ONNX_ARGS=(--cabin-evidence-backend onnx --cabin-evidence-model "$MODEL_PATH" --cabin-evidence-class-map "$CLASS_MAP_PATH")
else
    echo "[WARN] Cabin ONNX model or class map not found."
    echo "       Model:     $MODEL_PATH ($([ -f "$MODEL_PATH" ] && echo 'EXISTS' || echo 'MISSING'))"
    echo "       Class map: $CLASS_MAP_PATH ($([ -f "$CLASS_MAP_PATH" ] && echo 'EXISTS' || echo 'MISSING'))"
    echo "       Running with dummy cabin evidence backend."
    echo ""
    CABIN_ONNX_ARGS=(--cabin-evidence-backend dummy)
fi

# Build display arguments
DISPLAY_ARGS=()
if [ "${HEADLESS:-0}" != "1" ]; then
    if [ "${FAST_LIVE:-0}" = "1" ]; then
        DISPLAY_ARGS=(--display)
        echo "[INFO] FAST_LIVE display mode enabled (--display only, no status window)"
    else
        DISPLAY_ARGS=(--display --status-window)
        echo "[INFO] Display mode enabled (--display --status-window)"
    fi
else
    echo "[INFO] Headless mode (no display window)"
fi

echo "[INFO] Starting DMS..."
echo ""

# Run DMS demo
RUN_ARGS=(
    --camera "$CAMERA_INDEX"
    --config configs/dms/dualsight_dms_axon.yaml
    --debug-overlay
    "${DISPLAY_ARGS[@]}"
    "${CABIN_ONNX_ARGS[@]}"
)

if [ "${FAST_LIVE:-0}" = "1" ]; then
    echo "[INFO] FAST_LIVE=1 enabled: disabling output video and heavy logs."
    RUN_ARGS+=(--axon-face-box-tracker)
else
    RUN_ARGS+=(
        --output "$OUTPUT_DIR/webcam_output.mp4"
        --jsonl "$OUTPUT_DIR/webcam_state.jsonl"
        --debug-trace "$OUTPUT_DIR/webcam_trace.jsonl"
        --event-log "$OUTPUT_DIR/webcam_events.csv"
        --event-json "$OUTPUT_DIR/webcam_events.json"
        --learning-memory "$OUTPUT_DIR/webcam_learning.jsonl"
    )
fi

python apps/run_dms_demo.py \
    "${RUN_ARGS[@]}" \
    "${DMS_EXTRA_ARGS[@]}"

echo ""
echo "[INFO] DMS run complete. Output files in: $OUTPUT_DIR"
echo ""
