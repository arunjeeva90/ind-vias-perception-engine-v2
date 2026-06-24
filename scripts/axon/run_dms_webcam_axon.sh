#!/usr/bin/env bash
# run_dms_webcam_axon.sh
# Run DualSight DMS v0.2.9 on live webcam on AXON board
#
# Usage:
#   bash scripts/axon/run_dms_webcam_axon.sh [CAMERA_INDEX] [OUTPUT_DIR]
#
# Arguments:
#   CAMERA_INDEX  - camera device index (default: 0)
#   OUTPUT_DIR    - output directory (default: outputs/axon_webcam_live)
#
# Environment variables:
#   HEADLESS=1    - run without display (no --display, no --status-window)
#
# Examples:
#   bash scripts/axon/run_dms_webcam_axon.sh 0
#   bash scripts/axon/run_dms_webcam_axon.sh 1 outputs/axon_cam1
#   HEADLESS=1 bash scripts/axon/run_dms_webcam_axon.sh 0 outputs/axon_headless_test

set -e

# Arguments
CAMERA_INDEX="${1:-0}"
OUTPUT_DIR="${2:-outputs/axon_webcam_live}"

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
    echo "       Running without cabin ONNX evidence (dummy backend)."
    echo ""
fi

# Build display arguments
DISPLAY_ARGS=()
if [ "${HEADLESS:-0}" != "1" ]; then
    DISPLAY_ARGS=(--display --status-window)
    echo "[INFO] Display mode enabled (--display --status-window)"
else
    echo "[INFO] Headless mode (no display window)"
fi

echo ""
echo "[INFO] Starting DMS..."
echo ""

# Run DMS demo
python apps/run_dms_demo.py \
    --camera "$CAMERA_INDEX" \
    --config configs/dms/dualsight_dms_axon.yaml \
    --debug-overlay \
    "${DISPLAY_ARGS[@]}" \
    "${CABIN_ONNX_ARGS[@]}" \
    --output "$OUTPUT_DIR/webcam_output.mp4" \
    --jsonl "$OUTPUT_DIR/webcam_state.jsonl" \
    --debug-trace "$OUTPUT_DIR/webcam_trace.jsonl" \
    --event-log "$OUTPUT_DIR/webcam_events.csv" \
    --event-json "$OUTPUT_DIR/webcam_events.json" \
    --learning-memory "$OUTPUT_DIR/webcam_learning.jsonl"

echo ""
echo "[INFO] DMS run complete. Output files in: $OUTPUT_DIR"
echo ""
