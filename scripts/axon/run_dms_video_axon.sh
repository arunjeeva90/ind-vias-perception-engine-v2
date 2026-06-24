#!/usr/bin/env bash
# run_dms_video_axon.sh
# Run DualSight DMS v0.2.9 on a video file on AXON board
#
# Usage:
#   bash scripts/axon/run_dms_video_axon.sh <INPUT_VIDEO> [OUTPUT_DIR]
#
# Arguments:
#   INPUT_VIDEO   - path to input video file (required)
#   OUTPUT_DIR    - output directory (default: outputs/axon_video_test)
#
# Environment variables:
#   HEADLESS=1    - run without display (no --display, no --status-window)
#
# Examples:
#   bash scripts/axon/run_dms_video_axon.sh samples/test.mp4 outputs/axon_video_test
#   HEADLESS=1 bash scripts/axon/run_dms_video_axon.sh samples/test.mp4 outputs/axon_video_headless

set -e

# Arguments
INPUT_VIDEO="${1:-}"
OUTPUT_DIR="${2:-outputs/axon_video_test}"

# Validate input video argument
if [ -z "$INPUT_VIDEO" ]; then
    echo "[ERROR] No input video path provided."
    echo ""
    echo "Usage: bash scripts/axon/run_dms_video_axon.sh <INPUT_VIDEO> [OUTPUT_DIR]"
    echo ""
    echo "Example:"
    echo "  bash scripts/axon/run_dms_video_axon.sh samples/test.mp4 outputs/axon_video_test"
    exit 1
fi

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Check input video exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "[ERROR] Input video file not found: $INPUT_VIDEO"
    echo "        Please provide a valid path to a video file."
    exit 1
fi

echo "========================================"
echo " AXON DMS Video File Run"
echo "========================================"
echo "  Input video: $INPUT_VIDEO"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Headless:    ${HEADLESS:-0}"
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
    DISPLAY_ARGS=(--display --status-window)
    echo "[INFO] Display mode enabled (--display --status-window)"
else
    echo "[INFO] Headless mode (no display window)"
fi

echo ""
echo "[INFO] Starting DMS on video file..."
echo ""

# Derive output filename from input video
VIDEO_BASENAME=$(basename "$INPUT_VIDEO" | sed 's/\.[^.]*$//')

# Run DMS demo
python apps/run_dms_demo.py \
    --video "$INPUT_VIDEO" \
    --config configs/dms/dualsight_dms_axon.yaml \
    --debug-overlay \
    "${DISPLAY_ARGS[@]}" \
    "${CABIN_ONNX_ARGS[@]}" \
    --output "$OUTPUT_DIR/${VIDEO_BASENAME}_output.mp4" \
    --jsonl "$OUTPUT_DIR/${VIDEO_BASENAME}_state.jsonl" \
    --debug-trace "$OUTPUT_DIR/${VIDEO_BASENAME}_trace.jsonl" \
    --event-log "$OUTPUT_DIR/${VIDEO_BASENAME}_events.csv" \
    --event-json "$OUTPUT_DIR/${VIDEO_BASENAME}_events.json" \
    --learning-memory "$OUTPUT_DIR/${VIDEO_BASENAME}_learning.jsonl"

echo ""
echo "[INFO] DMS video run complete. Output files in: $OUTPUT_DIR"
echo ""
