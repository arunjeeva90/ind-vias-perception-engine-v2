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
#   FAST_LIVE=1   - disable video/log recording and use lightweight face-box tracking
#   ENABLE_106_RKNN=1 - opt in to internal-PoC driver-only 106-point NPU evidence
#
# Examples:
#   bash scripts/axon/run_dms_webcam_axon.sh 0
#   bash scripts/axon/run_dms_webcam_axon.sh 1 outputs/axon_cam1
#   HEADLESS=1 bash scripts/axon/run_dms_webcam_axon.sh 0 outputs/axon_headless_test
#
# Low-lag AXON mode (performance appears in the status consoles, not on video):
#   FAST_LIVE=1 bash scripts/axon/run_dms_webcam_axon.sh 1 --camera-fps 20 --inference-fps 12 --width 640 --height 480 --fourcc MJPG

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

# The retained COCO phone model uses the RKNN Model Zoo multi-output DFL
# contract. The integrated cabin-evidence parser does not yet support that
# contract, so this head/eye launcher must remain dummy instead of claiming a
# working phone integration. Use the preserved standalone phone tools later.
echo "[INFO] Phone detection disabled for this vehicle-test phase."
echo "       Retained standalone baseline: old_baseline_coco_phone_detector"
CABIN_ONNX_ARGS=(--cabin-evidence-backend dummy)

# Build display arguments
DISPLAY_ARGS=()
if [ "${HEADLESS:-0}" != "1" ]; then
    DISPLAY_ARGS=(--display --status-window --window-layout vehicle-test)
    echo "[INFO] DualSight three-window vehicle-test display enabled."
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

if [ "${ENABLE_106_RKNN:-0}" = "1" ]; then
    echo "[WARN] ENABLE_106_RKNN=1: internal-PoC-only InsightFace weights."
    echo "       Runtime will fail safe to MediaPipe/EAR if RKNNLite or /dev/rknpu is unavailable."
    RUN_ARGS+=(--landmark-106-backend rknn)
fi

if [ "${FAST_LIVE:-0}" = "1" ]; then
    echo "[INFO] FAST_LIVE=1 enabled: keeping all consoles, disabling video/heavy logs."
    RUN_ARGS+=(--axon-face-box-tracker)
else
    echo "[INFO] Recording full feedback bundle:"
    echo "       overlay MP4, state/trace/performance JSONL, events JSON/CSV,"
    echo "       learning memory, and session manifest JSON."
    RUN_ARGS+=(
        --output "$OUTPUT_DIR/webcam_output.mp4"
        --jsonl "$OUTPUT_DIR/webcam_state.jsonl"
        --debug-trace "$OUTPUT_DIR/webcam_trace.jsonl"
        --event-log "$OUTPUT_DIR/webcam_events.csv"
        --event-json "$OUTPUT_DIR/webcam_events.json"
        --learning-memory "$OUTPUT_DIR/webcam_learning.jsonl"
        --perf-jsonl "$OUTPUT_DIR/webcam_performance.jsonl"
        --session-json "$OUTPUT_DIR/webcam_session.json"
    )
fi

python apps/run_dms_demo.py \
    "${RUN_ARGS[@]}" \
    "${DMS_EXTRA_ARGS[@]}"

echo ""
echo "[INFO] DMS run complete. Output files in: $OUTPUT_DIR"
echo ""
