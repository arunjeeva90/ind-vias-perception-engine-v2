#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="models/dms"
MODEL_PATH="${MODEL_DIR}/face_detection_yunet.onnx"
SOURCE_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

mkdir -p "${MODEL_DIR}"

echo "[INFO] Source URL: ${SOURCE_URL}"
echo "[INFO] License note: OpenCV Zoo YuNet directory is MIT licensed."

if [[ -f "${MODEL_PATH}" ]]; then
  echo "[INFO] Model already exists, not overwriting: ${MODEL_PATH}"
  exit 0
fi

curl -L "${SOURCE_URL}" -o "${MODEL_PATH}"
echo "[OK] Downloaded YuNet ONNX model: ${MODEL_PATH}"
