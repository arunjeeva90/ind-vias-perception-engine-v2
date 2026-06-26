#!/usr/bin/env bash
set -euo pipefail

INPUT_W="${1:-192}"
INPUT_H="${2:-192}"
ONNX_MODEL="models/dms/landmark_106.onnx"
RKNN_MODEL="models/dms/landmark_106_rk3588.rknn"

if [[ -d ".venv-rknn" ]]; then
  # shellcheck disable=SC1091
  source ".venv-rknn/bin/activate"
fi

if [[ ! -f "${ONNX_MODEL}" ]]; then
  echo "[ERROR] Missing local 106-point ONNX model: ${ONNX_MODEL}" >&2
  echo "[INFO] Obtain InsightFace 2d106det manually and place it at ${ONNX_MODEL}." >&2
  echo "[INFO] Do not commit this model unless licensing is cleared." >&2
  exit 1
fi

echo "[INFO] Inspecting ONNX model: ${ONNX_MODEL}"
python tools/rknn/inspect_landmark_onnx.py --onnx "${ONNX_MODEL}"

echo "[INFO] Converting 106-point landmark ONNX to RKNN"
python tools/rknn/convert_landmark_onnx_to_rknn.py \
  --onnx "${ONNX_MODEL}" \
  --output "${RKNN_MODEL}" \
  --target-platform rk3588 \
  --input-size "${INPUT_W}" "${INPUT_H}"

echo "[OK] Wrote RKNN model: ${RKNN_MODEL}"
