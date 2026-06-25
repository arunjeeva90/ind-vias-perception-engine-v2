# RKNN Face Landmark PoC Tools

These tools prepare the first standalone AXON RK3588 face landmark proof of concept. They are not wired into the DMS runtime yet, and the default MediaPipe FaceMesh path remains unchanged.

## Workflow

The intended model flow is:

1. Start from a face landmark model exported as ONNX or TFLite.
2. Convert the model to `.rknn` with the RKNN toolkit.
3. Load the `.rknn` file with `rknnlite.api.RKNNLite` on AXON RK3588.
4. Run inference on a face crop or test image.
5. Map model outputs into the existing `FaceLandmarkResult` contract used by DMS.

For the first PoC, conversion uses `build(do_quantization=False)` so the path can validate model compatibility and runtime execution before calibration or int8 quantization is introduced.

## Environment

Use `.venv-rknn` for these tools:

```bash
source .venv-rknn/bin/activate
```

The normal `.venv` remains the MediaPipe DMS runtime environment.

## Convert ONNX to RKNN

```bash
python tools/rknn/convert_landmark_onnx_to_rknn.py \
  --onnx models/dms/landmarks.onnx \
  --output models/dms/landmarks.rknn \
  --target-platform rk3588 \
  --input-size 192 192
```

An optional `--dataset` path is accepted for future quantized builds. It is logged but not used while `do_quantization=False`.

## Smoke Test RKNNLite Runtime

```bash
python tools/rknn/test_rknn_landmark_runtime.py \
  --model models/dms/landmarks.rknn \
  --image samples/face.jpg
```

The runtime test resizes the image to `192x192`, runs inference, and prints output tensor shapes, dtypes, and value ranges.

## Postprocessing Gap

Actual landmark postprocessing depends on the chosen model output format. Some models emit normalized `(x, y)` pairs, some emit heatmaps, and others include confidence or visibility channels. The RKNN placeholder backend should only map outputs to `FaceLandmarkResult` after the final landmark model and output layout are selected.
