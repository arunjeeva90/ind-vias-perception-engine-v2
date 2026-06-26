# AXON 106-Point Landmark PoC Plan

This plan prepares a local/internal Stage 2 probe for the InsightFace `2d106det` landmark model. The current 68-point RKNN landmark path remains the default, and the main DMS MediaPipe runtime is not part of this experiment.

The InsightFace pretrained model is internal-PoC-only unless separately licensed. Do not commit `models/dms/landmark_106.onnx`, converted RKNN artifacts, snapshots, or performance CSVs unless the model license is cleared for the intended use.

## Local-Only Workflow

1. Obtain the `2d106det` ONNX model manually from InsightFace.

   Source references:

   - https://github.com/deepinsight/insightface/tree/master/alignment/coordinate_reg
   - https://github.com/deepinsight/insightface/tree/master/model_zoo

2. Place the local ONNX file here:

```bash
models/dms/landmark_106.onnx
```

3. Inspect the ONNX contract:

```bash
python tools/rknn/inspect_landmark_onnx.py \
  --onnx models/dms/landmark_106.onnx
```

Expected high-level contract for `2d106det` is a `192x192` loose face crop input and direct 106-point 2D coordinate regression output. Confirm the exact input name, layout, and output shape locally before trusting the converted model.

4. Convert to RKNN for RK3588:

```bash
tools/rknn/convert_106_landmark_to_rknn.sh
```

The wrapper defaults to `192 192` input size and writes:

```bash
models/dms/landmark_106_rk3588.rknn
```

To override the input size:

```bash
tools/rknn/convert_106_landmark_to_rknn.sh 192 192
```

5. Benchmark the converted model:

```bash
python tools/rknn/benchmark_rknn_landmark.py \
  --model models/dms/landmark_106_rk3588.rknn \
  --image samples/face.jpg \
  --input-size 192 192
```

Compare mean, P50, P90, P99, and throughput against the current 68-point RKNN benchmark envelope of roughly 7.5-10 ms and 100+ inferences per second.

6. Run the live demo with YuNet, 106 landmarks, and CSV logging:

```bash
python tools/rknn/live_rknn_landmark_webcam.py \
  --model models/dms/landmark_106_rk3588.rknn \
  --input-size 192 192 \
  --landmark-count 106 \
  --detector yunet \
  --perf-log outputs/rknn_live_106_perf.csv
```

## Evaluation Notes

Use the existing Stage 2 comparison dimensions from `docs/AXON_PREMIUM_LANDMARK_MODEL_EVALUATION.md`:

- Visual stability under steady face crops.
- Landmark validity and normalized output range behavior.
- Side pose, head-down, glasses, partial occlusion, and low-light behavior.
- RKNN inference latency and end-to-end live FPS.
- Whether the extra 106 points improve DMS-relevant eye, mouth, and contour signals.

If the ONNX output is not `landmark_count * 2` direct normalized values, stop and document the actual output contract before modifying the live demo further.
