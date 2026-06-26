# AXON Premium Landmark Model Evaluation

This document starts Stage 2 of the premium AXON DMS model path. The goal is to evaluate a higher-grade landmark model for the RKNN proof of concept while leaving the main DMS MediaPipe runtime untouched.

Do not commit downloaded model weights unless the license is clearly permissive for the intended use. For research-only or unclear model licenses, keep artifacts local and document the source.

## Selection Criteria

Prefer candidates with these properties:

- ONNX is directly available.
- Input size is close to the current crop path: 112, 128, 160, or 192.
- Output is direct landmark coordinates rather than heatmaps or task-specific bundles.
- License is permissive, or at least clearly usable for an internal proof of concept.
- RKNN conversion is likely to preserve a simple output contract.

## Candidate Summary

| Candidate | Source URL | License | Input Shape | Output Shape | Landmark Count | ONNX Available | RKNN Conversion Risk | Commercial/OEM Usage Concern |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| InsightFace `2d106det` coordinate regression | https://github.com/deepinsight/insightface/tree/master/alignment/coordinate_reg and https://github.com/deepinsight/insightface/tree/master/model_zoo | Code is MIT, but InsightFace states pretrained models and training data are non-commercial research only | `1x3x192x192` loose face crop | Expected direct coordinate vector, likely `1x212`; confirm with `inspect_landmark_onnx.py` after local download | 106 2D points | Yes, Google Drive link in InsightFace docs | Medium-low technically: MobileNet-0.5 coordinate regression is RKNN-friendly, but preprocessing and coordinate scaling must be replicated exactly | High. Treat as internal PoC only unless a commercial license is obtained |
| OpenVINO `facial-landmarks-98-detection-0001` HRNet | https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/facial-landmarks-98-detection-0001 | Open Model Zoo model metadata points to Apache 2.0 license | `1x3x64x64` BGR | `1x98x16x16` heatmaps | 98 points | No direct ONNX artifact in the Open Model Zoo entry; FP32/FP16/INT8 OpenVINO IR artifacts are provided | High: not direct landmarks, input is smaller than preferred, and IR-to-ONNX or source reconstruction would add risk | Lower than InsightFace if Apache 2.0 applies to the model artifacts, but confirm before shipping |
| MediaPipe Face Landmarker / FaceMesh-V2 | https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker | Google documentation content is CC BY 4.0 and code samples are Apache 2.0; model bundle usage terms should be verified separately | Bundle lists FaceDetector `192x192`, FaceMesh-V2 `256x256`, Blendshape `1x146x2` | Face mesh output estimates 478 3D landmarks; optional blendshape output has 52 scores | 478 dense 3D points | No direct ONNX in the standard task bundle | High: task bundle/TFLite conversion, dense output mapping, and runtime integration are outside the current RKNN landmark PoC | Medium/unclear. Use only as an optional visualization/reference candidate until model terms and conversion path are explicit |

## Initial Ranking

1. InsightFace `2d106det` is the best technical Stage 2 probe because it is ONNX, 192 input, direct-coordinate, and lightweight. The license blocks OEM use without separate permission, so it should remain an internal benchmark candidate.
2. OpenVINO `facial-landmarks-98-detection-0001` is the best 98-point reference candidate, but it is not an easy drop-in replacement because it outputs heatmaps and is distributed as OpenVINO IR rather than ONNX.
3. MediaPipe Face Landmarker is useful only as a dense visualization baseline. It is not a near-term RKNN replacement for the current 68-point model.

## Evaluation Workflow

1. Keep candidate downloads outside git unless license review allows committing the artifact.
2. Inspect each ONNX candidate before conversion:

```bash
python tools/rknn/inspect_landmark_onnx.py \
  --onnx models/dms/candidate_landmark.onnx
```

3. Record input shape, output shape, opset, file size, and initializer count in this document or a follow-up model note.
4. Convert only candidates with a simple enough graph and clear output contract.
5. Benchmark RKNN latency with the existing RKNN benchmark tool before changing the live demo.

## Comparison Plan

Use the existing live demo and CSV logging after a candidate is converted, but do not modify the live demo until the model contract is known.

Compare these dimensions:

- Visual stability: landmarks should not jitter excessively on a steady face crop.
- Landmark validity: output should remain finite, within the expected coordinate range, and anatomically plausible.
- Side pose and head-down behavior: evaluate yaw, pitch-down, glasses, partial occlusion, and low cabin light.
- Inference latency: compare mean, P50, P90, and P99 RKNN inference timing.
- FPS: compare end-to-end live FPS with the same detector and camera settings.
- Output point count: record whether the model gives 68, 98, 106, 478, or another count, and map which points are useful for DMS signals.

## Stage 2 Recommendation

Start with a local-only InsightFace `2d106det` inspection and RKNN conversion attempt. If conversion succeeds and latency stays near the current 7.5-10 ms benchmark envelope, use it as the premium landmark quality baseline while legal/commercial licensing is resolved.

In parallel, keep OpenVINO's 98-point model as a license-friendlier reference, but do not prioritize it unless we are willing to implement heatmap postprocessing and accept a smaller `64x64` input.
