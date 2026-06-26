# AXON Landmark Model Source

## Selected PoC Model

- Model name: `face_landmark_Nx3x160x160.onnx`
- Local test path: `models/dms/landmark.onnx`
- Source repository: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/043_face_landmark
- Source archive: https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/043_face_landmark/resources.tar.gz
- Upstream model references:
  - https://github.com/610265158/face_landmark
  - https://github.com/610265158/Peppa_Pig_Face_Landmark
- License: Apache License 2.0 for the `043_face_landmark` model folder and the referenced upstream repositories.
- Local SHA-256: `509a6b89514fc3ec6941a3d6d6ab9c1a41e28685a15722b492b2398d33c363f0`

The ONNX file is ignored by git through the repository `*.onnx` rule, so the weight file is available for local RKNN testing without being committed.

## Model Shape

- Input name: `images`
- Input shape: `N x 3 x 160 x 160`
- Input dtype: `float32`
- Layout: NCHW

## Outputs

The graph exposes three float32 outputs:

- `Identity`: `N x 4`
- `Identity_1`: `N x 3`
- `Identity_2`: `N x 136`

The PINTO demo uses `Identity_2` as the landmark tensor and reshapes it to `N x 68 x 2`. The values are normalized `(x, y)` landmark coordinates relative to the detected face crop, then scaled back by crop width and height.

## Landmark Count

- Landmark count: 68
- Coordinate count: 136 values per face

## Notes and Blockers

This is not a PFLD model. A PFLD-style 68/98/106 landmark ONNX model would still be preferable if we find one with a clearly permissive source and weight license. For this PoC, this ShuffleNetV2-style 68-point model is small, Apache-2.0 licensed, directly available as ONNX, and has a simple coordinate output suitable for initial RKNN conversion tests.

Actual DMS postprocessing should not be wired yet. The mapping from this model's 68-point layout to the existing DMS `FaceLandmarkResult` indices still needs a deliberate compatibility layer, especially because current MediaPipe FaceMesh logic expects MediaPipe's denser landmark indexing.
