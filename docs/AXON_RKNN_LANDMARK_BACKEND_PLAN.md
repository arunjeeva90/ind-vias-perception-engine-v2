# AXON RKNN Landmark Backend Plan

This note tracks the RKNN/NPU path for AXON without changing the current DMS runtime default.

## Runtime Environments

AXON uses two Python virtual environments on purpose:

- `.venv` is the normal DMS runtime environment. It keeps the current MediaPipe FaceMesh path available and remains the default for the demo scripts.
- `.venv-rknn` is the RKNN environment. It carries Rockchip-specific packages such as `rknn.api` and `rknnlite.api`, plus conversion and runtime dependencies that may not be suitable for the default MediaPipe environment.

Keeping them separate avoids turning the stable CPU MediaPipe runtime into a board-specific dependency stack.

## RKNN Package Roles

`rknn.api` is the RKNN conversion toolkit. It is used offline or during model preparation to import an ONNX/TFLite model, configure preprocessing and quantization, build the RKNN graph, and export a `.rknn` model file.

`rknnlite.api` is the runtime API. It is the package the deployed DMS process will use on AXON to load a `.rknn` model and run inference on the RK3588 NPU.

## Current Face Landmark Backend

The current DMS face landmark backend is MediaPipe FaceMesh. It is CPU-bound in the present runtime and remains the default backend. No current demo path should require RKNN packages or the NPU.

## Future Model Path

The intended landmark acceleration path is:

1. Select or train a face landmark model in ONNX or TFLite format.
2. Convert the model with `rknn.api` into an RKNN artifact.
3. Load and execute that artifact with `rknnlite.api.RKNNLite` on AXON.
4. Convert model outputs into the existing `DMS FaceLandmarkResult` shape.

The placeholder `RKNNFaceLandmarkBackend` exists to reserve that integration point. Detection and postprocess will remain unimplemented until the exact model output layout is chosen.
