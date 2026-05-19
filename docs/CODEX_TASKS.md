# Codex Task Backlog

## Task 1: Replace dummy detection head
Implement `perception/heads/detection/onnx_detection_head.py` with ONNXRuntime support, letterbox preprocessing, NMS, class mapping, and tests.

## Task 2: Add MobileNetV4-Hybrid implementation
Implement or integrate a permissively licensed MobileNetV4-Hybrid backbone behind `BackboneProtocol`. Do not copy incompatible code.

## Task 3: Add lane/free-space model adapters
Add segmentation tensor parsing for lane boundary and drivable area masks.

## Task 4: Add ground-contact parser
Convert ground-contact heatmaps/keypoints into `(u_gc, v_gc)` per detection.

## Task 5: Implement IMM/UGTF
Replace the simple Kalman tracker with IMM models and uncertainty-gated fusion.

## Task 6: Add deployment export
Add ONNX export, calibration dataset collection, and TIDL import documentation.

## Task 7: Add scenario tests
Create tests for cut-in, occlusion, glare, static ego, and dense traffic.
