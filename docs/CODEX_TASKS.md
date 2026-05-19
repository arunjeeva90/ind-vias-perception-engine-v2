# Codex Task Backlog

## Task 1: Replace dummy detection head
Implement `perception/heads/detection/onnx_detection_head.py` with OpenCV DNN ONNX loading, letterbox preprocessing, NMS, class mapping, and tests.

## Using a real ONNX detector

This repo does not ship trained detector weights. To try a real detector, place your ONNX model at:

```text
models/weights/detector.onnx
```

Then switch `configs/default.yaml` to the ONNX backend:

```yaml
detection:
  backend: onnx
```

Smoke-test that OpenCV DNN can find and load the model:

```powershell
python scripts/check_onnx_detector.py --config configs/default.yaml
```

Run image inference with visualization:

```powershell
python -m ind_vias_perception --image examples/test_frame.jpg --output examples/out_frame.jpg
```

Add the debug overlay to show the active detection backend, track ids, distance, TTC, and the safety payload:

```powershell
python -m ind_vias_perception --image examples/test_frame.jpg --output examples/out_frame.jpg --debug-overlay
```

The ONNX detector is only a temporary 3A Object Detection provider. It is converted into existing `Detection` objects and still flows through `MetricMonocularPipeline`, ground-contact/depth/uncertainty, scale anchors, tracker, TTC, CAIS, Sentinel FSM, SafetyGate, and visualization.

Licensing note: third-party YOLO/ONNX models can carry GPL, AGPL, commercial, dataset, or export restrictions. Verify the model architecture, weights, labels, and training data licenses before integrating or distributing them.

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
