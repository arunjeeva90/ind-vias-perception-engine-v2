# IND-VIAS Metric Monocular ADAS Perception Engine v2.1

This repository is an **atomic, layered workspace** for the IND-VIAS monocular ADAS perception stack. It is designed for Codex-assisted development and later migration to TDA4/TIDL/OpenVX style deployment.

## Important status

This is a **production-style scaffold**, not a trained production model. It contains:

- Separate folders for every backbone and head family
- Stable typed interfaces
- Dummy compilable implementations for every atomic component
- Geometry distance anchor
- Multi-anchor scale fusion
- Kalman tracking
- TTC fusion
- CAIS compute controller
- Sentinel FSM
- Safety gate
- CLI runners
- Unit tests

It does **not** include trained model weights for MobileNetV4, EfficientNet-Lite, YOLO, lane, depth, or DMS. Those must be trained or integrated later.

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

## Quick start on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
ind-vias-inspect
```

## Atomic source layout

```text
src/ind_vias_perception/
  apps/
  common/
  config/
  pipeline/
  perception/
    backbones/
      base/
      mobilenetv4_hybrid/
      efficientnet_lite/
      mobilevit/
      efficientvit/
      adapters/
    necks/
      fpn/
      bifpn/
    heads/
      detection/
      lane/
      freespace/
      ground_contact/
      depth/
      uncertainty/
      scene_quality/
      tsr/
      dms/
  geometry/
    calibration/
    ground_plane/
    scale_anchors/
    scale_fusion/
  temporal/
    trackers/
    motion_models/
    ugft/
  ttc/
    depth_ttc/
    expansion_ttc/
    flow_ttc/
    fusion/
  safety/
    sentinel_fsm/
    safety_gate/
    can/
  runtime/
    cais/
    profiling/
    logging/
  deployment/
    onnx/
    tidl/
    openvx/
```

## Recommended Codex workflow

Open this repo in Codex Desktop and use small tasks:

1. Implement ONNX detector adapter.
2. Implement real MobileNetV4-Hybrid backbone or integrate a permissive implementation.
3. Add lane/free-space model adapter.
4. Add ground-contact head parser.
5. Replace simple Kalman with IMM/UGTF.
6. Add TIDL export scripts.

See `docs/CODEX_TASKS.md`.
