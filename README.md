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

Use the dedicated ONNX demo config. `configs/default.yaml` intentionally stays on the dummy backend so the repo runs without model weights:

```yaml
detection:
  backend: onnx
```

For temporary COCO YOLOv8n testing, use `configs/yolov8n_coco_demo.yaml`. That config maps only the COCO IDs relevant to ADAS (`pedestrian`, `cyclist`, `car`, `motorcycle`, `bus`, `truck`). It is only for temporary COCO YOLO testing; the production class map remains IND-VIAS-specific.

Smoke-test that OpenCV DNN can find and load the model:

```powershell
python scripts/check_onnx_detector.py --config configs/onnx_demo.yaml
```

Run image inference with visualization:

```powershell
python -m ind_vias_perception --config configs/onnx_demo.yaml --image examples/test_frame.jpg --output examples/out_frame.jpg
```

Add the debug overlay to show the active detection backend, track ids, distance, TTC, and the safety payload:

```powershell
python -m ind_vias_perception --config configs/onnx_demo.yaml --image examples/test_frame.jpg --output examples/out_frame.jpg --debug-overlay
```

For video runs, omit `--max-frames` to process the full input video. `--max-frames` is only a debug/development limiter: for example, `--max-frames 300` on a 30 FPS video processes about 10 seconds (`300 / 30 = 10`). At video start, the CLI prints the input FPS, estimated total frames when available, the `max_frames` value, and whether the run is full-video or limited.

### Debug overlay user manual

The debug overlay is an engineering view, not a final HMI. See `docs/DEBUG_OVERLAY_USER_MANUAL.md` for field meanings, warning interpretation, distance labels, CAIS/Sentinel state, and the config thresholds that affect cut-in and FCW behavior.

The ONNX detector is only a temporary 3A Object Detection provider. It is converted into existing `Detection` objects and still flows through `MetricMonocularPipeline`, ground-contact/depth/uncertainty, scale anchors, tracker, TTC, CAIS, Sentinel FSM, SafetyGate, and visualization.

Licensing note: third-party YOLO/ONNX models can carry GPL, AGPL, commercial, dataset, or export restrictions. Verify the model architecture, weights, labels, and training data licenses before integrating or distributing them.

## Calibrating phone-mounted demo videos

Phone videos need a rough ground-plane calibration before monocular distance numbers are useful. This repo includes a helper for demo tuning only; it does not claim production accuracy.

For 1440x1440 phone footage, start from:

```text
configs/phone_demo_1440.yaml
```

Pick a frame with a visible target at a known camera-relative distance, note the target bounding box, then run:

```powershell
python scripts/tune_ground_distance.py --config configs/phone_demo_1440.yaml --image examples/test_frame.jpg --bbox 100,200,300,1000 --known-distance-m 10.0
```

The script prints the bbox bottom-center ground contact, current `Dcam`, current bumper-relative `Dbump` when `vehicle.camera_to_front_bumper_offset_m` is configured, plus suggested `fy` and `horizon_y` values. Use those suggestions to iterate on the demo config, then rerun the normal pipeline through `python -m ind_vias_perception`; the script is only an offline tuning aid and does not bypass `MetricMonocularPipeline`.

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
