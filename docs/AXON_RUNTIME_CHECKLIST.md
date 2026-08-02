# AXON Runtime Checklist

Quick checklist for verifying DualSight DMS v0.2.9 is running on AXON.

## Pre-flight Checks

- [ ] Ubuntu 22.04 booted on AXON
- [ ] Repository cloned (`git clone ... && git checkout feature/axon-runtime-v029`)
- [ ] `.venv` created (`python3 -m venv .venv`)
- [ ] Virtual environment activated (`source .venv/bin/activate`)
- [ ] Requirements installed (`bash scripts/axon/setup_axon_ubuntu.sh`)
- [ ] Environment check passes (`bash scripts/axon/check_axon_env.sh`)

## Camera Checks

- [ ] Camera visible under `/dev/video*` (`bash scripts/axon/list_cameras.sh`)
- [ ] Camera probe successful (`python apps/axon_camera_probe.py`)
- [ ] Sample frame saved to `outputs/axon_camera_probe/`

## Runtime Checks

- [ ] DMS live webcam launched (`bash scripts/axon/run_dms_webcam_axon.sh 0`)
- [ ] `IND-VIAS DualSight DMS - Video` window visible
- [ ] `IND-VIAS DualSight DMS - Status` window visible
- [ ] `IND-VIAS Vehicle Monitor` window visible
- [ ] HMI banner is in the separate header above the video and does not cover
      camera pixels
- [ ] Performance telemetry appears in the status/vehicle consoles, not over
      the driver face
- [ ] Head-pose axes and road/gaze vector visible on a validated driver face
- [ ] RGB head-pose triad is visible in the left instrument strip and does not
      cover the driver face
- [ ] Axis legend reports red/X lateral, green/Y vertical, blue/Z depth
- [ ] Head Angle and Raw/Relative values are fully visible in the full-width
      Head & Road card
- [ ] Largest validated face inside the driver ROI is labelled `DRIVER`
- [ ] Other faces, including smaller faces inside the driver ROI, are labelled
      `PASSENGER` and have boxes without landmark/pose overlays
- [ ] Status reports direct `FaceMesh` backend for the mounted-driver profile
- [ ] Status reports `CPU / MediaPipe XNNPACK`, `NPU NOT ACTIVE`, and
      `0.00 (inactive)` TOPS for the current integrated runtime
- [ ] Feature/model latency, FPS, CPU, and RAM fields update in the consoles
- [ ] Startup/standby suppresses alerts at or below 30 km/h
- [ ] `+` raises simulated speed above 30 km/h and activates DMS monitoring
- [ ] Eye runtime reports `LANDMARK_EAR`; unaccepted Eye CNN reports `DISABLED`
- [ ] Output logs generated in `outputs/axon_webcam_live/`
- [ ] CPU/FPS/latency observed in the premium status consoles
- [ ] Headless run verified (`HEADLESS=1 bash scripts/axon/run_dms_webcam_axon.sh 0 outputs/axon_headless_test`)

## Output Verification

- [ ] `*_output.mp4` file created
- [ ] `*_state.jsonl` file created
- [ ] `*_trace.jsonl` file created
- [ ] `*_events.csv` file created
- [ ] `*_events.json` file created
- [ ] `*_learning.jsonl` file created

## Notes

- The retained phone baseline is
  `models/mobile_phone_detector/yolov8n.onnx` plus
  `configs/dms/cabin_object_class_map_coco_phone.json`. It remains a standalone
  Model Zoo ONNX/RKNN baseline because the integrated parser does not yet
  support its multi-output DFL tensor contract.
- Reviewed eye and seat-belt ONNX classifiers remain disabled until their
  held-out metrics and AXON runtime checks pass.
- When enabling them, verify the ONNX file and matching `.metadata.json` class
  map are both present; missing models must leave eye EAR fallback active and
  seat-belt state `UNKNOWN`.
- RKNN calibration lists must contain training rows only. Conversion success is
  not runtime acceptance; compare ONNX and RKNN on identical samples.
- `models/mobile_phone_detector/yolov8n.rknn` and the landmark RKNN artefacts
  remain available to their standalone tools. They are not silently substituted
  into the integrated DMS pipeline.
- Expected FPS on CPU: verify on the mounted camera; the current processing
  width is 480 pixels and results depend on face size, lighting, and display load
- Use `FAST_LIVE=1` for the in-vehicle console test without video/log recording
- `ENABLE_106_RKNN=1` explicitly opts into the internal-PoC-only driver 106
  landmark model. Confirm `/dev/rknpu*` exists first. The runtime must fall back
  safely when the NPU driver is absent.
- The AXON head/eye profile disables MediaPipe Hands until the phone phase and
  uses direct FaceMesh to avoid the slower proposal-plus-crop pass.
- Use `HEADLESS=1` for SSH sessions without display
- Reduce `frame_resize_width` in config if FPS is too low
