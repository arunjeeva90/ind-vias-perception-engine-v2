# AXON Deployment Guide - DualSight DMS v0.2.9

## 1. Purpose

This document provides step-by-step instructions to deploy and run the
DualSight DMS v0.2.9 on the Vicharak AXON board running Ubuntu 22.04 ARM64.

The integrated DMS currently uses the stable MediaPipe/OpenCV CPU path. RKNN
artefacts are used only by model-specific tools that have their own validation
evidence; conversion alone is not treated as permission to replace a runtime
backend.

## 2. Target Board and OS

- **Board:** Vicharak AXON
- **OS:** Ubuntu 22.04 (ARM64/aarch64)
- **Integrated runtime:** MediaPipe/OpenCV CPU
- **Available NPU tools:** qualified model-specific RKNN demos/converters
- **Camera:** USB webcam via /dev/video* or video file input

> If you are running a different Ubuntu version, some package names may need
> minor adjustments (e.g., `libgl1` vs `libgl1-mesa-glx`).

## 3. Branch Name

```
feature/axon-runtime-v029
```

Based on commit: `6576c94 Stabilize DualSight DMS v0.2.9 simplified cabin phone logic`

## 4. Original AXON Baseline Changes

The original AXON deployment baseline added:

- `requirements/requirements-axon-cpu.txt` - ARM64 CPU runtime Python packages
- `scripts/axon/setup_axon_ubuntu.sh` - Ubuntu 22.04 environment setup
- `scripts/axon/check_axon_env.sh` - Environment readiness check
- `scripts/axon/list_cameras.sh` - Camera device listing
- `scripts/axon/run_dms_webcam_axon.sh` - Live webcam DMS runner
- `scripts/axon/run_dms_video_axon.sh` - Video file DMS runner
- `configs/dms/dualsight_dms_axon.yaml` - AXON-optimized DMS config
- `apps/axon_camera_probe.py` - OpenCV camera probe utility
- `docs/AXON_DEPLOYMENT.md` - This deployment guide
- `docs/AXON_RUNTIME_CHECKLIST.md` - Quick runtime checklist

## 5. Original Baseline Exclusions

The original deployment baseline did not modify DMS core logic, FSM behavior,
phone ROI/gating, or seat-belt inference.

The reviewed 2026-07-30 handoff integration adds disabled-by-default ONNX eye
and seat-belt classifier hooks, deterministic training tooling, and a
one-class phone map. It preserves the existing FSM, phone evidence fusion, and
EAR fallback. These optional classifiers must remain disabled until their
held-out metrics and AXON runtime checks pass. See
`reports/handoff_integration/20260730_analysis_and_build.md`.

## 6. Clone Repo on AXON

```bash
cd ~
git clone https://github.com/arunjeeva90/ind-vias-perception-engine-v2.git
cd ind-vias-perception-engine-v2
git checkout feature/axon-runtime-v029
```

## 7. Create/Activate Virtual Environment

```bash
cd ~/ind-vias-perception-engine-v2
python3 -m venv .venv
source .venv/bin/activate
```

## 8. Run Setup Script

```bash
bash scripts/axon/setup_axon_ubuntu.sh
```

This script will:
1. Install required system packages (python3, v4l-utils, ffmpeg, libgl1, etc.)
2. Create `.venv` if it does not exist
3. Activate the virtual environment
4. Upgrade pip/setuptools/wheel
5. Install Python packages from `requirements/requirements-axon-cpu.txt`

The script is idempotent and safe to run multiple times.

## 9. Check AXON Environment

```bash
source .venv/bin/activate
bash scripts/axon/check_axon_env.sh
```

This checks:
- System info (kernel, CPU, OS)
- Python and pip availability
- Virtual environment status
- Package imports (numpy, cv2, mediapipe, onnxruntime)
- Camera devices under /dev/video*
- Disk and memory
- Required files (model, config, app)

## 10. List Cameras

```bash
bash scripts/axon/list_cameras.sh
```

Lists all /dev/video* devices and queries details via v4l2-ctl.

## 11. Probe Cameras

```bash
python apps/axon_camera_probe.py
```

Opens each camera index (0-5) with OpenCV, captures a test frame, and saves it.
Reports resolution, FPS, and frame shape for each working camera.

Optional arguments:
```bash
python apps/axon_camera_probe.py --max-index 3
python apps/axon_camera_probe.py --output-dir outputs/my_probe
```

## 12. Run Live Webcam with Display

Requires a connected monitor or X11 forwarding.

```bash
source .venv/bin/activate
bash scripts/axon/run_dms_webcam_axon.sh 0
```

This runs the DMS with:
- Camera index 0
- Debug overlay enabled
- Screenshot-style DualSight video, status, and vehicle-monitor windows
- A separate HMI header above the video; no status/performance panel obscures
  camera pixels
- Direct FaceMesh, head-pose axes, road/gaze vector, EAR eye state, PERCLOS,
  and vehicle gate
- Largest validated face inside the driver ROI selected as `DRIVER`; every
  other face shown as `PASSENGER` with a box only
- Phone/seat-belt model inference disabled for this test phase
- All output files written to `outputs/axon_webcam_live/`

The normal launcher records a feedback bundle until `q` is pressed:

- `webcam_output.mp4`: camera video with the complete DMS overlay;
- `webcam_state.jsonl`: serialized DMS state for every processed frame;
- `webcam_trace.jsonl`: detailed decision/input trace;
- `webcam_performance.jsonl`: capture, inference, latency, CPU/RAM, and backend
  measurements;
- `webcam_events.json` and `webcam_events.csv`: event summaries;
- `webcam_learning.jsonl`: learning-memory observations;
- `webcam_session.json`: valid JSON manifest with commit, configuration,
  output paths, final performance, frame count, and runtime error status.

Pass a distinct second argument to preserve multiple runs, for example:

```bash
bash scripts/axon/run_dms_webcam_axon.sh 0 outputs/axon_feedback_run01
```

`FAST_LIVE=1` intentionally disables this recording bundle.

The video window contains a dedicated instrument strip on the left of the
camera image. Head-pose and gaze graphics are not drawn over the driver's face.
The 3D axis convention is:

- red `X`: lateral head axis;
- green `Y`: vertical head axis;
- blue `Z`: depth/forward head axis.

Yaw, pitch, roll, and the current gaze zone are printed below the triad.

For lowest live-test latency while retaining all three consoles:

```bash
FAST_LIVE=1 bash scripts/axon/run_dms_webcam_axon.sh 0
```

The retained `old_baseline_coco_phone_detector` is never enabled implicitly.
Its ONNX/RKNN files use the RKNN Model Zoo multi-output DFL contract and remain
available through the preserved standalone phone tools. The integrated
cabin-evidence parser does not select that contract in this phase.

The integrated vehicle-test runtime currently prints and displays:

- compute backend: `CPU / MediaPipe XNNPACK`;
- NPU: `NOT ACTIVE`;
- NPU TOPS: `0.00 (inactive)`;
- capture-to-feature latency, model latency, FPS, CPU, and RAM.

`NPU TOPS` is not inferred from nominal model GOPS. It can report non-zero
utilization only after a selected RKNN runtime exposes a real measurement.

### Optional internal 106-point NPU evidence

The local InsightFace `2d106det` ONNX/RKNN weights are internal-PoC-only unless
separately licensed. They are disabled by default. To request driver-only RKNN
geometry evidence:

```bash
ENABLE_106_RKNN=1 FAST_LIVE=1 \
  bash scripts/axon/run_dms_webcam_axon.sh 0
```

The backend uses only the selected driver's face crop. Agreement with
MediaPipe eye geometry can raise the observation confidence; disagreement is
advisory and cannot override the accepted EAR path. If RKNNLite, the converted
model, or the `/dev/rknpu*` kernel device is unavailable, the runtime reports
the specific failure and continues with MediaPipe/EAR.

RKNNLite does not expose a live TOPS utilization counter in this integration.
When NPU inference is active, the console reports `NPU ACTIVE`, the measured
106-point inference latency, and `NPU TOPS: UNAVAILABLE` rather than inventing
a utilization value.

To use a different camera:
```bash
bash scripts/axon/run_dms_webcam_axon.sh 1
bash scripts/axon/run_dms_webcam_axon.sh 2 outputs/axon_cam2
```

## 13. Run Live Webcam Headless

For SSH sessions or when no display is available:

```bash
HEADLESS=1 bash scripts/axon/run_dms_webcam_axon.sh 0 outputs/axon_headless_test
```

Output files are still generated; only the display window is suppressed.

## 14. Run Video File with Display

```bash
bash scripts/axon/run_dms_video_axon.sh path/to/input.mp4 outputs/axon_video_test
```

## 15. Run Video File Headless

```bash
HEADLESS=1 bash scripts/axon/run_dms_video_axon.sh path/to/input.mp4 outputs/axon_video_headless
```

## 16. Expected Output Files

After a DMS run, the output directory will contain:

| File | Description |
|------|-------------|
| `*_output.mp4` | Video with debug overlay rendered |
| `*_state.jsonl` | Per-frame DMS state (JSON lines) |
| `*_trace.jsonl` | Debug trace log (JSON lines) |
| `*_events.csv` | DMS events in CSV format |
| `*_events.json` | DMS events in JSON format |
| `*_learning.jsonl` | Learning memory log |

## 17. Keyboard Controls (Display Mode)

When running with `--display`:

| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Calibrate road/head reference |
| `r` | Reset road/head calibration |
| `=` | Increase speed by 1 km/h |
| `+` | Increase speed by 5 km/h |
| `-` | Decrease speed |
| `9` | Left indicator toggle |
| `0` | Right indicator toggle |

## 18. Common Issues

### Camera index wrong
- Try different indices: 0, 1, 2
- Run `bash scripts/axon/list_cameras.sh` to see available devices
- Run `python apps/axon_camera_probe.py` to test each camera

### Permission denied on /dev/video*
```bash
sudo chmod 666 /dev/video0
# Or permanently:
sudo usermod -aG video $USER
# Then log out and back in
```

### OpenCV cannot open display
- Ensure a display is connected, or use X11 forwarding
- If running via SSH without display, use `HEADLESS=1`
- Install display packages: `sudo apt install libgtk-3-0`

### Missing libGL
```bash
sudo apt install libgl1 libglib2.0-0
```

### mediapipe import issue
- mediapipe may not have ARM64 wheels for all versions
- Try: `pip install mediapipe` (latest available)
- If unavailable, check GitHub releases for ARM64 builds
- Alternative: build from source (advanced)

### onnxruntime install issue on ARM64
- Try: `pip install onnxruntime`
- If unavailable via pip, try building from source
- Or use pre-built ARM64 wheel from ONNX Runtime releases

### Optional retained phone baseline
- ONNX: `models/mobile_phone_detector/yolov8n.onnx`
- RKNN: `models/mobile_phone_detector/yolov8n.rknn`
- Class map: `configs/dms/cabin_object_class_map_coco_phone.json`
- The ONNX and RKNN files are preserved for the standalone Model Zoo phone
  tools.
- A direct integrated smoke check currently reports
  `UNSUPPORTED_OUTPUT_SHAPE` because the cabin parser expects a different YOLO
  output layout. It therefore remains disabled instead of being misrepresented
  as integrated.

### Low FPS on CPU
- Reduce `frame_resize_width` in config (try 480 or 320)
- Use headless mode for better performance
- Close other applications
- Use `FAST_LIVE=1` to disable video/log recording while retaining the consoles.
- Do not substitute an RKNN model without task gates and ONNX/RKNN parity.

### Status window not visible in SSH
- Status window requires a display
- Use `HEADLESS=1` for SSH sessions
- Or use X11 forwarding: `ssh -X user@axon`

## 19. Performance Notes

- CPU mode on AXON will be slower than desktop x86 machines
- Expected FPS range: 5-15 FPS depending on resolution and face count
- The AXON config uses `frame_resize_width: 480` (reduced from 960)
- The mounted-driver profile uses direct FaceMesh; the wider-cabin
  proposal-plus-crop path remains available but is slower on AXON.
- MediaPipe Hands is disabled with the deferred phone phase.
- Further reduce to 480 or 320 if FPS is too low
- Use headless mode (`HEADLESS=1`) for slightly better performance
- Avoid running with `--display` over SSH (adds latency)
- Existing phone and landmark RKNN artefacts remain separate from the integrated
  runtime until their backend, licensing, accuracy, and parity gates are met.

## 20. Safety Notes

- This deployment does NOT change DMS detection logic or behavior
- This does NOT affect the final DMS state machine decisions
- This does NOT add any new detection capabilities
- This is NOT a production deployment - it is a development/evaluation runtime
- All behavioral parameters match the stable v0.2.9 baseline exactly
- No seatbelt heuristic logic is included

## 21. Future Work

The following items are planned for future branches (not part of this deployment):

- **Integrated RKNN face/landmark backend** - replace MediaPipe only after
  accuracy, licensing, ONNX/RKNN parity, and live latency validation
- **Accepted RKNN eye classifier** - only after the closed-eye and balanced
  accuracy gates are met; current trial models remain disabled
- **systemd service** - Auto-start DMS on boot
- **Camera calibration** - Intrinsic/extrinsic calibration for AXON camera setup
- **Thermal/FPS profiling** - Long-running thermal and performance analysis
- **Forward ADAS + DMS multi-camera** - Simultaneous forward and cabin camera
- **Production hardening** - Watchdog, auto-restart, error recovery
- **OTA updates** - Remote model and configuration updates
