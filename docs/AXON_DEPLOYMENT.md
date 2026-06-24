# AXON Deployment Guide - DualSight DMS v0.2.9

## 1. Purpose

This document provides step-by-step instructions to deploy and run the
DualSight DMS v0.2.9 on the Vicharak AXON board running Ubuntu 22.04 ARM64.

This is a **Stage 1 CPU/OpenCV runtime deployment** only. The goal is to get
the existing stable DMS running on AXON hardware with a USB webcam or video
file input using CPU-based inference.

## 2. Target Board and OS

- **Board:** Vicharak AXON
- **OS:** Ubuntu 22.04 (ARM64/aarch64)
- **Runtime:** CPU/OpenCV (no GPU/NPU acceleration in this stage)
- **Camera:** USB webcam via /dev/video* or video file input

> If you are running a different Ubuntu version, some package names may need
> minor adjustments (e.g., `libgl1` vs `libgl1-mesa-glx`).

## 3. Branch Name

```
feature/axon-runtime-v029
```

Based on commit: `6576c94 Stabilize DualSight DMS v0.2.9 simplified cabin phone logic`

## 4. What This Branch Changes

This branch **adds new files only** for AXON deployment:

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

## 5. What This Branch Does NOT Change

- No modifications to `src/ind_vias_perception/` (forward ADAS)
- No modifications to `src/ind_vias_dms/` (DMS core logic)
- No modifications to `apps/run_dms_demo.py`
- No modifications to existing configs or tests
- No changes to DMS FSM behavior
- No changes to phone ROI/gating logic
- No changes to phone scenario logic
- No seatbelt heuristic logic
- No RKNN/NPU acceleration implementation

## 6. Clone Repo on AXON

```bash
cd ~
git clone <your-repo-url> ind-vias-perception-engine-v2
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
- Display window and status window visible
- Cabin ONNX evidence (if model file exists)
- All output files written to `outputs/axon_webcam_live/`

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

### ONNX model missing
- Place cabin object detection model at: `models/dms/cabin_objects.onnx`
- Place class map at: `configs/dms/cabin_object_class_map_coco_phone.json`
- Without the model, the system runs with dummy cabin evidence backend

### Low FPS on CPU
- Reduce `frame_resize_width` in config (try 480 or 320)
- Use headless mode for better performance
- Close other applications
- RKNN/NPU acceleration will be added in a future branch

### Status window not visible in SSH
- Status window requires a display
- Use `HEADLESS=1` for SSH sessions
- Or use X11 forwarding: `ssh -X user@axon`

## 19. Performance Notes

- CPU mode on AXON will be slower than desktop x86 machines
- Expected FPS range: 5-15 FPS depending on resolution and face count
- The AXON config uses `frame_resize_width: 640` (reduced from 960)
- Further reduce to 480 or 320 if FPS is too low
- Use headless mode (`HEADLESS=1`) for slightly better performance
- Avoid running with `--display` over SSH (adds latency)
- RKNN/NPU acceleration will be a separate future branch for hardware speedup

## 20. Safety Notes

- This deployment does NOT change DMS detection logic or behavior
- This does NOT affect the final DMS state machine decisions
- This does NOT add any new detection capabilities
- This is NOT a production deployment - it is a development/evaluation runtime
- All behavioral parameters match the stable v0.2.9 baseline exactly
- No seatbelt heuristic logic is included

## 21. Future Work

The following items are planned for future branches (not part of this deployment):

- **RKNN model conversion** - Convert ONNX models to RKNN format for NPU
- **NPU inference backend** - Hardware-accelerated inference on AXON NPU
- **systemd service** - Auto-start DMS on boot
- **Camera calibration** - Intrinsic/extrinsic calibration for AXON camera setup
- **Thermal/FPS profiling** - Long-running thermal and performance analysis
- **Forward ADAS + DMS multi-camera** - Simultaneous forward and cabin camera
- **Production hardening** - Watchdog, auto-restart, error recovery
- **OTA updates** - Remote model and configuration updates
