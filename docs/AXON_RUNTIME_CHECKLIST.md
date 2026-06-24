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
- [ ] Display window visible (if not headless)
- [ ] Output logs generated in `outputs/axon_webcam_live/`
- [ ] CPU/FPS observed (check terminal output)
- [ ] Headless run verified (`HEADLESS=1 bash scripts/axon/run_dms_webcam_axon.sh 0 outputs/axon_headless_test`)

## Output Verification

- [ ] `*_output.mp4` file created
- [ ] `*_state.jsonl` file created
- [ ] `*_trace.jsonl` file created
- [ ] `*_events.csv` file created
- [ ] `*_events.json` file created
- [ ] `*_learning.jsonl` file created

## Notes

- Cabin ONNX evidence requires `models/dms/cabin_objects.onnx` (optional)
- Expected FPS on CPU: 5-15 FPS at 640px width
- Use `HEADLESS=1` for SSH sessions without display
- Reduce `frame_resize_width` in config if FPS is too low
