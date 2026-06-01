# IND-VIAS DualSight DMS v0.1 Architecture

IND-VIAS DualSight DMS v0.1 is a standalone Driver Monitoring System prototype. It reads an in-cabin camera stream, estimates face presence, face landmarks, head pose, eye openness, PERCLOS, gaze zone, distraction, drowsiness, availability, and a simple driver readiness score.

The DMS implementation is intentionally isolated under `src/ind_vias_dms`. It does not merge with the existing forward monocular ADAS perception executable, and it does not change the forward perception pipeline. The standalone demo entrypoint is `apps/run_dms_demo.py`.

## Pipeline

`DMSPipeline` coordinates the v0.1 modules:

- `vision`: MediaPipe Face Mesh landmarks, OpenCV `solvePnP` head pose, EAR-style eye openness, heuristic gaze, and phone/seatbelt placeholders.
- `temporal`: blink tracking, time-based PERCLOS, drowsiness FSM, and distraction FSM.
- `interface`: DMS JSONL serialization plus future ADAS/fusion packet placeholders.
- `visualization`: debug overlay with face box, landmarks, head axis, gaze hint, telemetry, and warnings.
- `utils`: video source handling and JSONL writing.

## v0.1.1 Stabilization

The v0.1.1 update normalizes head-pose Euler angles, smooths yaw/pitch/roll with a lightweight exponential moving average, and treats low-confidence or outlier pose as unknown before gaze classification. `PHONE_DOWN` requires sustained downward gaze instead of a single frame. Driver availability is also duration-gated so short distraction spikes degrade the state before they can make the driver unavailable.

The debug video overlay can now omit the embedded telemetry panel and show a separate OpenCV status dashboard when `--display --status-window` is used.

## v0.1.2 Mobile And Calibration Notes

The v0.1.2 update adds lightweight mobile-distraction heuristics. When MediaPipe Hands is available, hand proximity to the ear, cheek, mouth, or lower cabin region is combined with gaze state to flag phone-to-ear, phone-down, texting, or hand-near-face suspicion. If hand tracking is unavailable, the DMS still falls back to gaze-only phone-down suspicion.

ROAD gaze is calibrated relative to the DMS camera mount. In a production vehicle, looking into an inward-facing camera near the IRVM is not automatically the same as looking at the road. During a displayed demo, press `c` to use the current smoothed yaw/pitch as the road center and `r` to reset to the config defaults. Calibration is in-memory only and does not rewrite the YAML file.

## Future Integration

Future versions can publish the DMS packet into the existing ADAS risk stack through a fusion layer. v0.1 deliberately keeps this as a documented interface boundary only: forward perception remains independently runnable, while DMS produces its own JSONL state stream for later integration.
