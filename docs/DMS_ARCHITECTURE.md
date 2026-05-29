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

## Future Integration

Future versions can publish the DMS packet into the existing ADAS risk stack through a fusion layer. v0.1 deliberately keeps this as a documented interface boundary only: forward perception remains independently runnable, while DMS produces its own JSONL state stream for later integration.
