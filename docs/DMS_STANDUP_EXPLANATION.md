# IND-VIAS DualSight DMS Standup Explanation

## 1. Executive Summary

We built a standalone IND-VIAS DualSight Driver Monitoring System (DMS) prototype for in-cabin video. The module processes either a video file or webcam feed, detects cabin faces, selects the configured driver occupant, estimates driver attention and drowsiness signals, and emits both visual debug output and per-frame JSONL state.

It was built standalone on purpose. The existing forward monocular ADAS perception engine remains independently runnable and unchanged. DMS currently lives under `src/ind_vias_dms`, with its own demo app in `apps/run_dms_demo.py`, so we can prototype cabin monitoring without merging it into the forward perception executable too early.

The current scope is a v0.1-series prototype, including v0.1.5 behavior in practice: multi-face occupant awareness, driver ROI selection, stable driver session tracking, road-gaze calibration, mobile-distraction heuristics, adaptive eye-state logic, time-based PERCLOS, debug overlay, separate status dashboard, and JSONL packet output. It is not yet a trained production DMS model or ASIL-ready safety component.

Outputs produced:

- Debug overlay video with face boxes, driver labels, landmarks, pose/gaze hints, and warning banner.
- Optional separate OpenCV status dashboard window.
- Per-frame JSONL DMS state packet.
- Basic driver presence, availability, gaze, drowsiness, distraction, phone-use, occupant, and readiness fields.

## 2. Folder Structure

### `apps/run_dms_demo.py`

This is the standalone DMS command-line entrypoint. It parses `--video` or `--camera`, loads `configs/dms/dualsight_dms_v0_1.yaml`, opens the video source, creates `DMSPipeline`, writes optional JSONL/output video, and handles display windows.

It also owns runtime road-gaze calibration hotkeys:

- `c`: calibrate current yaw/pitch as road center and optionally save calibration.
- `r`: reset road calibration to config defaults.
- `q`: quit display loop.

The app supports a separate status dashboard via `--status-window` and optional volatile track-ID display via `--show-track-id`.

### `configs/dms/dualsight_dms_v0_1.yaml`

This YAML file centralizes prototype tuning. It includes:

- Face backend and multi-face count.
- Frame resize/output FPS.
- Head-pose smoothing/outlier thresholds.
- Occupant layout, camera view, `driver_image_side`, and ROI generation.
- Driver session and re-association thresholds.
- Road-gaze calibration options and calibration file path.
- Eye baseline, PERCLOS, blink, drowsiness, and distraction thresholds.
- Mobile-distraction heuristic timings.
- Overlay/status-dashboard options.

This makes the prototype tuneable without rewriting code for each cabin camera setup.

### `src/ind_vias_dms/core`

Core contains the orchestration and typed state model:

- `pipeline.py`: coordinates the full per-frame DMS pipeline.
- `types.py`: stable enums and dataclasses for JSON output.
- `config.py`: YAML-to-dataclass config loader.
- `occupant_manager.py`: multi-face tracking, zone assignment, driver ROI selection, and duplicate-face suppression.
- `driver_session.py`: stable `driver_session_id` management across temporary face loss and short-term track changes.
- `driver_body.py`: lightweight body/seat continuity fallback during face loss.
- `road_calibration.py`: save/load road-gaze calibration YAML.
- `timing.py`: FPS utility.

This folder exists to keep business logic and state decisions separate from raw vision primitives.

### `src/ind_vias_dms/vision`

Vision contains frame-level perception modules:

- `face_landmarks.py`: MediaPipe Face Mesh backend with multi-face observations.
- `head_pose.py`: OpenCV `solvePnP` head-pose estimation and angle normalization.
- `eye_state.py`: EAR-like eye openness from landmarks.
- `gaze.py`: camera-calibrated gaze-zone heuristic.
- `phone_detection.py`: lightweight MediaPipe Hands/gaze mobile-distraction heuristic.
- `seatbelt.py`: placeholder for future seatbelt authenticity.
- `face_reid.py`: placeholder for future face re-identification.

This folder isolates perception backends so MediaPipe can later be replaced by ONNX/TIDL-compatible production models.

### `src/ind_vias_dms/temporal`

Temporal contains time-window and finite-state logic:

- `head_pose_smoother.py`: exponential moving average and outlier rejection for yaw/pitch/roll.
- `eye_temporal.py`: adaptive open-eye baseline, normalized eye openness, four eye states, blink/closure duration, and PERCLOS validity gating.
- `perclos.py`: time-based rolling PERCLOS tracker.
- `blink_tracker.py`: simple blink count/rate tracker.
- `drowsiness_fsm.py`: drowsiness level state machine with sustain gating.
- `distraction_fsm.py`: gaze/mobile distraction state machine with temporal thresholds.

This keeps frame measurements from flickering into safety states too quickly.

### `src/ind_vias_dms/interface`

Interface defines serialization boundaries:

- `dms_packet.py`: converts `DMSState` to dict/JSON.
- `adas_packet.py`: placeholder for future forward ADAS input packet.
- `fusion_packet.py`: placeholder for future fused ADAS+DMS risk packet.

This is the future integration boundary with the forward perception stack.

### `src/ind_vias_dms/visualization`

Visualization renders engineering debug output:

- `overlay.py`: video overlay, warning banner, occupant labels, pose/gaze vector clamping, embedded telemetry panel, and separate status dashboard.
- `colors.py`: status color helpers.

The overlay is intentionally a debug/HMI-development aid, not a final production cluster UI.

### `src/ind_vias_dms/utils`

Utilities contain small IO helpers:

- `video_io.py`: open webcam/video, resize frames, create output video writer.
- `jsonl_writer.py`: optional JSONL writer with output directory creation.

### `tests`

The DMS tests currently cover type serialization, enum stability, FSM transitions, gaze calibration behavior, duplicate-face suppression, occupant/driver ROI behavior, driver session re-association, status dashboard formatting, road calibration save/load, eye temporal behavior, PERCLOS timing, and key availability guardrails. Tests avoid webcam and MediaPipe requirements by using synthetic objects.

## 3. End-To-End Processing Flow

The runtime flow is:

```text
video/webcam input
-> optional resize/preprocessing
-> multi-face landmark detection
-> duplicate face suppression
-> occupant tracking and zone assignment
-> driver ROI selection
-> stable driver session update
-> driver body/seat continuity fallback
-> head pose estimation and smoothing
-> eye openness and adaptive eye-state update
-> blink and PERCLOS tracking
-> road-calibrated gaze-zone estimation
-> driver/cabin mobile-distraction estimation
-> drowsiness FSM
-> distraction FSM
-> driver availability and readiness scoring
-> overlay/status dashboard/JSONL output
```

The important design point is that drowsiness, gaze, distraction, and phone-use safety logic are applied only to the selected driver track/session. Passenger phone events can be reported as cabin events, but they do not directly become driver distraction.

## 4. Algorithms And Techniques Used

### MediaPipe Face Mesh / Face Landmark Backend

What it does: Detects one or more faces and returns dense face landmarks. The implementation converts those landmarks into pixel coordinates, face bounding boxes, normalized boxes, face centers, and approximate confidence.

Why selected for v0.1: It is lightweight, easy to install, real-time friendly on development machines, and provides enough landmarks to prototype head pose, eye openness, and occupant boxes without training a model.

Limitations: MediaPipe can fail on strong side profile, glare, occlusion, low light, and production camera domains. Confidence is approximate in this prototype.

Upgrade path: Replace with an ONNX/TIDL-compatible face detector and landmark model, ideally validated on cabin IR/RGB datasets and side-profile cases.

### OpenCV `solvePnP` For Head Pose

What it does: Uses selected face landmarks, a generic 3D face model, and camera intrinsics approximation to estimate yaw, pitch, and roll. The code normalizes angles and folds frontal ambiguity to avoid false +/-180 degree jumps.

Why selected: It is explainable, fast, and does not need training. It gives useful head orientation signals for early gaze and side-profile heuristics.

Limitations: Accuracy depends on landmark quality, assumed camera intrinsics, and face geometry. Strong profile or landmark drift can destabilize the estimate.

Upgrade path: Calibrated camera intrinsics, learned head-pose models, or a production landmark model with better profile robustness.

### Eye Openness / EAR-Like Logic

What it does: Computes eye aspect ratio style openness from MediaPipe eye landmarks. It uses vertical eye distances divided by horizontal eye distance.

Why selected: It is simple, transparent, and good enough for early blink/PERCLOS experiments.

Limitations: Sensitive to glasses, glare, partial eyelid occlusion, head pose, camera angle, and landmark noise.

Upgrade path: Dedicated eye-state classifier, IR eye-region model, or landmark refinement model trained on cabin conditions.

### Adaptive Eye Baseline

What it does: Learns a per-driver open-eye baseline from recent stable, high-confidence, mostly frontal frames. It then classifies `OPEN`, `PARTIALLY_CLOSED`, `CLOSED`, or `UNKNOWN` using normalized openness.

Why selected: A single fixed threshold is brittle across drivers, spectacles, lighting, and camera position. Adaptive normalization makes the prototype more tunable.

Limitations: Baseline can still be biased if initial frames are poor. The implementation guards against large yaw/pitch and low visibility, but more dataset validation is needed.

Upgrade path: Robust calibration protocol, confidence-aware eye model, or personalized baseline persistence.

### Time-Based PERCLOS

What it does: Tracks rolling time windows for eye closure percentage. It uses elapsed milliseconds, not frame count. `UNKNOWN` eye state and temporary driver loss pause accumulation instead of counting as open.

Why selected: PERCLOS is a widely understood drowsiness indicator and time-based tracking works better under variable FPS.

Limitations: It depends heavily on reliable eye-state classification and appropriate window/threshold tuning.

Upgrade path: Dataset-calibrated PERCLOS thresholds, eye-state classifier, and additional drowsiness cues such as yawn, nodding, and eyelid dynamics.

### Blink And Eye Closure Tracking

What it does: Tracks continuous closure duration and blink count/rate from eye states. Microsleep triggers from continuous closure duration.

Why selected: Blink rate and closure duration are explainable early signals for drowsiness.

Limitations: Blink detection is only as reliable as eye landmarks. Short occlusions or side profile can look like eye closure unless visibility gating catches them.

Upgrade path: Add blink event classifier and confidence-weighted temporal filtering.

### Gaze-Zone Heuristic

What it does: Classifies approximate gaze zones from smoothed head yaw/pitch relative to calibrated road-center offsets: `ROAD`, `LEFT`, `RIGHT`, `DOWN`, `UP`, `PHONE_DOWN`, or `UNKNOWN`.

Why selected: It gives useful first-order attention zones without claiming accurate eye-gaze vector estimation.

Limitations: It is head-pose-driven, not true gaze tracking. Camera mount and road-center calibration matter a lot.

Upgrade path: Add eye-gaze model, camera calibration, driver-specific calibration, and cabin camera mounting validation.

### Distraction FSM

What it does: Converts gaze-away duration and mobile-distraction states into `NONE`, `LOW`, `MEDIUM`, `HIGH`, or `UNKNOWN`. Temporal thresholds avoid making high distraction from one or two frames.

Why selected: It makes behavior explainable and threshold-tunable during testing.

Limitations: Heuristic thresholds can over- or under-trigger depending on camera, road context, and driver behavior.

Upgrade path: Train/validate against annotated attention datasets and integrate richer context from ADAS and cabin sensors.

### Drowsiness FSM

What it does: Uses PERCLOS, continuous closure duration, blink behavior, visibility confidence, and valid observation time to classify drowsiness. Medium/high levels require sustain time; microsleep remains immediate when closure duration exceeds threshold.

Why selected: It avoids flicker and makes warning behavior easier to debug.

Limitations: Still threshold-based and dependent on eye-state quality.

Upgrade path: Multi-signal temporal model combining eyelids, head nod, yawning, face orientation, and personalized baseline.

### Driver Availability Scoring

What it does: Combines face visibility, eye confidence, drowsiness, distraction, gaze, phone state, session state, and reason codes into `AVAILABLE`, `DEGRADED`, or `UNAVAILABLE`. Short gaze-away generally degrades before becoming unavailable; high distraction must be sustained.

Why selected: It separates warning/degraded operation from hard unavailable states.

Limitations: This is a heuristic readiness score, not a validated safety decision system.

Upgrade path: Calibrated safety logic, scenario validation, and integration with ADAS risk fusion.

### Occupant-Aware Driver ROI Logic

What it does: Detects multiple faces, suppresses duplicate boxes, assigns zones, and selects the driver using the configured image-space driver ROI. The config separates `vehicle_layout` from `driver_image_side` because a rear-facing dashboard camera in an Indian RHD car can show the driver on the left side of the image.

Why selected: It prevents passenger face/phone/drowsiness signals from being treated as driver state.

Limitations: ROI selection is geometry-based and still needs broader multi-occupant validation.

Upgrade path: Add seat occupancy, body pose, face re-ID, and camera-specific calibration profiles.

### Driver Session / Re-Association Logic

What it does: Separates volatile short-term `track_id` from stable `driver_session_id`. If FaceMesh temporarily loses the driver, the session can stay in `LOST_TEMP`, then re-associate a newly assigned track back to the same driver session.

Why selected: Production DMS needs stable identity across side profile, glare, and brief occlusion.

Limitations: Current re-association is deterministic geometry, not biometric face embedding.

Upgrade path: Add optional ONNX face embedding such as MobileFaceNet or ArcFace-small for driver swap confirmation.

### Mobile Distraction Prototype

What it does: Uses MediaPipe Hands when available and gaze context to detect hand-near-face, phone-to-ear suspicion, phone-down suspicion, and texting suspicion. It runs occupant-aware cabin logic so passenger phone events do not become driver distraction.

Why selected: It gives a lightweight prototype without adding YOLO or heavy object detection.

Limitations: It does not actually detect a phone object. It infers mobile use from hand/face/gaze patterns.

Upgrade path: Add lightweight phone object detection or hand-object interaction model.

### JSONL Packet Serialization

What it does: Serializes `DMSState` dataclasses and stable string enums to one JSON object per frame.

Why selected: JSONL is easy to inspect, replay, diff, and feed into future fusion experiments.

Limitations: It is not yet a final production CAN/SOME-IP/ROS interface.

Upgrade path: Convert to a versioned schema and align with future ADAS risk-fusion packet contracts.

### Overlay And Status Dashboard

What it does: Draws occupant boxes, driver label, landmarks, pose/gaze hints, warning banner, optional embedded telemetry panel, and a separate status dashboard.

Why selected: It accelerates debugging and threshold tuning during webcam/video testing.

Limitations: It is engineering visualization, not production HMI.

Upgrade path: Build a dedicated validation dashboard and production-grade HMI signals.

## 5. Why These Algorithms Were Chosen

The v0.1-series DMS uses lightweight, explainable algorithms because the immediate goal is to validate the pipeline shape and debugging interface, not to claim production-grade perception accuracy.

The approach is:

- Lightweight enough for real-time webcam/video testing.
- Explainable enough to debug in standup and field testing.
- No model training required for the first prototype.
- Easy to tune through YAML thresholds.
- Modular enough to replace MediaPipe/OpenCV heuristics later with ONNX/TIDL models.
- Safe for integration planning because it does not disturb the existing forward perception executable.

## 6. Current Configuration And Calibration Concept

`vehicle_layout`: Describes the vehicle driving layout, currently `RHD` for Indian right-hand-drive context.

`camera_mount_position`: Describes where the DMS camera is mounted. Current tested setup is `DASHBOARD_FRONT`.

`camera_view_direction`: Describes camera direction. Current tested setup is `CABIN_REARWARD`, meaning a front dashboard-mounted mobile rear camera points backward into the cabin.

`driver_image_side`: Explicitly maps the driver to image space, independent of RHD/LHD. In the tested RHD rear-facing dashboard video, the real driver appears on image-left, so `driver_image_side: LEFT`.

`mirror_input`: Allows image-side ROI interpretation to flip if the input is mirrored.

Driver ROI generation: When `auto_generate_rois_from_layout` is true, the code generates driver/front-passenger ROIs from `driver_image_side`. `LEFT` makes the driver ROI the image-left half and passenger ROI the image-right half. `RIGHT` does the opposite. Explicit ROI values are used only when auto-generation is disabled.

Road gaze calibration: Gaze is not assumed from raw camera-frontal pose. The system subtracts road-center yaw/pitch offsets from smoothed head pose. Pressing `c` during display calibrates current yaw/pitch as road center and can save it to `outputs/dms_road_calibration.yaml`. Startup can load that file and mark calibration source as `FILE`.

PERCLOS windows: The config uses short and long rolling windows, defaulting to 5 seconds and 60 seconds. They are time-based, not frame-count-based.

Eye thresholds: The prototype has both fixed `eye_closed_threshold` fallback and adaptive normalized thresholds for closed/partial eye states. Baseline update is gated by yaw, pitch, and eye visibility.

Distraction thresholds: Gaze-away and phone-suspicion durations determine `LOW`, `MEDIUM`, or `HIGH` distraction. High distraction must be sustained before driver availability becomes unavailable.

Drowsiness thresholds: PERCLOS thresholds determine medium/high candidates, but sustain timers reduce false warnings from short spikes. Microsleep still triggers immediately from continuous closure duration.

## 7. Output Interface

The DMS writes one JSON object per frame as JSONL. Important fields are:

`dms_health`: Camera/frame health and face detection health. `camera_status` means the frame stream is valid or errored. `face_detection_status` separately reports whether face detection found a face.

`driver_presence`: Driver face/session state, including `PRESENT`, `NOT_VISIBLE`, `LOST_TEMP`, `LOST_LONG`, `LOST`, `ABSENT`, or `UNKNOWN`.

`driver_availability`: Main driver availability decision: `AVAILABLE`, `DEGRADED`, `UNAVAILABLE`, or `UNKNOWN`, plus a score and reason codes such as `DRIVER_FACE_LOST_TEMP`, `ROAD_GAZE_NOT_CALIBRATED`, `EYE_VISIBILITY_LOW`, `GAZE_AWAY`, `DROWSINESS_MEDIUM`, or `MICROSLEEP`.

`occupants`: Cabin face count, driver track id for compatibility, driver body presence, and per-face track/zone metadata.

`driver_identity`: Stable driver-session packet. It includes `driver_session_id`, current short-term `driver_track_id`, `session_state`, `reassociated`, `time_since_seen_ms`, and `driver_body_state`.

`gaze`: Gaze zone, eyes-off-road duration, yaw/pitch/roll, confidence, and calibration source.

`drowsiness`: Drowsiness level plus detailed eye telemetry: eye state, raw openness, normalized openness, eye calibration state, eye visibility, PERCLOS values, valid observation times, closure duration, blink rate, and confidence.

`distraction`: Distraction level/type, duration, and confidence.

`phone_use`: Compatibility `state`, driver-specific `driver_state`, confidence, and occupant-aware `cabin_events`.

`seatbelt_authenticity`: Placeholder fields for future buckle/visual belt authenticity logic.

`driver_readiness_score`: Heuristic 0-to-1 score and risk level.

## 8. Current Test / Validation Status

Validation used three levels:

- Python compile validation with `compileall` over `src/ind_vias_dms` and `apps/run_dms_demo.py`.
- Unit tests with `pytest tests/test_dms_types.py tests/test_dms_fsm.py`.
- Webcam/video smoke testing through `apps/run_dms_demo.py`.

The known tested video path from the latest work is:

```powershell
.\.venv\Scripts\python.exe apps\run_dms_demo.py --video D:\Workspace\PoC\Codex\OnRoadData\20250602\dms4.mp4 --output outputs\dms_demo4.mp4 --jsonl outputs\dms_state4.jsonl --debug-overlay --display --status-window
```

Observed during testing:

- Driver ROI mapping for the dashboard-front, cabin-rearward RHD test video required `driver_image_side: LEFT`.
- Multi-face logic prevents passenger faces from automatically becoming driver state.
- Stable `driver_session_id` keeps identity continuity across short FaceMesh losses.
- Side-profile frames can still produce `LOST_TEMP`, but the driver session/body fallback keeps the state degraded rather than treating it as full cabin loss.
- Eye/PERCLOS behavior improved with adaptive baseline and time-based accumulation, but still needs real cabin dataset tuning.
- Road calibration file loading/saving exists and must continue to be validated over repeated runs.

## 9. Known Limitations / Open Issues

- Strong side-profile turns can still cause MediaPipe Face Mesh loss.
- Glasses, glare, low light, sunlight reflections, and partial face crop can affect eye landmarks.
- PERCLOS tuning still needs more cabin data and annotated drowsiness examples.
- Mobile phone detection is still heuristic/prototype and does not detect a phone object directly.
- Multi-occupant and driver selection require more validation across camera mounts, seat positions, and passengers.
- Road calibration persistence and calibration-source handling need careful validation in repeated demo runs.
- Seatbelt authenticity is a placeholder, not implemented vision logic.
- Face re-identification is a placeholder, not active biometric identity matching.
- The module is not production-grade, ASIL-ready, or validated for safety decisions.

## 10. Next Steps

- Improve side-profile fallback using body pose, shoulder/torso cues, or a profile-capable face backend.
- Improve adaptive eye baseline and PERCLOS robustness with more cabin datasets.
- Add a lightweight phone object or hand-object interaction model.
- Add seatbelt visual authenticity later.
- Collect and test more cabin videos: RHD/LHD, mirrored/unmirrored, day/night, glasses, glare, passengers, and phone-use scenarios.
- Define the versioned DMS packet contract for future ADAS risk fusion.
- Integrate DMS packet consumption into a future fusion layer without merging executables prematurely.
- Replace MediaPipe prototype backend with ONNX/TIDL-compatible face, landmark, eye, and phone models.

## 11. One-Minute Standup Script

Today I can summarize the DMS work as a standalone IND-VIAS DualSight prototype. We built a separate Python/OpenCV pipeline that reads webcam or cabin video, detects multiple faces, selects the configured driver occupant, tracks a stable driver session, estimates head pose, gaze, eye state, PERCLOS, drowsiness, distraction, mobile-use suspicion, availability, and readiness. It outputs a debug overlay video, a separate status dashboard, and per-frame JSONL.

The key reason it is standalone is to avoid disturbing the existing forward ADAS perception pipeline while we validate cabin monitoring. The latest version is occupant-aware and camera-layout-aware, so in our Indian RHD dashboard-front rear-facing setup we explicitly set `driver_image_side: LEFT`. It also keeps `driver_session_id` stable across short FaceMesh losses and pauses PERCLOS during unknown eye states rather than treating them as open eyes.

Current limitations are side-profile face loss, glasses/glare effects on eye landmarks, heuristic phone detection, and the need for more PERCLOS and multi-occupant validation. Next step is to harden side-profile/body fallback, improve eye/PERCLOS robustness, and prepare the DMS packet for future ADAS fusion.

## 12. Five-Minute Technical Explanation

The DMS was built as a separate module under `src/ind_vias_dms`, with `apps/run_dms_demo.py` as its executable. The app loads a YAML config, opens a webcam or video, resizes frames, passes each frame to `DMSPipeline`, and writes three outputs: optional overlay video, optional display/status windows, and JSONL state.

Inside the pipeline, the first stage is MediaPipe Face Mesh. It can detect up to the configured number of faces, currently four. Each detected face becomes a `FaceLandmarkResult` with landmarks, bounding box, normalized box, center, and area. Before tracking, duplicate face boxes are suppressed so one driver face does not become multiple occupants.

Next, `CabinOccupantManager` assigns track IDs and cabin zones. Driver selection is based on the configured driver image-side ROI. This is important because RHD/LHD alone does not determine image side: our phone rear camera is mounted in front and points rearward, so the RHD driver appears on the left side of the image. The config therefore separates `vehicle_layout` from `driver_image_side`.

After driver selection, `DriverSessionManager` maintains a stable `driver_session_id`. The short-term `track_id` can change when FaceMesh loses and reacquires the face, but the stable session can remain `D1`. During temporary loss, the system reports `LOST_TEMP`, keeps body continuity if the driver-seat ROI is still held, pauses PERCLOS, and marks availability as `DEGRADED` rather than immediately `UNAVAILABLE`.

For the selected driver, head pose is estimated using OpenCV `solvePnP` from common face landmarks. Head-pose angles are normalized and smoothed with an exponential moving average. Gaze is a heuristic zone classifier using smoothed yaw/pitch relative to road calibration offsets. Road calibration can be loaded from file or set at runtime with the `c` hotkey.

Eye state starts with EAR-like landmark openness, then a temporal tracker builds an adaptive open-eye baseline from stable frontal frames. The output eye states are `OPEN`, `PARTIALLY_CLOSED`, `CLOSED`, and `UNKNOWN`. PERCLOS is time-based: it accumulates elapsed milliseconds where the driver eyes are closed, with optional partial-closure weighting. Unknown eye state and temporary driver loss pause accumulation instead of counting as open.

Drowsiness and distraction are then decided by state machines. Drowsiness uses PERCLOS, closure duration, blink behavior, confidence, and sustain timers. Distraction uses gaze-away duration and mobile-use suspicion. Mobile detection is currently heuristic: MediaPipe Hands, when available, is combined with gaze and hand proximity to face/lower cabin regions. Passenger phone events can be reported as cabin events but do not become driver distraction.

Finally, the pipeline produces `DMSState`, which includes health, driver presence, occupant metadata, driver identity/session state, gaze, drowsiness, distraction, phone use, seatbelt placeholder, availability, and readiness score. The interface serializes that to JSONL, while the overlay renders either embedded telemetry or a separate dashboard for debugging.

## 13. Q&A Preparation

### Why MediaPipe?

Because it gives a fast landmark-based prototype without training data or model integration work. It is suitable for early pipeline validation, debugging, and threshold tuning. It is not the intended production backend; the module is structured so we can replace it later with ONNX/TIDL-compatible models.

### Why `solvePnP`?

`solvePnP` is a lightweight, explainable way to estimate head pose from 2D landmarks and a simple 3D face model. It is enough for early yaw/pitch/roll and gaze-zone heuristics. Its limitation is that it depends on landmark quality and approximate camera intrinsics.

### Why PERCLOS?

PERCLOS is a widely used drowsiness indicator based on the percentage of time eyes are closed. We implemented it time-based rather than frame-count-based so it behaves better when FPS varies. It still needs tuning against real cabin data.

### Why standalone DMS?

To protect the existing forward ADAS pipeline while we prototype cabin monitoring. DMS can evolve independently, produce JSONL packets, and later integrate through a fusion interface.

### Why is `driver_image_side` separate from RHD/LHD?

Because vehicle layout does not always map directly to image side. In the current Indian RHD test, the camera is mounted at the dashboard/front and points rearward into the cabin, so the real driver appears on the left side of the image. `driver_image_side` captures that image-space reality.

### How does driver selection work?

The system detects multiple faces, suppresses duplicates, assigns zones, and selects the driver using overlap with the configured driver ROI. It prioritizes driver ROI overlap, temporal continuity, confidence, area, and distance from ROI center.

### How does the system handle no face?

If no driver face is found but the driver session was recently active, the session becomes `LOST_TEMP`. Eye/PERCLOS accumulation pauses, gaze becomes unknown, body fallback can keep driver body present, and availability becomes degraded. If the loss exceeds configured timeouts, it can become `LOST_LONG` or unavailable.

### Why does side profile become DMS degraded?

Strong side profile can make FaceMesh fail even when the driver is still seated. The safe prototype behavior is to degrade DMS confidence, preserve the driver session temporarily, and avoid computing drowsiness or gaze from another occupant.

### How will this integrate with forward ADAS later?

DMS already emits a structured JSONL packet and has placeholder ADAS/fusion packet classes. A future fusion layer can consume forward ADAS risk and DMS readiness/distraction/drowsiness signals without merging the executables prematurely.

### Is this production-ready?

No. It is a working engineering prototype. It is useful for architecture validation, field debugging, and threshold exploration, but production needs trained/validated models, larger cabin datasets, calibrated cameras, robust side-profile/eye/mobile detection, safety analysis, and ASIL-oriented validation.
