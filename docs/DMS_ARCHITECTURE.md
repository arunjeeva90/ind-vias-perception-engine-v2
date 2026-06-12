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

## v0.1.3 Occupant Awareness

The v0.1.3 update adds a cabin occupant manager. MediaPipe Face Mesh is configured for multiple faces, each observation is assigned a lightweight track id and cabin zone, and the DMS pipeline selects the driver using the configured driver ROI for RHD/LHD layouts. Drowsiness, gaze, availability, and driver mobile distraction are computed only for the selected driver track. Passenger phone events can appear as cabin events, but they do not directly become driver distraction.

If faces are visible but none overlaps the driver ROI, driver presence is reported as `NOT_VISIBLE`, occupant count still reflects the cabin faces, and driver safety logic pauses/resets instead of computing PERCLOS on a passenger face.

Non-driver occupants now use a two-stage path: raw face proposals are first treated as possible faces, then promoted to confirmed occupants only after confidence, face size, landmark sanity, ROI plausibility, and temporal persistence checks pass. The driver ROI remains more tolerant for safety-critical driver monitoring, while front-passenger and rear-seat false positives such as headrests stay as unconfirmed proposals and do not inflate `occupants.face_count`.

## v0.1.6 Attention-State Layer

The v0.1.6-style update adds a temporal attention classifier in `src/ind_vias_dms/temporal/attention_state.py`. It runs after driver selection, gaze, eye/PERCLOS, phone, drowsiness, and distraction estimation, but before final availability scoring and JSONL output.

The final decision layer now uses `src/ind_vias_dms/temporal/dms_evidence_aggregator.py` to separate observation quality from behavioral evidence. Camera/face/eye/pose quality can degrade confidence, but it does not erase accumulated behavior such as head-down, gaze-off-road, or phone-down posture. The aggregator maintains hysteresis-style durations for head-down, uncertain head-down, gaze-off-road, phone-down candidate, visual distraction, and degraded observation.

The classifier first treats head-down or off-road behavior as an attention-loss candidate. It then disambiguates the likely substate:

- phone or visual distraction when gaze/head pose is down or off-road, eyes are open or intermittent, and phone/hand/lap evidence exists
- drowsiness or microsleep when reliable eye visibility shows sustained closure, elevated PERCLOS, and no strong phone explanation
- ambiguous attention loss when eye visibility, face geometry, or side-profile tracking is too weak to separate phone posture from sleep
- face-lost degraded mode when the driver session and body continuity are held but FaceMesh is temporarily unavailable

This prevents a phone-looking posture from automatically inflating drowsiness, and it prevents side-profile FaceMesh loss from becoming high distraction or drowsiness by itself. Future ADAS fusion should consume `driver_availability` and the attention state/substate as the primary DMS risk signal, with gaze, PERCLOS, phone, and drowsiness as supporting evidence.

Sustained down-looking posture can become `PHONE_DOWN_SUSPECTED` using temporal posture evidence even when hand landmarks are unavailable. The warning remains conservative: short mirror checks stay normal or low, phone-like attention loss becomes a distraction warning and degraded availability, and unavailable is reserved for microsleep, prolonged no-driver visibility, or long sustained phone/attention loss.

Banner priority is handled as a final decision, not as a side effect of individual vision modules: unavailable and microsleep remain highest priority, drowsiness warning follows, then distraction warning, then DMS degraded, then normal. The overlay renderer applies a short banner hold so the UI does not flicker between normal and degraded on adjacent frames.

## v0.2 Standalone Decision Matrix

The v0.2 target is standalone DMS stabilization, not forward-ADAS fusion. `src/ind_vias_dms/temporal/dms_v02_decision.py` keeps four independent heads: drowsiness state, distraction state, driver availability state, and DMS confidence state. The final standalone banner is derived from these heads using the v0.2 hierarchy: `NORMAL`, `MONITOR`, `WARNING`, `DANGER`, `CRITICAL`, and `DEGRADED`.

`src/ind_vias_dms/core/occupancy.py` adds cabin occupancy output without changing driver identity or driver temporal state. It reports driver, front passenger, rear-left, rear-center, rear-right, and unknown occupancy using confirmed faces and stable partial rear proposals. Driver ROI priority remains true, and occupancy never replaces D1, resets PERCLOS, or creates distraction warnings by itself.

This task intentionally does not implement v0.3 ADAS risk fusion, TTC-aware escalation, CAN-FD fusion contracts, or L2/L2+ driver-aware fusion. Those remain future integration work after the standalone DMS state machine is stable.

## Future Integration

Future versions can publish the DMS packet into the existing ADAS risk stack through a fusion layer. v0.1 deliberately keeps this as a documented interface boundary only: forward perception remains independently runnable, while DMS produces its own JSONL state stream for later integration.
