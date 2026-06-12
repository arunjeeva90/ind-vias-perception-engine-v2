# IND-VIAS DualSight DMS v0.1 Interface

The standalone DMS demo writes one JSON object per frame as JSONL. Enum values are stable strings so downstream consumers can parse packets without importing Python classes.

## DMS JSONL Packet

Top-level fields:

- `timestamp_ms`
- `frame_id`
- `dms_health`
- `driver_presence`
- `driver_availability`
- `gaze`
- `drowsiness`
- `distraction`
- `attention`
- `dms_v02`
- `occupancy`
- `phone_use`
- `seatbelt_authenticity`
- `driver_readiness_score`

`phone_use` and `seatbelt_authenticity` are placeholders in v0.1 and return `UNKNOWN` with confidence `0.0`.

In v0.1.2, `phone_use.state` may also report `NO_PHONE`, `PHONE_TO_EAR_SUSPECTED`, `PHONE_DOWN_SUSPECTED`, `TEXTING_SUSPECTED`, or `HAND_NEAR_FACE`. These are heuristic states, not object-detection claims.

`gaze.zone` is camera-mount calibrated. A raw frontal face relative to the DMS camera is classified as `ROAD` only when it falls within the configured road yaw/pitch offsets and tolerances.

## Attention State

The attention layer is a temporal classifier that runs after face selection, gaze, eye/PERCLOS, phone, drowsiness, and distraction estimation. It does not treat a head-down pose as automatically equal to phone use or drowsiness. Instead it first marks an attention-loss candidate and then disambiguates the likely substate.

`attention` fields:

- `attention_state`: `NORMAL`, `ATTENTION_LOST`, `DEGRADED`, or `UNKNOWN`
- `attention_substate`: `ROAD`, `VISUAL_DISTRACTION`, `PHONE_SUSPECTED`, `PHONE_DOWN_SUSPECTED`, `PHONE_TO_EAR_SUSPECTED`, `TEXTING_SUSPECTED`, `PHONE_CONFIRMED`, `DROWSY`, `MICROSLEEP`, `AMBIGUOUS`, `FACE_LOST`, or `UNKNOWN`
- `attention_confidence`
- `head_down_duration_ms`
- `gaze_offroad_duration_ms`
- `eye_closed_duration_ms`
- `attention_lost_duration_ms`
- `side_profile_lost_duration_ms`
- `microsleep_candidate`
- `phone_suspicion_candidate`
- `ambiguous_attention_loss`
- `low_head_motion`
- `attention_reason_codes`
- `driver_availability_reason`
- `phone_down_candidate_duration_ms`
- `visual_distraction_duration_ms`
- `observation_degraded_duration_ms`
- `final_decision_path`

Head-down plus low eye visibility is usually reported as ambiguous or phone-suspected attention loss unless reliable sustained eye closure supports drowsiness. This keeps phone-looking posture from inflating PERCLOS-based drowsiness. ADAS fusion should consume `driver_availability` and `attention` first, then use phone/drowsiness/distraction labels as supporting evidence.

Sustained phone-looking posture can set `phone_use.driver_state` to `PHONE_DOWN_SUSPECTED` even when MediaPipe Hands is unavailable. This is still a heuristic suspicion, not object-detection confirmation.

PERCLOS, phone, and attention reason codes use separate namespaces. PERCLOS validity reasons describe only eye-observation validity, while phone and attention reasons can describe posture such as `POSSIBLE_PHONE_POSTURE`, `HEAD_DOWN`, or `GAZE_OFF_ROAD`.

## v0.2 Decision Heads

`dms_v02` contains the standalone v0.2 decision matrix outputs:

- `drowsiness_state`: `NONE`, `EARLY_DROWSY`, `DROWSY`, `MICROSLEEP`, or `UNKNOWN`
- `distraction_state`: `NONE`, `MONITOR`, `VISUAL`, `MANUAL`, `PHONE_SUSPECTED`, `PHONE_CONFIRMED`, `COMBINED`, or `UNKNOWN`
- `driver_availability_state`: `AVAILABLE`, `PARTIALLY_AVAILABLE`, `DEGRADED`, `UNAVAILABLE`, or `UNCONFIRMED`
- `dms_confidence_state`: `HIGH`, `MEDIUM`, `LOW`, or `UNAVAILABLE`
- `final_level`: `NORMAL`, `MONITOR`, `WARNING`, `DANGER`, `CRITICAL`, or `DEGRADED`
- `final_banner`
- `final_decision_path`

v0.2 is standalone DMS only. It does not include final ADAS risk fusion, TTC-aware escalation, CAN-FD fusion contracts, or L2/L2+ driver-aware fusion.

## v0.2.1 Observability and Profile Stability

v0.2.1 separates driver observability from driver availability:

- `driver_observability.state`: `OBSERVABLE`, `PARTIALLY_OBSERVABLE`, `UNOBSERVABLE_TEMP`, `UNOBSERVABLE_LONG`, or `UNKNOWN`
- Temporary side-profile/FaceMesh loss with driver session continuity or driver body presence is reported as degraded observability, not immediate `DRIVER UNAVAILABLE`.
- `DRIVER UNAVAILABLE` is reserved for sustained absence/body loss, microsleep or critical drowsiness, or long critical attention loss.

`dms_health` also exposes the active imaging and threshold profile:

- `nir_mode_detected`
- `input_color_mode`
- `active_eye_threshold_profile`
- `active_perclos_profile`
- `nir_preprocessing_active`
- `nir_reason_codes`

Daylight BGR input should report `BGR_DAY`; grayscale/NIR-like input should report `NIR_NIGHT`.

## Occupants

v0.1.3 adds an `occupants` object with cabin face count, selected driver track id, driver zone, and per-face track metadata. `driver_presence.state` can now be `NOT_VISIBLE` when other occupants are visible but no face is selected inside the driver ROI.

`phone_use` keeps the compatibility `state` field and adds `driver_state` plus `cabin_events`. Driver distraction uses only `driver_state`; passenger phone-to-ear observations are reported as cabin events such as `PASSENGER_PHONE_TO_EAR`.

`occupants.face_count` and `occupants.confirmed_face_count` count confirmed human faces only. Raw detector boxes are exposed separately as `occupants.proposal_count`, with `occupants.unconfirmed_proposal_count` and optional `occupants.rejected_proposals` reason codes for debugging false positives such as headrests or short-lived proposal boxes. Non-driver faces require stronger confidence, landmark sanity, ROI plausibility, and temporal persistence than the driver face.

`occupancy` adds seat-level cabin awareness for driver, front passenger, rear-left, rear-center, and rear-right seats. Each seat reports occupancy state, detection source, confidence, track id, stable frame count, occlusion state, face/body visibility, and normalized bbox when available. Occupancy is an additional output and does not override driver identity, PERCLOS, attention timers, or driver availability.

## Future ADAS Input Packet

`ADASInputPacket` is a placeholder for forward perception risk data, including timestamp, forward risk level, and reason codes. It is not consumed by v0.1.

## Future Fused Output Packet

`FusionPacket` is a placeholder for later DMS plus ADAS risk fusion. It can carry a fused risk level and reason codes from both subsystems. v0.1 does not alter the forward ADAS executable or emit fused control decisions.
