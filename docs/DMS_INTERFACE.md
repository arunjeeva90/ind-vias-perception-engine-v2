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
- `phone_use`
- `seatbelt_authenticity`
- `driver_readiness_score`

`phone_use` and `seatbelt_authenticity` are placeholders in v0.1 and return `UNKNOWN` with confidence `0.0`.

In v0.1.2, `phone_use.state` may also report `NO_PHONE`, `PHONE_TO_EAR_SUSPECTED`, `PHONE_DOWN_SUSPECTED`, `TEXTING_SUSPECTED`, or `HAND_NEAR_FACE`. These are heuristic states, not object-detection claims.

`gaze.zone` is camera-mount calibrated. A raw frontal face relative to the DMS camera is classified as `ROAD` only when it falls within the configured road yaw/pitch offsets and tolerances.

## Future ADAS Input Packet

`ADASInputPacket` is a placeholder for forward perception risk data, including timestamp, forward risk level, and reason codes. It is not consumed by v0.1.

## Future Fused Output Packet

`FusionPacket` is a placeholder for later DMS plus ADAS risk fusion. It can carry a fused risk level and reason codes from both subsystems. v0.1 does not alter the forward ADAS executable or emit fused control decisions.
