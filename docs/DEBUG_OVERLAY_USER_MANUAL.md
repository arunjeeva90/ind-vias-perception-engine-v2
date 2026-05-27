# Debug Overlay User Manual

The debug overlay is an engineering visualization for the IND-VIAS Metric Monocular ADAS Perception Engine v2. It is not a final HMI. It exposes internal perception, tracking, distance, TTC, cut-in, SafetyGate, CAIS, and Sentinel state so the frozen pipeline can be validated without bypassing the architecture:

```text
frame -> ONNXDetectionHead/dummy detection -> MetricMonocularPipeline -> tracking -> cut-in lateral TTC -> TTC/SafetyGate -> CAIS/Sentinel -> visualization
```

## Example overlay label

```text
car 0.86 id:2 miss:0 pred:False 1.9m side:LEFT cut:NONE cv:False
```

- `car`: detected class name. Common values include `car`, `truck`, `pedestrian`, `motorcycle`, and `two_wheeler_agent`.
- `0.86`: detector confidence score.
- `id:2`: tracker ID for the object.
- `miss:0`: number of consecutive frames the tracker has carried this object without a detector match.
- `pred:False`: whether this is a predicted-only track. `False` means the current frame had a matched detection.
- `1.9m`: selected displayed object distance. When vehicle offset is configured, safety logic uses bumper-relative distance.
- `side:LEFT`: side state relative to the ego corridor. Values are `LEFT`, `RIGHT`, `IN`, or `UNKNOWN`.
- `cut:NONE`: cut-in state. Values are `NONE`, `LEFT_CUT_IN`, `RIGHT_CUT_IN`, or `IN_PATH`.
- `cv:False`: `cutin_valid_for_safety`. `False` means the cut-in signal is debug-only and must not generate `cut_in_risk`.

Newer debug builds may also show compact cut-in fields:

- `ce`: `corridor_entry_confirmed`, true only when corridor overlap grows over multiple frames.
- `ov`: current `corridor_overlap_ratio`.
- `dov`: `corridor_overlap_delta`, the change in corridor overlap over track history.
- `lat`: lateral TTC in seconds.
- `elig`: `cutin_warning_eligible`, the final cut-in warning eligibility before SafetyGate confirmation.
- `cross`: deterministic pedestrian/two-wheeler crossing state, such as `left_to_right`, `right_to_left`, `parallel`, or `uncertain`.
- `xconf`: crossing confidence.
- `xvalid`: `crossing_valid_for_safety`; false means the crossing label is debug-only or suppressed by validity gates.

## Confidence score

The confidence score comes from `ONNXDetectionHead` detector output. For the YOLOv8 COCO demo, it is the maximum class probability after ONNX output decoding. It is filtered by `detection.confidence_threshold` from config. It is not the same as distance confidence, target relevance, cut-in confidence, or safety confidence.

## Side State

- `IN`: object ground contact is in or overlapping the ego corridor.
- `LEFT`: object is left of the ego corridor.
- `RIGHT`: object is right of the ego corridor.
- `UNKNOWN`: insufficient geometry or context.

Side state is based on bbox ground-contact and ego-corridor geometry. It is not full lane-level semantic understanding yet.

## Cut-In State

- `NONE`: no lateral cut-in behavior detected.
- `LEFT_CUT_IN`: object appears to enter the ego path from the left.
- `RIGHT_CUT_IN`: object appears to enter the ego path from the right.
- `IN_PATH`: object is already in the ego path; longitudinal TTC/FCW should handle it, not lateral cut-in warning.

Cut-in uses tracking history, lateral pixel movement, lateral TTC, distance validity, target relevance, corridor overlap, ego motion, and safety validity checks. Low-relevance side objects need a clear crossing trend before they can become warning-eligible.

Cut-in warning eligibility also requires progressive corridor entry (`ce:True`), stable smoothed lateral motion, and a sane lateral TTC range. Very tiny lateral TTC values are treated as jitter-like evidence and suppressed. Crossing labels are separately gated by VRU class, history, displacement, corridor approach, distance, boundary, object size, and confidence.

## cv

`cv` means `cutin_valid_for_safety`.

- `cv:True`: cut-in passed safety validity checks and can be considered by SafetyGate.
- `cv:False`: cut-in information is shown for debug only and must not generate `cut_in_risk`.

Typical false reason codes include `low_relevance_no_crossing_trend`, `insufficient_corridor_entry`, `invalid_distance_for_safety`, `ego_not_straight`, `near_image_boundary`, and `insufficient_lateral_history`.

## Threshold / Config Mapping

| Overlay field | Meaning | Main config key | Typical current value | Effect if increased/decreased |
|---|---|---:|---:|---|
| confidence score | Detector class confidence | `detection.confidence_threshold` | `0.25` | Higher removes weak detections; lower shows more detections and more noise. |
| boxes after NMS | Duplicate suppression | `detection.nms_threshold` | `0.45` | Higher keeps more overlapping boxes; lower suppresses more boxes. |
| `cut_conf` / `cv` | Minimum cut-in confidence for warning | `cutin.min_confidence_for_warning` | `0.75` | Higher makes cut-in rarer; lower allows weaker lateral evidence. |
| `rel` / `cv` | Minimum target relevance for warning | `cutin.min_relevance_for_warning` | `0.45` | Higher suppresses side objects; lower allows more low-relevance cut-ins. |
| `overlap` / `cv` | Required corridor entry overlap for low-relevance cut-ins | `cutin.min_corridor_overlap_for_warning` | `0.15` | Higher requires clearer entry into corridor; lower admits earlier edge overlap. |
| `ttc_lat` | Lateral TTC threshold | `cutin.lateral_ttc_threshold_s` | `2.8` | Higher warns earlier; lower warns later and less often. |
| distance | Maximum cut-in-relevant distance | `cutin.max_relevant_distance_m` | `22.0` | Higher considers farther side objects; lower focuses on nearby objects. |
| `hist` | Minimum lateral history count | `cutin.min_history` | `5` | Higher requires more stable tracks; lower reacts sooner but noisier. |
| `ego_motion` | Yaw score threshold | `ego_motion.yaw_score_threshold` | `0.65` | Higher makes turning detection less sensitive; lower marks turning more often. |
| `confirm` | Required turning frames | `ego_motion.required_turning_frames` | `3` | Higher resists yaw spikes; lower detects turning sooner. |
| warning confirmation | Warning confirmation frames | `safety_confirmation.required_frames.warning` | `2` | Higher delays warnings; lower confirms sooner. |
| strong warning confirmation | Strong warning confirmation frames | `safety_confirmation.required_frames.strong_warning` | `3` | Higher reduces strong-warning flicker; lower is more responsive. |
| AEB ready confirmation | AEB-ready confirmation frames | `safety_confirmation.required_frames.aeb_ready` | `3` | Higher makes AEB readiness harder to assert; lower confirms sooner. |

## How To Interpret Warning Lines

- `warning`: final confirmed warning shown by the overlay.
- `raw_warning_level`: immediate SafetyGate candidate before confirmation.
- `confirmed_warning_level`: warning after multi-frame confirmation.
- `warning_candidate`: the candidate currently being counted by confirmation.
- `confirmation_count / confirmation_required`: current count versus frames required.
- `aeb_ready`: true only when the confirmed warning meets AEB-ready criteria.
- `warning_suppressed_reason`: why a warning was suppressed, such as invalid distance or no valid safety target.

## How To Interpret Distance Lines

- `Dg`: ground-plane distance.
- `Ds`: semantic-size distance.
- `Df`: fused camera-relative distance.
- `Dbump`: bumper-relative distance after subtracting configured camera-to-front-bumper offset.
- `confD`: distance confidence.
- `rel`: target relevance for safety selection.
- `valid`: `distance_valid_for_safety`.
- `reason`: distance quality reason codes.

## How To Interpret CAIS / Sentinel / Ego Motion

- `cais_mode`: compute adaptation mode: `nominal`, `enhanced`, or `critical`.
- `cais_score`: CAIS internal escalation score.
- `cais_reason_codes`: why CAIS selected its mode.
- `sentinel_state`: Sentinel FSM safety state.
- `ego_motion_state`: `straight`, `turning`, or `uncertain`.
- `yaw_confidence`: confidence in ego yaw/turning state.
- `turning_confirmation_count`: number of consecutive frames supporting turning.

## What This Overlay Does NOT Mean

- It is not a certified ADAS output.
- It is not calibrated production distance yet.
- The COCO YOLO detector is temporary.
- Side state is not full lane-level understanding yet.
- Debug values are for engineering validation only.

## Recommended Demo Interpretation

Describe the overlay this way: green boxes are detected and tracked agents; the yellow corridor is an ego-path approximation; distance is bumper-relative where configured; cut-in risk should be rare; `IN_PATH` means the object is already in the longitudinal risk path; `cv:False` means do not trust the cut-in signal for a safety decision.
