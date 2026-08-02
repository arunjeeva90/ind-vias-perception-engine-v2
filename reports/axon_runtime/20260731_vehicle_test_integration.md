# AXON DualSight vehicle-test integration

Date: 2026-07-31
Branch: `feature/axon-runtime-v029`
Status: **CODE READY; PHYSICAL CAMERA/DISPLAY VALIDATION PENDING**

## Repository comparison

- fetched remote:
  `origin/feature/axon-runtime-v029 = 8be25b53f6526d72f244c70356e6fe9935fabe87`;
- local HEAD:
  `2ca322de07fb3e1c51139410d17240f51f1252e3`;
- local HEAD is one commit ahead:
  `2ca322de Add standalone EyeNet webcam demo`;
- the requested DualSight windows, vehicle state manager, speed gate, status
  dashboard, head-pose estimator, road-axis calibration, eye EAR/PERCLOS path,
  phone evidence hooks, and vehicle monitor already existed locally;
- the AXON masking was primarily in configuration and launch selection:
  vehicle monitor, status, head axes, and gaze vector were disabled, while the
  launch script could silently enable a phone ONNX model if a file appeared.

## Restored AXON vehicle-test behavior

`configs/dms/dualsight_dms_axon.yaml` now enables:

- `IND-VIAS DualSight DMS - Video`;
- `IND-VIAS DualSight DMS - Status`;
- `IND-VIAS Vehicle Monitor`;
- head-pose axes;
- road/gaze vector;
- the existing simulated vehicle-speed gate:
  - startup ramp: 0 to 25 km/h;
  - DMS activation: strictly above 30 km/h;
  - DMS deactivation: below 28 km/h;
  - standby suppresses DMS alerts.

The `vehicle-test` window layout requests:

- video: 960 x 720 at (10, 20);
- status: 720 x 1000 at (1180, 20);
- vehicle monitor: 780 x 390 at (415, 650).

Window-manager borders and display scaling may shift the final outer dimensions
slightly.

## Live status content

The status and vehicle consoles now use a high-contrast card layout at the
AXON vehicle-test window sizes. Long values are clipped to their cards instead
of being drawn outside the visible right edge. The video HMI occupies a
dedicated 46-pixel header above the camera image, and performance telemetry is
never drawn on camera pixels.

The status and vehicle canvases are rendered internally at 1.25x resolution
and downsampled by their fixed AXON windows for clearer anti-aliased text.
Benchmarks rejected a 2x canvas because it added roughly 75 ms across both
panels; 1.25x preserves the clarity improvement with about 8 ms additional
panel-rendering cost. Head & Road is now a full-width card, so both Head Angle
and Raw/Relative values are visible without ellipsis.

The RGB solvePnP head-pose triad has moved to a dedicated instrument strip on
the left of the camera image. No head-pose or gaze vector covers driver face
pixels. Its legend is red/X lateral, green/Y vertical, and blue/Z depth; yaw,
pitch, roll, and gaze zone are displayed below it.

The status dashboard includes:

- camera/face health and active face backend;
- compute backend, explicit NPU active state, NPU TOPS state, feature/model
  latency, FPS, CPU, and RAM;
- NIR/input/threshold state;
- driver-face/proposal/track state;
- eye runtime source, Eye CNN status, and optional 106-geometry status;
- raw/effective eye state, openness, visibility, closure, PERCLOS, drowsiness;
- head raw/relative pose, gaze, road calibration/offset;
- vehicle gate/speed/indicators;
- cabin backend/phone/belt/smoking state;
- final HMI banner.

The full existing telemetry list and JSONL serialization are preserved.

For the current integrated path the truthful readout is:

- compute backend: `CPU / MediaPipe XNNPACK`;
- NPU: `NOT ACTIVE`;
- NPU TOPS: `0.00 (inactive)`.

The retained RKNN artefacts are not treated as active merely because they
exist. The GOPS-based workload estimate remains separate from NPU utilization.

## Driver/passenger role policy

The AXON mounted-camera profile detects up to four faces and applies this
policy:

1. validate face observations;
2. consider only faces inside the driver ROI for driver selection;
3. select the largest image-space face as the nearest/front `DRIVER`;
4. label every other face `PASSENGER`, including a smaller face inside the
   driver ROI and any face outside it;
5. retain passenger boxes, but discard passenger landmarks immediately after
   validation so passenger faces cannot feed driver eye, head-pose, gaze, or
   temporal DMS decisions.

Passenger candidates retain the existing confidence, minimum-area, landmark,
and temporal-confirmation checks. Static-headrest size rejection is disabled
for the direct-FaceMesh AXON profile because FaceMesh itself supplies the human
face validation and a distant passenger must not be removed solely for having
a small box.

## Eye/head runtime selection

Enabled:

- direct MediaPipe FaceMesh for the mounted single-driver profile;
- existing head-pose estimation and smoothing;
- head axes and road-relative gaze vector;
- existing EAR eye openness/closure;
- bilateral/temporal eye tracking, blink suppression, PERCLOS, and sustained
  closure/microsleep logic;
- runtime road-axis calibration (`c`) and reset (`r`).

Intentionally disabled:

- reviewed-data Eye CNN: did not meet acceptance gates;
- optional 106-point geometry in the integrated runtime: internal PoC/licensing
  restriction and no accepted integrated backend;
- reviewed-data seat-belt classifier: not trained/accepted;
- reviewed-data cabin-specific phone detector: not trained/accepted.
- MediaPipe Hands phone-posture inference: deferred with the phone phase; its
  implementation is retained.

The status window explicitly reports `LANDMARK_EAR`, `Eye CNN: DISABLED`, and
`106 geometry: DISABLED` instead of implying NPU/model use.

## MediaPipe AXON startup correction

MediaPipe 0.10.18 imports the whole Tasks API from its root package. That
transitively imported unused audio/sound-device support and blocked DMS startup
in this AXON environment.

`src/ind_vias_dms/utils/mediapipe_loader.py` now loads the Solutions API while
providing a process-local minimal Tasks namespace. The DMS face mesh, face
detection, and hands APIs are unchanged. The installed MediaPipe package is not
modified.

Observed smoke behavior:

- before: initialization exceeded the 60-second guard;
- after: three-frame synthetic headless DMS run completed successfully in
  approximately 2.7 seconds.

## AXON frame-path optimization

A bounded comparison used 15 frames from
`WIN_20260728_11_36_11_Pro.mp4`, classified only as
`RUNTIME_REGRESSION_VIDEO`:

| Profile | Mean inference | Approx. inference FPS | Face present |
|---|---:|---:|---:|
| proposal + crop FaceMesh + Hands | 276.0 ms | 3.6 | 15/15 |
| proposal + crop FaceMesh, no Hands | 200.0 ms | 5.0 | 15/15 |
| direct FaceMesh + Hands | 121.4 ms | 8.2 | 15/15 |
| direct FaceMesh, no Hands | 30.3 ms | 33.0 | 15/15 |

The AXON head/eye vehicle profile therefore selects direct FaceMesh and disables
phone Hands in configuration. It does not delete either implementation.

A separate 60-frame integrated regression run after selection measured:

- mean DMS inference: 27.33 ms;
- median DMS inference: 21.26 ms;
- mean full loop including decode/overlay/telemetry: 43.48 ms;
- final processing rate: 23.45 FPS;
- maximum observed process RAM: 240.4 MiB.

Before the profile change, the same bounded integrated run measured 285.90 ms
mean inference, 317.77 ms mean loop time, and 3.14 FPS. These are runtime
integration measurements, not independent eye-state accuracy claims.

## Phone baseline retained as standalone

The existing baseline remains named:
`old_baseline_coco_phone_detector`.

Preserved hashes:

| Artifact | SHA-256 |
|---|---|
| `models/mobile_phone_detector/yolov8n.onnx` | `0c8716701f471067932b797eeb67c8e5db47c693c2557c881d7679ec12e21bc5` |
| `models/mobile_phone_detector/yolov8n.rknn` | `9f7a3a37158d19a252e7133ce38b8fcd809ae3cf894919fb6df9560eaa558bb5` |

It is no longer enabled merely because files exist. The default AXON
head-pose/eye vehicle run uses the dummy cabin backend.

An explicit one-frame integrated ONNX smoke check was performed. The model
loaded, but its nine-output RKNN Model Zoo DFL contract produced
`UNSUPPORTED_OUTPUT_SHAPE` in the integrated cabin parser. The baseline remains
ready in its preserved standalone ONNX/RKNN scripts, but it is not represented
as integrated. Adding that DFL parser is deferred with the phone phase.

No phone training was started and no latest phone handoff images were used in
this phase.

## RKNN/NPU audit

Existing converted artefacts:

| Task/artifact | RKNN SHA-256 | Integrated selection |
|---|---|---|
| COCO phone baseline `yolov8n.rknn` | `9f7a3a37158d19a252e7133ce38b8fcd809ae3cf894919fb6df9560eaa558bb5` | standalone tool only; phone phase deferred |
| 68-point landmark `landmark_rk3588.rknn` | `b9bc93d72013164b4ff19b6cd8c9eb89532b0ebdfb5e961f9a63e42e3e2fe211` | standalone PoC; integrated postprocess is not implemented |
| 106-point landmark `landmark_106_rk3588.rknn` | `e2b0ff23d9496f8f7895b90f3efa0ae1b10609e1343afdfb2700e5371cc854b6` | standalone internal PoC; licensing/integration gate |
| old EyeNet `eyenetrknn_mnv3s_96_int8.rknn` | `799f9e8041d2fa1db069b754af4a174b49c95ecb1b5735f13b5876aa4b601dfa` | not accepted for current eye runtime |

No new RKNN was produced because the only new candidate eye models failed the
mandatory crop gates. The generic RKNN face-landmark replacement still raises
`NotImplementedError` for its unknown model-specific postprocessing, so it
cannot replace MediaPipe. The known 106-point direct-coordinate model is
handled separately below as optional confidence evidence, not as the primary
face backend.

### Driver-only 106-point confidence adapter

The local InsightFace `2d106det` contract was re-audited:

- ONNX input: dynamic batch, `3x192x192`;
- ONNX output: `1x212`, decoded as 106 coordinate pairs;
- ONNX SHA-256:
  `f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf`;
- RKNN SHA-256:
  `e2b0ff23d9496f8f7895b90f3efa0ae1b10609e1343afdfb2700e5371cc854b6`.

A driver-only RKNNLite adapter now implements the known loose-face crop,
RGB/NHWC input, 212-value output selection, coordinate decoding, inference
latency, lifecycle release, and explicit runtime failure states. It runs
independently of the disabled eye CNN. Geometry agreement may increase EAR
confidence, while disagreement is advisory until validation is completed.

The ONNX adapter returned 106 finite points on the local face snapshot. A
separate available local regression frame produced geometry disagreement with
MediaPipe, correctly leaving EAR confidence unchanged.

NPU execution could not be validated in this session: RKNNLite loaded the
converted model, but `init_runtime()` failed because `/dev/rknpu*` and the
`rknpu` kernel driver were unavailable. The integrated opt-in path then
continued successfully on MediaPipe/EAR and reported
`NPU_RUNTIME_UNAVAILABLE`. Default operation remains unchanged.

The 106 weights remain internal-PoC-only due to the InsightFace pretrained
model license. Enable them explicitly with:

```bash
ENABLE_106_RKNN=1 FAST_LIVE=1 \
  bash scripts/axon/run_dms_webcam_axon.sh 0
```

RKNNLite provides measured inference latency but no live TOPS utilization
counter. Active status must therefore show `NPU TOPS: UNAVAILABLE`, never an
estimated or nominal TOPS value.

## Verification

- Python compilation: passed;
- AXON launcher shell syntax: passed;
- dashboard/window/profile tests: passed;
- synthetic video, headless integrated pipeline: passed;
- 60-frame approved `RUNTIME_REGRESSION_VIDEO` integration run: passed at
  23.45 FPS final processing rate;
- premium status/vehicle/video renders inspected at their native AXON sizes;
- bounded 10-frame headless integration smoke on the existing local webcam
  output: passed; final sample reported 16.08 processing FPS, 75.03 ms model
  latency, 84.33 ms capture-to-feature latency, CPU backend, and NPU inactive;
- full test suite: **421 passed**;
- three-frame feedback-bundle smoke: valid overlay MP4 with three frames,
  three state rows, three performance rows, and a valid completed
  `webcam_session.json`;
- baseline phone hashes: unchanged;
- no raw prohibited video was opened or processed;
- no commit or push was performed.

The previously used external USB regression video was not mounted for the
latest UI/role change. The bounded local smoke above is an integration check,
not an independent accuracy or generalisation claim. Physical multi-person
camera behavior, three-window window-manager scaling, mounted-camera
calibration, low-light/glasses behavior, thermal stability, and sustained live
FPS remain on-board validation steps.

## Vehicle command

Recommended first physical run, preserving all three consoles but disabling
video/log recording:

```bash
FAST_LIVE=1 bash scripts/axon/run_dms_webcam_axon.sh 0
```

Full evidence-recording run:

```bash
bash scripts/axon/run_dms_webcam_axon.sh 0 outputs/axon_vehicle_test
```

Controls:

- `q`: quit;
- `c`: calibrate the road/head reference while looking straight ahead;
- `r`: reset road calibration;
- `=`: +1 km/h;
- `+`: +5 km/h;
- `-`: -1 km/h;
- `9`: toggle left indicator;
- `0`: toggle right indicator.

Start below 30 km/h, confirm `STANDBY` and suppressed alerts, press `+` twice
from the 25 km/h ramp target to reach 35 km/h, and confirm `DMS ACTIVATED`
followed by `DMS ACTIVE MONITORING`.
