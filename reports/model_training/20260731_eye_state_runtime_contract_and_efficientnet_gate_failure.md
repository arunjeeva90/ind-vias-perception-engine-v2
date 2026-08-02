# Eye-state runtime contract correction and EfficientNet-Lite0 gate result

Date: 2026-07-31
Status: **STOPPED AT CROP GATE — NOT DEPLOYABLE**

## Decision

The direct 96x96 crop-classification approach is the correct basic design, but
the live crop must have the same geometric and photometric contract as the
training crop. It is not an exact-image/template comparison. The classifier
learns a decision function over similar patterns; it still needs:

1. consistent eye localization;
2. corner-based roll alignment;
3. a scale-normalized square context window;
4. the exact resize, colour order, and normalization;
5. crop-quality abstention;
6. independent landmark-geometry agreement;
7. per-eye inference and conservative bilateral/temporal fusion.

The previous runtime violated item 2 and item 3. It used an axis-aligned
rectangle, while the authoritative reviewed images use a corner-aligned square
whose native side is `1.65 * eye_width`, shifted upward by
`0.10 * eye_width`, reflect-padded, and resized without changing aspect ratio.

That contract is now implemented, but remains disabled because neither trained
classifier passed the crop acceptance gates.

## Authoritative data use

Only this root was used:

`/home/vicharak/Mobility_ADAS/ADVIS/DMS/DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/01_Eye_state_dataset`

The current folder contains:

- `closed`: 445 reviewed images;
- `open`: 4,684 reviewed images;
- `uncertain`: 19 reviewed images.

All 5,129 reviewed `closed` and `open` images were used directly in the
deterministic split:

- train: 375 closed / 3,738 open = 4,113;
- validation: 70 closed / 946 open = 1,016.

The 19 `uncertain` images are the only eye images under this root not used for
binary training. Every one has `recommended_for_training=false` in the final
reviewed manifest. No accepted closed/open crop was silently omitted.

Integrity/provenance:

- prepared manifest SHA-256:
  `510f29a06fbb8dd90c852636b3476e3b6fc83e2d92c36e819ce550002cb6ac4c`;
- split SHA-256:
  `160a2a0e20ba04e9ca0da0facfb57b68595b26854e40bc43088bbfb8374ac61b`;
- final reviewed label manifest SHA-256:
  `439806fe43abe06f9d7a5062b6c6bf93a18af99caffd2bee97359829bf82dca3`;
- exact train/validation hash overlap: 0;
- source-video train/validation overlap: 0;
- validation description:
  **source-video-exclusive held-out reviewed crop validation**.

No raw `dmsNit207.mp4` or `mob_belt.mp4` processing was performed. Their
already-reviewed derivative crops remained in the accepted split as directed.

## 106-point landmark audit and integration

Local artefacts audited:

- `models/dms/landmark_106.onnx`
  - SHA-256:
    `f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf`;
  - input: NCHW float, `3x192x192`;
  - output: 212 values / 106 2D points.
- `models/dms/landmark_106_rk3588.rknn`
  - SHA-256:
    `e2b0ff23d9496f8f7895b90f3efa0ae1b10609e1343afdfb2700e5371cc854b6`.

The upstream InsightFace inference contract was reproduced:

- loose square face crop at `1.5 * max(face_width, face_height)`;
- RGB order;
- raw 0..255 input for this graph because its first nodes contain embedded
  subtraction/multiplication normalization;
- output decoded from -1..1 and inverse-affine-mapped to the source frame.

Two defects were found in the old PoC eye mapping:

- image-left eye used `33..40`, omitting valid eye points 41 and 42;
- image-right eye mixed eyebrow points 98, 99, and 100 into the crop while
  omitting valid eye points.

The corrected groups are:

- image-left eye: `33..42`;
- image-right eye: `87..96`;
- image-left eyebrow: `43..51`;
- image-right eyebrow: `97..105`.

The old fixed 42-pixel crop was also replaced in the reusable path by the
scale/roll-normalized reviewed-crop contract.

The 106-point model is integrated only as optional independent geometry
evidence:

- it does not decide open versus closed;
- it checks agreement between its eye corners and MediaPipe eye corners;
- disagreement yields `UNKNOWN` when the optional backend is explicitly
  enabled;
- weights and backend remain disabled by default;
- the pretrained InsightFace weights are internal-PoC-only unless a suitable
  commercial/OEM licence is separately obtained.

Existing performance evidence:

- RK3588 RKNN landmark inference mean: approximately 2.50 ms over the existing
  14,106-row live log;
- local ONNX CPU reference: mean 10.876 ms, median 9.252 ms, p95 18.677 ms over
  30 inferences;
- the old YuNet CPU face detector, not landmark inference, dominates the
  existing PoC latency.

## Runtime fusion implemented

The optional classifier path now performs:

1. reviewed-contract aligned crop per eye;
2. minimum eye-width, padding, blur, darkness, and overexposure checks;
3. per-eye CNN inference;
4. CNN/EAR disagreement -> per-eye `UNKNOWN`;
5. bilateral disagreement -> frame `UNKNOWN`;
6. single valid eye -> confidence penalty;
7. bilateral agreement -> conservative minimum confidence.

Configuration remains `enabled: false`. The legacy axis-aligned path can be
selected explicitly for diagnostics, but is not the configured contract.

## EfficientNet-Lite0 trials

Both trials used:

- ImageNet-pretrained `timm/tf_efficientnet_lite0.in1k`;
- pretrained weights SHA-256:
  `0aa007d2f73be7b75909ac7aba43c229a8ff9a8c5a0e27bfd53faac31dc25381`;
- 96x96 RGB input, mean/std 0.5/0.5;
- weighted cross-entropy with square-root inverse-frequency weights;
- weighted balanced sampling;
- realistic training-only affine, colour, blur, and sensor-noise augmentation;
- frozen classifier-head stage followed by final-two-block fine-tuning;
- early checkpoint selection on closed-eye F1 plus balanced accuracy;
- no validation augmentation;
- no deployment export.

### Trial A — original orientation

Path:

`models/dms_classifiers/eye_state_efficientnet_lite0/efflite0_original_weighted_balanced_seed20260730`

Best selected operating point:

| Metric | Result | Gate | Pass |
|---|---:|---:|---|
| Closed precision | 0.4915 | >= 0.75 | No |
| Closed recall | 0.8286 | >= 0.85 | No |
| Closed F1 | 0.6170 | >= 0.80 | No |
| Balanced accuracy | 0.8826 | >= 0.85 | Yes |
| Exact-hash overlap | 0 | 0 | Yes |
| Source-video overlap | 0 | 0 | Yes |

No threshold passed all gates. Among thresholds satisfying recall >= 0.85 and
balanced accuracy >= 0.85, the maximum precision was 0.4196.

Selected checkpoint SHA-256:

`42cea1ae6acfe40e72176cf9ea2088739e6cd75706f30133f9d76448a882b5f5`

### Trial B — canonical orientation (mirror reviewed right-eye crops)

Path:

`models/dms_classifiers/eye_state_efficientnet_lite0/efflite0_mirror_right_weighted_balanced_seed20260730`

Best selected operating point:

| Metric | Result | Gate | Pass |
|---|---:|---:|---|
| Closed precision | 0.4351 | >= 0.75 | No |
| Closed recall | 0.9571 | >= 0.85 | Yes |
| Closed F1 | 0.5982 | >= 0.80 | No |
| Balanced accuracy | 0.9326 | >= 0.85 | Yes |
| Exact-hash overlap | 0 | 0 | Yes |
| Source-video overlap | 0 | 0 | Yes |

No threshold passed all gates. Among thresholds satisfying recall >= 0.85 and
balanced accuracy >= 0.85, the maximum precision was 0.4453.

Selected checkpoint SHA-256:

`758dae0dc65c50eaf226164c07209eed795f2c31b64c95e29f2c289239c1769f`

### Side diagnosis

Original-orientation Trial A:

- image-left closed: precision 0.7213 / recall 0.8462 / F1 0.7788;
- image-right closed: precision 0.2456 / recall 0.7778 / F1 0.3733.

Canonical Trial B increased right-eye recall to 1.0 but did not separate open
hard cases:

- image-left closed: precision 0.6712 / recall 0.9423 / F1 0.7840;
- image-right closed: precision 0.2222 / recall 1.0000 / F1 0.3636.

This is evidence against simply training longer or lowering the threshold. The
remaining problem is class separation for image-right/open hard cases, not a
lack of closed-eye sensitivity.

## Gate consequence

Because both EfficientNet trials failed:

- no ONNX was exported;
- no RKNN was generated;
- no runtime configuration was enabled;
- none of the four permitted videos was processed;
- no raw prohibited video was accessed;
- seat-belt and phone training did not begin;
- no commit or push was performed.

## Recommended next approach

Do not run another architecture blindly. The next useful work should be:

1. visually audit the repeated image-right false positives listed in
   `validation_errors.csv`, grouped by source frame, glasses/reflection,
   occlusion, blur, and crop centring;
2. preserve final human labels, but add non-label diagnostic tags so error
   modes can be measured;
3. audit left/right crop pairs from the same frame and train/evaluate a
   bilateral pair model or a shared per-eye encoder with a pair-level fusion
   head; one-eye binary training discards useful agreement context;
4. add an explicit `UNKNOWN`/quality head only if the authoritative uncertain
   set is expanded enough for source-exclusive validation; do not force
   ambiguous appearance into open/closed;
5. use the corrected 106/MediaPipe geometry agreement and temporal state
   machine only after a crop classifier passes the independent crop gate;
6. obtain more reviewed, source-diverse closed-eye and hard-open right-eye
   crops from future permitted collection. Do not mix any legacy data.

The most efficient production architecture is therefore:

`face -> independent landmark agreement -> aligned per-eye crops -> quality
gates -> shared eye encoder -> bilateral fusion -> UNKNOWN/open/closed evidence
-> temporal blink/closure/PERCLOS state machine`

Landmarks improve localization and confidence; they should not be counted as an
independent second vote for eye state when both landmark systems observe the
same pixels.

## Verification

- focused backend/source-policy tests: 112 passed;
- full suite excluding the known native `v0241` hard-termination cases:
  403 passed, 8 deselected;
- `git diff --check`: clean.
