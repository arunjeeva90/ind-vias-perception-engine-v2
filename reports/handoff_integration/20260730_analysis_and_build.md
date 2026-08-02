# DMS handoff analysis and initial build — 2026-07-30

## Outcome

The final-reviewed handoff has been integrated as a reproducible, optional
model path without replacing the existing DualSight architecture. The source
handoff remains unchanged. Eye and seat-belt ONNX classifiers are disabled by
default and fail safely to EAR/landmark behavior or `UNKNOWN`. The existing
cabin-object detector already supports the reviewed one-class YOLOv8
channel-first tensor, so phone integration required only an explicit class map
and a regression test, not a new parser.

The generated one-epoch eye artifact is a plumbing smoke test only. Its
balanced accuracy is 0.50 and it is not configured or accepted for runtime.

## Repository and source audit

- Branch: `feature/axon-runtime-v029`
- HEAD before edits: `2ca322de07fb3e1c51139410d17240f51f1252e3`
- Local-only commit: `2ca322de Add standalone EyeNet webcam demo`
- Staged files/stashes/deletions before edits: none
- Pre-existing `.gitignore`, dataset, model, result, test-image, mobile-phone,
  and RKNN script work was preserved.
- The required fetch could not update `.git/FETCH_HEAD` because `.git` is
  mounted read-only. Cached remote-tracking state showed ahead 1, behind 0.
- Detailed preflight: `reports/handoff_preflight/20260730_preflight.md`

## Handoff verification

The task prompt, `README_FIRST.md`, final dataset card, training/integration
specification, package summary, validation report, task manifests, existing
runtime modules, EyeNet tooling, AXON configuration, and AXON deployment
documents were reviewed.

Direct Linux verification of `SHA256SUMS.txt` is affected by its CRLF
filenames. CR-stripped verification found 7,848 valid entries and one mismatch:

```text
01_IMAGES/03_Phone_detection/object_detection_yolo/images/train/
WIN_20260728_11_33_25_Pro_t0000005000_f0000150.jpg
```

Expected SHA-256:
`e462f6ef3e3e01137b322847b86ddbb9ce80bc851334de306b0d1f842db67721`

Actual SHA-256:
`6fa96089a6ddd518f211eaa896c08fe2a1d129a11e0f44530b961585718b2c3c`

The deterministic preparer records and excludes this frame. Its empty label
remains untouched. Upstream provenance should be reconciled before restoring
it.

## Existing implementation versus handoff

| Area | Existing repository | Handoff evidence | Applied result |
|---|---|---|---|
| Eye | EAR estimator plus EyeNet tooling whose defaults declared 5 outputs and whose RKNN runtime hard-coded `bad_crop`, `eye_closed`, `eye_open` | Reviewed data is binary: 445 closed, 4,684 open; no reviewed bad-crop class | Binary metadata-driven model contract, class weighting, optional two-eye ONNX fusion, EAR fallback retained |
| Seat belt | `SeatbeltDetectionPlaceholder` always returned unknown | Reviewed 224x224 two-class crops and a tested face-to-torso geometry helper | Compatibility class retained but now performs quality-gated torso classification, low-confidence unknown, and temporal confirmation |
| Phone | Existing phone posture logic plus generic cabin ONNX parser/fusion | One-class YOLO and hard-negative classifier crops | Existing parser retained; class 0 maps to `PHONE`; parser regression test added; ROI/fusion contracts unchanged |
| Dataset prep | No authoritative handoff-manifest consumer | Final task manifests, reviewed flags, source videos, dimensions, hashes, empty labels | Strict deterministic verifier/splitter/materializer with provenance metadata |
| AXON export | EyeNet ONNX/RKNN scripts with hard-coded class assumptions and non-deterministic calibration seed | Static ONNX first, training-only calibration, metadata-driven class order | Opset-12 static export, checkpoint-derived class map, deterministic training-only calibration list |

The handoff Python files are extraction, curation, review, validation, and
dataset-building utilities. They are not production runtime models. The
seat-belt torso geometry was reused; existing DMS state packets, temporal
systems, phone evidence fusion, and overlays were preserved.

## Deterministic prepared data

Preparation seed: `20260730`

Prepared manifest SHA-256:
`510f29a06fbb8dd90c852636b3476e3b6fc83e2d92c36e819ce550002cb6ac4c`

| Task | Train | Validation |
|---|---:|---:|
| Eye closed | 375 | 70 |
| Eye open | 3,738 | 946 |
| Seat belt off | 364 | 54 |
| Seat belt on | 407 | 178 |
| Phone classifier hard negative | 358 | 90 |
| Phone classifier positive | 192 | 40 |
| YOLO hard negative | 80 | 20 |
| YOLO positive | 167 | 65 |

The phone detector contains 332 verified image links and 332 label links,
including 100 empty hard-negative label files. The supplied detector split is
unchanged except that the checksum-mismatched training hard negative is
excluded. Known eye, seat-belt, and phone-classifier source videos are
group-exclusive. Rows without a known source video are training-only to avoid
claiming leakage-safe validation.

Generated local artifacts:

- `local_data/dms_handoff_20260730/prepared_manifest.csv`
- `local_data/dms_handoff_20260730/preparation_metadata.json`
- `local_data/dms_handoff_20260730/phone_yolo/data.yaml`
- `local_data/dms_handoff_20260730/eye_state_rknn_calibration.txt`

These are reproducible local outputs and are ignored by Git.

## Runtime behavior

- Eye classifier input is an RGB, ImageNet-normalized, static 96x96 tensor.
  Both landmark-derived eye crops are classified and probabilities averaged.
  Missing model, invalid crop, or low confidence retains EAR output.
- Seat-belt input is a 224x224 driver-torso crop derived from the reviewed
  handoff geometry. Invalid ROI, low blur/brightness quality, missing model,
  unsupported class, and low confidence all return `UNKNOWN`.
- Seat-belt `WORN`/`NOT_WORN` requires a configurable stable candidate period.
- Both model paths are disabled by default and use explicit ordered class maps
  loaded from training metadata.
- One-class YOLO output `[1, 5, N]` is parsed through the existing cabin
  detector, with class 0 mapped to `PHONE`. Association and evidence fusion are
  unchanged.

## Training/export smoke result

Environment: Python 3.10, PyTorch 2.12.1 CPU, torchvision 0.27.1, ONNX 1.22.0,
ONNX Runtime 1.23.2.

One eye epoch, no pretrained weights:

- validation accuracy: 0.9311
- balanced accuracy: 0.5000
- confusion matrix `[[0, 70], [0, 946]]`
- closed recall/F1: 0.0/0.0
- open recall/F1: 1.0/0.9643

The high raw accuracy is caused by class imbalance; balanced accuracy correctly
rejects this model. The ONNX plumbing passed:

- static opset: 12
- ONNX checker: passed
- OpenCV DNN load/inference: passed
- ordered output labels: `eye_closed`, `eye_open`

Full multi-epoch eye/seat-belt/phone training is intentionally not represented
as complete. Ultralytics is not installed, so phone detector training has not
started. RKNN conversion is deferred because no trained model has yet met
acceptance metrics.

## Tests

- New model/backend plus cabin-evidence targeted suite: 101 passed.
- Complete repository suite under the system Python: 398 passed, 2 failed.
- Both failures are unchanged environment failures in two tests that construct
  the default pipeline; the system interpreter lacks MediaPipe.
- `pytest.ini` now scopes discovery to `tests/`, preventing CPU test collection
  from importing RKNN command-line tools named `test_*.py`.

## Acceptance work still required

1. Reconcile or replace the mismatched phone frame upstream and regenerate the
   handoff hashes.
2. Train multiple seeded runs and choose by balanced accuracy and safety-class
   recall, not raw accuracy.
3. Evaluate per-source-video confusion matrices and collect more diverse
   closed-eye, no-seat-belt, NIR, glasses/glare, occlusion, and torso-pose data.
4. Train the one-class detector in an isolated Ultralytics environment and
   evaluate the retained empty-label hard negatives.
5. Enable ONNX models only after real-cabin latency and false-positive testing.
6. Convert accepted ONNX models with the RKNN environment, then compare ONNX
   and RKNN on identical samples and record accuracy delta, p50/p95 latency,
   FPS, memory, and thermal observations on AXON.

No commit, push, reset, clean, or source-handoff modification was performed.
