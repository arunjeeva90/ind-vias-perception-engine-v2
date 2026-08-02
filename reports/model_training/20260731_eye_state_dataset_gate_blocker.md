# Eye-state dataset gate blocker (2026-07-31)

## Outcome

MobileNetV3-Small training was initially paused because the authoritative
prepared split appeared to conflict with the prohibition on using
`dmsNit207.mp4` and `mob_belt.mp4`. The user resolved the ambiguity on
2026-07-31:

- the prohibition applies to direct processing of the raw video files;
- their already packaged, manually reviewed 96×96 derivative crops remain
  permitted;
- the accepted deterministic split must remain unchanged;
- this evaluation must be described as
  **source-video-exclusive held-out reviewed crop validation**.

The dataset gate is therefore cleared. The raw videos remain prohibited for
video-level validation, event timelines, annotated-video generation, temporal
tuning, and runtime conclusions.

No source image, split manifest, existing checkpoint, or runtime artefact was
modified.

## Folder-count clarification

The apparent counts of 447 items under `closed` and 4,686 items under `open`
include the two anatomical-side directories (`left` and `right`) in each class
folder. Exact recursive file counts are:

| Reviewed class | PNG files | Child directories |
|---|---:|---:|
| `closed` | 445 | 2 |
| `open` | 4,684 | 2 |
| `uncertain` | 19 | 2 |

All 5,129 eligible reviewed binary PNG files are represented in the prepared
manifest. The 19 uncertain files are excluded as required.

## Authoritative provenance

- Image root:
  `/home/vicharak/Mobility_ADAS/ADVIS/DMS/DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/01_Eye_state_dataset`
- Reviewed labels:
  `/home/vicharak/Mobility_ADAS/ADVIS/DMS/DMS_VICHARAK_HANDOFF_2026_0730/02_ANNOTATIONS_MANIFESTS/eye_state/final_reviewed_eye_labels.csv`
- Reviewed-label manifest SHA-256:
  `439806fe43abe06f9d7a5062b6c6bf93a18af99caffd2bee97359829bf82dca3`
- Prepared manifest:
  `local_data/dms_handoff_20260730/prepared_manifest.csv`
- Prepared manifest SHA-256:
  `510f29a06fbb8dd90c852636b3476e3b6fc83e2d92c36e819ce550002cb6ac4c`
- Eye split SHA-256:
  `160a2a0e20ba04e9ca0da0facfb57b68595b26854e40bc43088bbfb8374ac61b`
- Class order: `0: eye_closed`, `1: eye_open`
- Re-verification result: 5,129/5,129 files readable and SHA-256 correct
- Exact train/validation SHA-256 overlap: 0
- Source groups present in both train and validation: 0

Prepared counts:

| Split | `eye_closed` | `eye_open` | Total |
|---|---:|---:|---:|
| Train | 375 | 3,738 | 4,113 |
| Validation | 70 | 946 | 1,016 |

The 70 validation closed-eye samples are essential for measuring the mandatory
closed-eye precision, recall, and F1 gates.

## Prohibited-source conflict

The prepared validation rows have this source distribution:

| Source group | `eye_closed` | `eye_open` | Total |
|---|---:|---:|---:|
| `dmsNit207.mp4` | 49 | 691 | 740 |
| `mob_belt.mp4` | 21 | 222 | 243 |
| `mobile.mp4` | 0 | 33 | 33 |

The continuation prompt says not to use `dmsNit207.mp4` or `mob_belt.mp4` for
further diagnosis, threshold selection, validation, training, or model
conclusions. Applying that rule to crops derived from those videos removes 983
of 1,016 validation rows, including every validation closed-eye sample. The
remaining 33 rows are all `eye_open`.

An open-only validation set cannot produce meaningful closed-eye recall, F1, or
balanced accuracy and cannot support safety-threshold selection or early
stopping. Continuing with the unchanged split would use prohibited-source
crops for exactly those purposes. Silently moving other source groups would
change the accepted deterministic split and its hash.

## Authoritative-video overlap classification

The two continuously closed videos also contributed exact-provenance crops to
the training split:

| Authoritative video | Training crops | Labels | Classification |
|---|---:|---|---|
| `WIN_20260728_11_36_11_Pro.mp4` | 110 | all closed | `RUNTIME_REGRESSION_VIDEO` |
| `WIN_20260728_11_37_31_Pro.mp4` | 138 | all closed | `RUNTIME_REGRESSION_VIDEO` |

They must not be described as external held-out generalization evidence. They
remain useful for end-to-end runtime regression, eyeglass comparison, temporal
continuity, UNKNOWN handling, and latency, provided their non-independent
status is explicit.

## Resolution

The reviewed derivative crops are allowed under the accepted split. They must
not be re-extracted, regenerated, or relabelled from the prohibited raw videos.
MobileNetV3-Small crop-level training and evaluation may proceed.
