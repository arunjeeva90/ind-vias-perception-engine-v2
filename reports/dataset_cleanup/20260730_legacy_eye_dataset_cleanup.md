# Legacy eye dataset cleanup — 2026-07-30

## Scope

Only obsolete local eye training-image directories are in scope. EyeNet source
code, training/export/conversion scripts, checkpoints, ONNX/RKNN models,
calibration metadata, runtime integration, tests, and documentation are
explicitly excluded.

These datasets were untracked/ignored local data. They were not part of the Git
repository or the authoritative reviewed handoff.

## Inventory before removal

Repository:
`/home/vicharak/Mobility_ADAS/ADVIS/DMS/ind-vias-perception-engine-v2`

| Path | Files | Images | Bytes | Class counts |
|---|---:|---:|---:|---|
| `datasets/eye_state` | 990 | 990 | 11,680,744 | bad_crop 208; eye_closed 208; eye_open 574 |
| `datasets/eye_state_finetune` | 6,175 | 6,175 | 19,575,257 | bad_crop 1,280; eye_closed 2,337; eye_open 2,558 |
| `datasets/eye_state_live` | 3,409 | 3,409 | 8,385,691 | bad_crop 1,176; eye_closed 2,233; eye_open 0 |
| `datasets/bad_images_quarantine` | 2 | 2 | 6,462 | eye_open 2 |
| `datasets/eye_state_mrl` | 0 | 0 | 0 | empty |
| `datasets/raw_mrl` | 0 | 0 | 0 | empty |
| **Total** | **10,576** | **10,576** | **39,648,154** | legacy only |

The file-level checksum inventory is
`reports/dataset_cleanup/legacy_eye_datasets_before_removal.sha256`.

## Protected artifacts confirmed outside removal targets

- `src/ind_vias_dms/eyenetrknn/`
- `src/ind_vias_dms/vision/eye_state.py`
- `src/ind_vias_dms/vision/onnx_classifier.py`
- `tools/eyenetrknn/`
- `tools/dms_models/`
- `models/eyenetrknn/`
- `models/dms_classifiers/eye_state_smoke/`
- Eye-related documentation, configuration, tests, and reports

## Authoritative replacement verification

Authoritative eye source:

```text
/home/vicharak/Mobility_ADAS/ADVIS/DMS/DMS_VICHARAK_HANDOFF_2026_0730/
01_IMAGES/01_Eye_state_dataset
```

The prepared eye manifest has 5,129 readable and checksum-valid selected
images:

- train: 375 `eye_closed`, 3,738 `eye_open`
- validation: 70 `eye_closed`, 946 `eye_open`
- uncertain/review-only/non-training rows excluded
- eye validation errors: zero
- source-video grouping retained
- split hash:
  `160a2a0e20ba04e9ca0da0facfb57b68595b26854e40bc43088bbfb8374ac61b`

The overall prepared manifest SHA-256 is
`510f29a06fbb8dd90c852636b3476e3b6fc83e2d92c36e819ce550002cb6ac4c`.

## Removal record

Moved out of the active repository into recoverable quarantine:

```text
/tmp/dms_legacy_eye_datasets_20260730.FM4eIG/datasets/
```

Exact directories removed from the active workspace:

- `datasets/eye_state`
- `datasets/eye_state_finetune`
- `datasets/eye_state_live`
- `datasets/bad_images_quarantine`
- `datasets/eye_state_mrl`
- `datasets/raw_mrl`

Post-move verification:

- 10,576 quarantined files
- all 10,576 SHA-256 entries reverified successfully
- zero remaining entries under the repository `datasets/` directory
- protected code, scripts, checkpoints, model binaries, metadata, tests, and
  documentation remain in place

## Archive follow-up incident

At the beginning of the accepted archive follow-up, the quarantine path no
longer existed:

```text
/tmp/dms_legacy_eye_datasets_20260730.FM4eIG/datasets/
```

`/tmp` is a `tmpfs` on this host and was purged between work turns. The
repository copies had already been moved, so the durable archive could not be
created.

Recovery checks performed before stopping:

- searched `/tmp` for another `dms_legacy_eye_datasets*` directory: none
- searched `/home/vicharak` for the legacy dataset roots and representative
  filenames: none
- checked the user Trash for representative legacy files: none
- compared all 6,177 unique hashes in the retained legacy SHA manifest against
  all authoritative handoff eye images and the eye-related Trash images:
  zero matching hashes

The file/path/hash inventory remains available, but the 10,576 file payload is
no longer recoverable from the AXON filesystem. The authoritative reviewed
handoff dataset was not affected.

The retained SHA-256 inventory is historical evidence only. None of the missing
legacy images may be reconstructed, substituted, recreated from the new
handoff, or silently represented as recovered. The zero hash matches confirm
that the old and new eye datasets are different datasets.

If an original Windows-workspace backup is found later, it must be placed
outside the repository under a durable archive path, verified against the
historical SHA-256 inventory, and permanently excluded from new training.
