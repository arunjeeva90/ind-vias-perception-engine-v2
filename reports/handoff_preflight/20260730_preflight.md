# DMS Vicharak handoff preflight — 2026-07-30

## Repository identity

- Working directory and repository root:
  `/home/vicharak/Mobility_ADAS/ADVIS/DMS/ind-vias-perception-engine-v2`
- Remote:
  `origin https://github.com/arunjeeva90/ind-vias-perception-engine-v2.git`
- Branch: `feature/axon-runtime-v029`
- HEAD: `2ca322de07fb3e1c51139410d17240f51f1252e3`
- HEAD subject: `Add standalone EyeNet webcam demo`
- Upstream divergence from cached/refreshed
  `origin/feature/axon-runtime-v029`: ahead 1, behind 0.
- Local-only commit: `2ca322de Add standalone EyeNet webcam demo`.
- Remote-only commits on the current upstream: none.
- Stashes: none.
- Staged files: none.

## Preserved working tree

- Modified tracked file: `.gitignore` (+4 lines, local phone ONNX/RKNN
  exclusions).
- Deleted files: none.
- Untracked files: 1,002 before handoff work:
  - `datasets/`: 990
  - `models/`: 5
  - `result/`: 1
  - `src/`: 1
  - `test_images/`: 1
  - `tools/`: 4
- Important untracked source overlaps:
  - `src/ind_vias_dms/mobile_phone/__init__.py`
  - `tools/rknn/live_mobile_phone_detector_webcam.py`
  - `tools/rknn/test_mobile_phone_detector_image.py`
  - two backup variants of the image test script
- Planned handoff edits overlap tracked EyeNet, DMS pipeline, seat-belt,
  cabin-object, configuration, test and tooling areas. They do not overwrite
  the existing untracked phone scripts.

## Fetch result

`git fetch origin feature/axon-runtime-v029` could not update
`.git/FETCH_HEAD` because the workspace exposes `.git` read-only:

```text
error: cannot open .git/FETCH_HEAD: Read-only file system
```

The comparison therefore uses the remote-tracking references refreshed during
the immediately preceding repository audit.

## Baseline test discovery

Command:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest --collect-only -q -p no:cacheprovider
```

Result: 390 repository tests were discovered, but collection failed on three
RKNN command-line utilities under `tools/eyenetrknn/`. They are named
`test_*.py` and import `rknn`/`rknnlite`, which are intentionally absent from
the CPU/MediaPipe `.venv`. This is a baseline discovery/configuration defect,
not a handoff-model failure.

## Handoff checksum result

`SHA256SUMS.txt` uses CRLF endings, so a direct Linux `sha256sum -c` interprets
the carriage return as part of every filename. Verification with carriage
returns removed in-memory produced:

```text
ok=7848 failed=1 missing=0
```

The one mismatch is:

```text
01_IMAGES/03_Phone_detection/object_detection_yolo/images/train/
WIN_20260728_11_33_25_Pro_t0000005000_f0000150.jpg
```

- Manifest checksum:
  `e462f6ef3e3e01137b322847b86ddbb9ce80bc851334de306b0d1f842db67721`
- Actual checksum:
  `6fa96089a6ddd518f211eaa896c08fe2a1d129a11e0f44530b961585718b2c3c`
- Expected and actual size: 253,416 bytes.
- Label: hard negative with an intentionally empty YOLO label file.

This sample must be rejected from deterministic preparation until its
provenance is reconciled; it must not be silently repaired or relabelled.
