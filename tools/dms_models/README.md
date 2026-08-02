# Reviewed DMS model workflow

These tools consume the authoritative final-reviewed manifests from
`DMS_VICHARAK_HANDOFF_2026_0730`. They do not change or relabel the handoff.
Generated datasets and model artifacts are ignored by Git.

The only permitted image roots are:

```text
/home/vicharak/Mobility_ADAS/ADVIS/DMS/DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/01_Eye_state_dataset
/home/vicharak/Mobility_ADAS/ADVIS/DMS/DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/02_Seat_Belt_detection
/home/vicharak/Mobility_ADAS/ADVIS/DMS/DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/03_Phone_detection
```

Preparation and training fail if an image resolves outside the corresponding
task root. Legacy ImageFolder mixing tools are retained only as disabled
compatibility entry points.

## 1. Verify and prepare

```bash
PYTHONPATH=src python tools/dms_models/prepare_handoff_datasets.py \
  --output-dir local_data/dms_handoff_20260730
```

Strict mode exits with status 2 if any selected row is invalid. After reviewing
the recorded error, `--allow-invalid` excludes invalid rows without altering
the source. Add `--materialize-yolo` to create a verified YOLO symlink tree.

The preparation metadata records source-manifest hashes, seed, versions,
counts, exclusions, and validation errors. Known source videos are kept wholly
in one split. Classifier rows with unknown source video are training-only. The
phone detector's supplied train/validation split is preserved.

## 2. Train a classifier

```bash
PYTHONPATH=src python tools/dms_models/train_classifier.py \
  --manifest local_data/dms_handoff_20260730/prepared_manifest.csv \
  --task eye_state \
  --output-dir models/dms_classifiers/eye_state
```

Valid tasks are `eye_state`, `seat_belt`, and `phone_classifier`. The trainer
uses explicit ordered binary class maps, class-weighted loss, balanced accuracy
for checkpoint selection, static ONNX opset 12, ONNX checking, and ONNX Runtime
parity. Keep `--workers 0` on constrained AXON/container environments.

Do not enable a runtime backend only because export succeeded. Review its
confusion matrix, per-class recall/F1, parity, held-out video behavior, and
in-cabin latency first.

## 3. Train the one-class phone detector

```bash
PYTHONPATH=src python tools/dms_models/train_phone_yolo.py \
  --data local_data/dms_handoff_20260730/phone_yolo/data.yaml
```

This training-only command requires `ultralytics` in an isolated environment.
The runtime does not require Ultralytics. Empty YOLO label files are retained
as hard negatives. Export class 0 as `phone` and use
`configs/dms/cabin_object_class_map_phone.json`.

The existing COCO model remains the immutable
`old_baseline_coco_phone_detector`; see
`reports/phone_baseline/old_baseline_coco_phone_detector.md`. New artifacts
must stay under the distinct cabin-specific run/model directories.

## 4. RKNN calibration and conversion

Create calibration input exclusively from training rows:

```bash
PYTHONPATH=src python tools/eyenetrknn/make_rknn_calib_file.py \
  --manifest local_data/dms_handoff_20260730/prepared_manifest.csv \
  --task eye_state \
  --out local_data/dms_handoff_20260730/eye_state_rknn_calibration.txt
```

Only convert a model after CPU ONNX validation passes. Conversion is not AXON
runtime acceptance: compare ONNX and RKNN on identical samples and record
accuracy delta, latency percentiles, FPS, memory, and thermal behavior.
