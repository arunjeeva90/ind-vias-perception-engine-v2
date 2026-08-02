# MobileNetV3-Small eye-state interim gate failure (2026-07-31)

## Decision

The ImageNet-pretrained MobileNetV3-Small candidate is rejected for deployment.
Three controlled 96×96 experiments were completed. No evaluated threshold in
any experiment met all mandatory crop-level gates.

This is a genuine progression blocker under the approved stop rule. The model
was therefore not:

- exported to ONNX;
- converted to RKNN;
- enabled in runtime configuration;
- used for video-level accuracy or temporal conclusions;
- used to replace any existing EyeNet artefact.

No TinyCNN experiment was repeated. No seat-belt or phone training was started.
Nothing was committed or pushed.

## Dataset answer and source gate

The apparent 447 `closed` and 4,686 `open` item counts include the `left` and
`right` child directories. Exact recursive PNG counts are:

| Reviewed class | PNG files | Child directories |
|---|---:|---:|
| `closed` | 445 | 2 |
| `open` | 4,684 | 2 |
| `uncertain` | 19 | 2 |

Every eligible reviewed binary crop is used by the accepted deterministic
split. Validation crops are held out from gradient training:

| Split | `eye_closed` | `eye_open` | Total |
|---|---:|---:|---:|
| Train | 375 | 3,738 | 4,113 |
| Validation | 70 | 946 | 1,016 |

Validation is described as **source-video-exclusive held-out reviewed crop
validation**.

- Authoritative image root:
  `/home/vicharak/Mobility_ADAS/ADVIS/DMS/DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/01_Eye_state_dataset`
- Reviewed-label manifest:
  `02_ANNOTATIONS_MANIFESTS/eye_state/final_reviewed_eye_labels.csv`
- Reviewed-label manifest SHA-256:
  `439806fe43abe06f9d7a5062b6c6bf93a18af99caffd2bee97359829bf82dca3`
- Prepared manifest:
  `local_data/dms_handoff_20260730/prepared_manifest.csv`
- Prepared manifest SHA-256:
  `510f29a06fbb8dd90c852636b3476e3b6fc83e2d92c36e819ce550002cb6ac4c`
- Split SHA-256:
  `160a2a0e20ba04e9ca0da0facfb57b68595b26854e40bc43088bbfb8374ac61b`
- Class order: `0: eye_closed`, `1: eye_open`
- Readable and hash-verified: 5,129/5,129
- Exact train/validation SHA-256 overlap: 0
- Source-video groups crossing train/validation: 0
- Excluded: 19 reviewed `uncertain` crops

The reviewed derivative crops sourced from `dmsNit207.mp4` and `mob_belt.mp4`
remain in the accepted crop split as explicitly approved. Their raw videos were
not opened or processed.

## Training protocol

- Architecture: TorchVision MobileNetV3-Small
- Pretraining: `MobileNet_V3_Small_Weights.IMAGENET1K_V1`
- Official weight URL:
  `https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth`
- Official weight SHA-256:
  `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`
- Native input: RGB `3×96×96`; no 224×224 upscaling
- Normalization: ImageNet mean `[0.485, 0.456, 0.406]`, standard deviation
  `[0.229, 0.224, 0.225]`
- Seed: `20260730`
- Stage 1: classifier frozen-backbone training, AdamW, learning rate `1e-3`
- Stage 2: final four feature blocks plus classifier, AdamW, learning rate
  `1e-4`
- Scheduler: reduce-on-plateau using safety score
- Early stopping: closed-eye F1 and balanced-accuracy safety score
- Best checkpoint restored rather than final epoch
- Batch size: 128
- Weight decay: `1e-4`
- Training hardware: AXON CPU, four Torch threads
- Validation augmentation: none

Training-only augmentation was limited to mild affine translation/rotation and
scale, brightness/contrast/colour-temperature variation, mild Gaussian blur,
and mild sensor noise. Random anatomical flipping was disabled. Experiment B
used a deterministic right-eye mirror in both training and validation.

## Experiment results

Rows in each confusion matrix are actual `[closed, open]`; columns are predicted
`[closed, open]`.

| Experiment | Orientation | Imbalance | Threshold | Closed P | Closed R | Closed F1 | Balanced accuracy | False closure | False open | Matrix | Gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| A | Original | Weighted CE | 0.78 | 0.632 | 0.786 | 0.701 | 0.876 | 0.0338 | 0.2143 | `[[55,15],[32,914]]` | Fail |
| B | Mirror right | Weighted CE | 0.75 | 0.471 | 0.914 | 0.621 | 0.919 | 0.0761 | 0.0857 | `[[64,6],[72,874]]` | Fail |
| C | Original | Balanced sampler + CE | 0.90 | 0.558 | 0.900 | 0.689 | 0.924 | 0.0529 | 0.1000 | `[[63,7],[50,896]]` | Fail |

Mandatory gates:

- closed recall ≥ 0.85;
- closed precision ≥ 0.75;
- closed F1 ≥ 0.80;
- balanced accuracy ≥ 0.85;
- exact train/validation hash overlap = 0;
- source-video train/validation overlap = 0.

Experiment A had the highest closed precision and F1, but missed recall,
precision, and F1. Experiment C had the highest balanced accuracy and adequate
recall, but missed precision and F1. No row in any saved 0.01–0.99 threshold
sweep passed all gates.

## Anatomical-orientation finding

Mirroring the right eye did not improve the candidate:

| Experiment | Side | Closed P | Closed R | Closed F1 |
|---|---|---:|---:|---:|
| Original weighted | Left | 0.741 | 0.769 | 0.755 |
| Original weighted | Right | 0.455 | 0.833 | 0.588 |
| Mirror-right weighted | Left | 0.671 | 0.904 | 0.770 |
| Mirror-right weighted | Right | 0.258 | 0.944 | 0.405 |
| Original balanced | Left | 0.708 | 0.885 | 0.786 |
| Original balanced | Right | 0.354 | 0.944 | 0.515 |

The mirrored experiment greatly increased right-eye false closures. Original
orientation remains the better supported preprocessing choice.

## Source-specific finding

For the best balanced-accuracy experiment:

| Held-out reviewed source | Closed P | Closed R | Closed F1 | Matrix |
|---|---:|---:|---:|---|
| `dmsNit207.mp4` derivative crops | 0.554 | 0.939 | 0.697 | `[[46,3],[37,654]]` |
| `mob_belt.mp4` derivative crops | 0.567 | 0.810 | 0.667 | `[[17,4],[13,209]]` |
| `mobile.mp4` derivative crops | N/A | N/A | N/A | `[[0,0],[0,33]]` |

The `mobile.mp4` subset contains no closed-eye support and is not used to claim
closed-class source performance.

## Error stability

Thirty-one validation crops were misclassified by all three experiments:

- 28 reviewed open crops were repeatedly predicted closed;
- 3 reviewed closed crops were repeatedly predicted open;
- 21 of the repeated open errors came from `dmsNit207.mp4`;
- 7 of the repeated open errors came from `mob_belt.mp4`;
- several filenames retain historical `uncertain_src` provenance even though
  their final manually reviewed authoritative label is `open`.

This stable cross-treatment error set and the right-eye precision gap indicate
a domain/appearance ambiguity that threshold tuning alone cannot resolve. The
authoritative final labels were not changed. Complete error rows and hashes are
saved in each experiment's `validation_errors.csv`.

## Native diagnostic artefacts

These checkpoints are diagnostic only and are not deployment candidates:

| Experiment directory | Selected checkpoint SHA-256 | Size |
|---|---|---:|
| `models/dms_classifiers/eye_state_mobilenet/mnv3s_original_weighted_seed20260730_workers0` | `7f3047160cb3df4963afff0b468bc63d49c01fc9e8fb51c6df210e684fe4a3e6` | 6,214,887 bytes |
| `models/dms_classifiers/eye_state_mobilenet/mnv3s_mirror_right_weighted_seed20260730_retry` | `907c205f3f058a36a9599452ff0af3ba121ba8b95e366b918e1b1858713cdf17` | 6,214,887 bytes |
| `models/dms_classifiers/eye_state_mobilenet/mnv3s_original_balanced_sampler_seed20260730` | `09b7c25eb8108387523755ea2e6902ec8101b26a4d7108db7e8efc98e66007b4` | 6,214,951 bytes |

Each completed directory also contains:

- `best.pt`;
- `selected_native.pt`;
- `training_result.json`;
- `history.json`;
- `crop_evaluation.json`;
- `threshold_sweep.csv`;
- `validation_predictions.csv`;
- `validation_errors.csv`.

Two empty directories record attempts stopped before an epoch artefact:

- `mnv3s_original_weighted_seed20260730` — loader-worker configuration was
  stopped after excessive first-epoch time;
- `mnv3s_mirror_right_weighted_seed20260730` — Pillow 9 flip-constant
  incompatibility, corrected before the distinct retry.

No existing EyeNet checkpoint was overwritten.

## Video status

No video was processed after the crop gate failure. This is an intentional early
stop, not a video pass.

Known provenance already classifies:

- `WIN_20260728_11_36_11_Pro.mp4` as `RUNTIME_REGRESSION_VIDEO`;
- `WIN_20260728_11_37_31_Pro.mp4` as `RUNTIME_REGRESSION_VIDEO`.

Their training-crop overlap means they cannot provide independent
generalisation accuracy. The other two permitted videos still require the
detailed leakage audit before any accuracy claim. Raw `dmsNit207.mp4` and
`mob_belt.mp4` were not used.

## Verification

- Training command compiled with `python -m py_compile`.
- Focused source/backend tests: 13 passed.
- Repository suite with the existing native `v0241` cases deselected:
  396 passed, 8 deselected.
- The eight `v0241` tests terminate the test process in the current native
  environment before pytest can report a result; they were not weakened or
  removed.
- `git diff --check`: clean.

## Recommended next decision

The controlled MobileNetV3-Small result should be reviewed before further model
work. If continuation is approved, the previously mandated next candidate is
EfficientNet-Lite0 using the identical manifest, seed, preprocessing,
augmentations, imbalance comparison, and metrics. If it reproduces the stable
false-closure set, the next useful action is data/error review or the ResNet18
capacity diagnostic—not more MobileNet threshold tuning.
