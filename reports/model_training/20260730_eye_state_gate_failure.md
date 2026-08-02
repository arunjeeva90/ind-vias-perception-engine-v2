# Eye-state controlled training experiments — gate failure

Date: 2026-07-30

## Decision

No eye-state experiment passed all mandatory acceptance gates. Training stopped
before ONNX export, RKNN conversion, runtime enablement, seat-belt training, or
phone training.

## Authoritative provenance

- Image root:
  `/home/vicharak/Mobility_ADAS/ADVIS/DMS/DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/01_Eye_state_dataset`
- Final reviewed manifest SHA-256:
  `439806fe43abe06f9d7a5062b6c6bf93a18af99caffd2bee97359829bf82dca3`
- Prepared manifest SHA-256:
  `510f29a06fbb8dd90c852636b3476e3b6fc83e2d92c36e819ce550002cb6ac4c`
- Eye split SHA-256:
  `160a2a0e20ba04e9ca0da0facfb57b68595b26854e40bc43088bbfb8374ac61b`
- Class order: class 0 `eye_closed`, class 1 `eye_open`
- Train: 375 closed, 3,738 open
- Validation: 70 closed, 946 open
- Excluded: 19 uncertain/not recommended

Leakage audit:

- exact train hashes: 4,113 unique
- exact validation hashes: 1,016 unique
- exact-hash overlap: zero
- known source-video overlap: zero
- 12 unknown-source closed crops: train-only

## Common model and training contract

- Architecture: static-shape-friendly 8,010-parameter depthwise CNN
- Input: 96×96 RGB
- Normalization: ImageNet mean/std
- Training-only augmentation:
  - resize to 96×96
  - ±5° affine rotation
  - ±3% translation
  - 0.95–1.05 scale
  - mild brightness/contrast/saturation jitter
  - horizontal flip
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001
- Batch size: 128
- Seed: 20260730
- Selection score: mean of closed-eye F1 and balanced accuracy
- Threshold: swept on validation probabilities; raw accuracy not used
- Early stopping patience: five epochs

## Experiment results

| Experiment | Best epoch | Threshold | Closed precision | Closed recall | Closed F1 | Balanced accuracy | Raw accuracy | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Weighted cross-entropy, shuffled batches | 13 | 0.800 | 0.619 | 0.857 | 0.719 | 0.909 | 0.954 | fail |
| Cross-entropy, balanced sampler | 14 | 0.775 | 0.585 | 0.786 | 0.671 | 0.872 | 0.947 | fail |
| Class-weighted focal loss, shuffled batches | 13 | 0.625 | 0.484 | 0.857 | 0.619 | 0.895 | 0.927 | fail |

Required gates:

- closed recall ≥ 0.85
- closed precision ≥ 0.75
- closed F1 ≥ 0.80
- balanced accuracy ≥ 0.85

No threshold in a fine 0.001–0.999 sweep satisfied all four gates for any
checkpoint.

## Confusion matrices

Rows are actual `[eye_closed, eye_open]`; columns are predicted
`[eye_closed, eye_open]`.

Weighted cross-entropy:

```text
[[60, 10],
 [37, 909]]
```

Balanced sampler:

```text
[[55, 15],
 [39, 907]]
```

Focal loss:

```text
[[60, 10],
 [64, 882]]
```

The best fine-threshold result overall was the weighted-loss checkpoint at
threshold 0.803:

- precision 0.625
- recall 0.857
- F1 0.723
- balanced accuracy 0.910
- confusion matrix `[[60, 10], [36, 910]]`

It still has no feasible safety-gate threshold.

## Source-level failure pattern

At the selected weighted-loss threshold:

- false closed among open eyes:
  - `dmsNit207.mp4`: 19
  - `mob_belt.mp4`: 18
- missed closed eyes:
  - `dmsNit207.mp4`: 8
  - `mob_belt.mp4`: 2

The same false-closure pattern persists under balanced sampling and focal loss.
This indicates a source/domain and visual-boundary problem, not only numerical
class imbalance. Increasing the threshold improves precision only by dropping
closed-eye recall below the required gate.

## Retained native experiment artifacts

These are rejected research checkpoints, not deployable models.

| Experiment | Checkpoint | Bytes | SHA-256 |
|---|---|---:|---|
| Weighted CE | `models/dms_classifiers/eye_state_experiments/weighted_ce_seed20260730/best.pt` | 51,713 | `c09e04bedbc50017ad163209e93eeddb4b763f03fdaeca91ee57acd8521475c4` |
| Balanced sampler | `models/dms_classifiers/eye_state_experiments/balanced_sampler_seed20260730/best.pt` | 51,713 | `12c40890243621463ae61ae8263f592bd7fdb4f96a662158530f26da36d11b0e` |
| Focal loss | `models/dms_classifiers/eye_state_experiments/focal_seed20260730_retry/best.pt` | 51,713 | `54594e9510bdcab57c9ed1e0aba516042024e07f8b674fe8d45a6cb94fd4faae` |

Each experiment directory contains `training_result.json` with provenance,
history, class order, metrics, threshold, and failed gate checks. The first
focal process was externally terminated before epoch 1; the identical retry
completed and is the reported focal result.

No ONNX or RKNN model was exported.

## Recommended collection and review

1. Review the 37 weighted-model false closures from `dmsNit207.mp4` and
   `mob_belt.mp4` for eyelid boundary, blur, NIR/lighting, crop alignment,
   glasses/reflection, and partially closed ambiguity.
2. Collect more independently reviewed open-eye examples resembling those
   false closures, without adjacent-frame leakage.
3. Collect more closed eyes from additional subjects/videos, especially night,
   glasses, glare, head pose, partial occlusion, and both eye sides.
4. Add a separately reviewed crop-quality dataset if a future quality gate is
   desired; do not fabricate `bad_crop` labels from these binary samples.
5. Revisit ambiguous open/closed boundary policy before another training run.

Until new reviewed data or corrected labels change the precision–recall
frontier, retain EAR/landmark fallback and keep the ONNX eye backend disabled.
