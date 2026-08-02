# Assessment: RepViT, quality heads, bilateral fusion, and temporal eye state

Date: 2026-07-31
Decision: **Useful architecture direction, but not trainable as proposed from the current authoritative handoff**

## Executive decision

The proposal adds real value in four areas:

1. separate left/right eye inference with shared weights;
2. quality-aware `UNKNOWN` handling;
3. bilateral fusion;
4. temporal blink/closure/PERCLOS reasoning.

Those ideas are more important than choosing a CNN versus a transformer.

RepViT is a credible future student-backbone candidate, but it is not yet the
production choice for this RK3588 project. The cited RepViT latency is measured
on an iPhone 12, not the RK3588 NPU. Rockchip's official RKNN model zoo does not
currently list RepViT as a verified classification example, so ONNX export,
operator compatibility, numerical parity, FP16/INT8 conversion, and device
latency all remain unproven.

Do not start another blind architecture run. The two completed
EfficientNet-Lite0 experiments show that the current limitation is validation
class separation—especially hard-open image-right eyes—not insufficient model
capacity alone.

## What the cited evidence does and does not establish

### RepViT

The RepViT paper supports the statement that RepViT is a pure lightweight CNN
which adopts design lessons from lightweight ViTs and has a favorable mobile
accuracy/latency trade-off:

https://arxiv.org/abs/2307.09283

It does **not** establish:

- eye-state accuracy;
- glasses/glare robustness;
- RK3588 latency;
- RKNN operator compatibility;
- INT8 eye-state parity.

Therefore, "RepViT will be the winner" is a hypothesis, not evidence.

### Human-Centered Benchmarking paper

The June 2026 preprint compares MobileNetV3, ShuffleNetV2, EfficientNet-B0, and
DeiT-Tiny on MRL Eye and reports a real robustness/efficiency trade-off:

https://arxiv.org/abs/2606.08123

Its main lesson is valid: clean accuracy alone is inadequate, and closed-as-open
errors under degradation need separate measurement.

It does **not** evaluate:

- RepViT;
- the ADVIS handoff;
- RK3588/RKNN;
- a learned quality head;
- bilateral eye fusion;
- a TCN/GRU;
- the proposed five-class output.

Its results should shape the robustness test protocol, not be copied as an
architecture conclusion for this device.

### Rockchip model support

The official RKNN model zoo verifies conventional MobileNet/ResNet
classification examples for RK3588 in FP16/INT8, but it does not list RepViT:

https://github.com/airockchip/rknn_model_zoo

This makes MobileNetV3-Large the lower-risk deployment candidate. RepViT should
enter only as a conversion/latency probe after the data/label gate is repaired.

### Zero-DCE++

The paper supports that Zero-DCE++ is very small:

https://arxiv.org/abs/2103.00860

It does not show that enhancement preserves subtle eyelid evidence or improves
closed-eye safety on this camera. Enhancement remains an A/B experiment only,
after a raw-input classifier passes.

## Current authoritative handoff capability

Only this dataset may be used:

`/home/vicharak/Mobility_ADAS/ADVIS/DMS/DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/01_Eye_state_dataset`

Final reviewed contents:

| Label | Count |
|---|---:|
| open | 4,684 |
| closed | 445 |
| uncertain, excluded | 19 |
| total | 5,148 |

All 5,148 files are already 96x96 RGB. There are no higher-resolution native
eye crops in the permitted image root. Upscaling 96x96 to 128x128 or 160x160
cannot restore discarded eyelid/glare detail and would not be a valid resolution
comparison.

The final manifest contains no fields for:

- subject/driver identity;
- glasses type;
- glasses presence;
- glare/reflection;
- occlusion;
- partial closure;
- bad crop;
- blur/dark quality labels.

Consequences:

- a subject-disjoint split cannot be proven;
- five-state supervised training cannot be performed;
- the proposed six-label quality head cannot be trained;
- teacher/student distillation cannot create missing ground truth;
- temporal TCN/GRU training cannot be validated from independent labelled
  sequences.

The current split is source-video-exclusive, which is stronger than random
frame splitting, but it must not be called subject-disjoint because the
manifest does not prove that different videos contain different drivers.

## Existing provenance that may be used for analysis only

The original reviewed-pipeline provenance contains deterministic blur,
brightness, exposure, EAR, crop geometry, and frame information for 4,858 of
the 5,148 final images:

- joined open: 4,675 / 4,684;
- joined closed: 164 / 445;
- joined uncertain: 19 / 19.

Coverage is strongly label-biased because 281 closed images lack that
provenance. These numeric fields can support diagnostic stratification, but
they are not complete supervised quality labels. They contain no glare,
glasses, or occlusion ground truth.

The older synthetic multistate material under
`02_ANNOTATIONS_MANIFESTS/ORIGINAL_20260729_PROVENANCE` is not an authorised
training image source. It must not be introduced into the current model.

## Post-hoc evidence from the failed canonical EfficientNet trial

### Blur/quality abstention

Of the 87 open crops incorrectly predicted closed:

- all 87 have joined extraction provenance;
- median blur score is 32.317;
- 63 are image-right and 24 are image-left;
- 55 come from `dmsNit207.mp4`;
- 32 come from `mob_belt.mp4`.

Correctly classified open crops have a much higher median blur score of
114.420, so blur/quality evidence is useful.

However, a grid search over classifier threshold and minimum blur threshold,
treating rejected crops as `UNKNOWN`, found no operating point that passed all
four deployment gates. The best precision while retaining closed recall >=
0.85 and balanced accuracy >= 0.85 was:

- precision: 0.5446;
- recall: 0.8714;
- F1: 0.6703;
- balanced accuracy: 0.8511;
- minimum blur: 26;
- 109 open and 6 closed validation crops abstained.

Conclusion: quality abstention improves safety but does not rescue the current
single-eye model.

### Bilateral fusion

The crop-level validation manifest contains:

- 228 left/right pairs;
- 220 pairs with the same reviewed label;
- 8 mixed-label pairs;
- 560 unpaired single-eye groups.

Among the 220 same-label pairs there are only 13 closed pairs and 207 open
pairs. On this limited subset, mean probability fusion at threshold 0.65 gives:

- precision: 0.7500;
- recall: 0.9231;
- F1: 0.8276;
- balanced accuracy: 0.9519;
- confusion counts: TP 12, FP 4, TN 203, FN 1.

This passes the numeric gates on the paired subset and is strong evidence that
bilateral fusion is advantageous. It is **not** a deployment qualification:

- only 13 closed pairs are present;
- 560 validation groups lack a pair;
- pair labels are not an independent pair-level review;
- validation still represents only the existing source groups.

The correct next model should preserve pair identity and evaluate pair-level
gates explicitly.

## Adopt now

These changes are already represented by the current disabled-by-default
runtime work and should remain:

- reviewed-contract, roll/scale-normalised eye crops;
- separate per-eye inference through one shared classifier;
- deterministic blur/dark/overexposure/padding gates;
- `UNKNOWN` when both eyes are unusable;
- bilateral disagreement -> `UNKNOWN`;
- one-eye fallback with a confidence penalty;
- optional MediaPipe/106-point geometry agreement;
- rule-based temporal blink, sustained-closure, and PERCLOS logic;
- no forced-open default.

Rule-based temporal logic remains preferable to a GRU/TCN at this stage because
it requires no unverified temporal training labels and is inspectable in a
safety review.

## Adopt as the next data/benchmark specification

For future permitted collection and review:

1. preserve native-resolution aligned eye crops before creating 96/128/160
   derivatives;
2. assign stable subject IDs and keep subjects disjoint across train,
   validation, and test;
3. preserve frame/pair/sequence IDs;
4. review pair-level state;
5. add state labels:
   `open`, `partial`, `closed`, `occluded`, `bad_crop`;
6. add independent multi-label quality tags:
   blur, low light, overexposure, glasses reflection, occlusion, out of frame;
7. include real glasses/reflection conditions rather than relying on synthetic
   glare alone;
8. create a source-exclusive corruption benchmark for motion blur, defocus,
   noise, darkness, local shadow, compression, crop shift, and glare;
9. record unknown coverage and false-open/false-closed rates, not only accuracy.

Any future raw extraction must obey the existing raw-video prohibitions.
Reviewed derivative crops already in the handoff may continue to be used.

## Model sequence after the data gate

Use the same paired/quality dataset and benchmark for every candidate:

1. MobileNetV3-Large 128 as the lower-risk RKNN baseline;
2. RepViT-M1.0 as the primary research candidate;
3. ShuffleNetV2 x1.0 as the latency reference;
4. DeiT-Tiny as a robustness reference only.

For each candidate:

1. qualify native FP32 crop and pair metrics;
2. prove ONNX parity;
3. inspect all ONNX/RKNN operators;
4. prove FP16 RKNN parity and AXON latency;
5. establish INT8 PTQ per-condition metrics;
6. attempt QAT only if PTQ causes a measurable gate regression.

Teacher/student distillation should be introduced only after the teacher itself
passes subject/source-exclusive tests. Otherwise it will distil the same
dataset bias into a smaller model.

## Final architecture recommendation

With the current labels:

`face -> landmark agreement -> aligned 96x96 per-eye crops -> deterministic
quality gates -> shared binary encoder -> conservative bilateral fusion ->
UNKNOWN/open/closed evidence -> rule-based temporal state`

With a future upgraded dataset:

`face -> landmark agreement -> native/128px per-eye crops -> shared
RepViT-or-MobileNetV3-Large encoder -> state head + quality head ->
quality-weighted pair fusion -> optional small TCN/GRU -> temporal safety state`

No new model was trained or exported from this assessment. Runtime configuration
remains disabled. No prohibited raw video was processed, and no legacy or
synthetic training images were introduced.
