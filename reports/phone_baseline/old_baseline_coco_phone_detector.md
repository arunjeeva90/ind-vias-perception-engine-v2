# Retained phone baseline: `old_baseline_coco_phone_detector`

This name documents the existing local COCO YOLOv8n mobile-phone experiment as
the immutable comparison baseline. Existing paths are deliberately unchanged
because the current inference scripts reference them.

## Preserved model artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `models/mobile_phone_detector/yolov8n.onnx` | 12,650,184 | `0c8716701f471067932b797eeb67c8e5db47c693c2557c881d7679ec12e21bc5` |
| `models/mobile_phone_detector/yolov8n.rknn` | 4,327,819 | `9f7a3a37158d19a252e7133ce38b8fcd809ae3cf894919fb6df9560eaa558bb5` |

## Preserved configuration and class metadata

| Artifact | SHA-256 |
|---|---|
| `configs/dms/cabin_object_class_map_coco_phone.json` | `641d5b83b05cceae96eb1b8a5a2732f93e4a0d03edb3674af45127304497b927` |
| `configs/yolov8n_coco_demo.yaml` | `725eb477a3463d16d4a89a57ddffdf92fd005ed410dde54f4aafc40e3cbf650c` |
| `configs/phone_demo_1440.yaml` | `0cfcdd121f89765aca06e8aa1603a6ce4e5d86130c512aad499e92cbed62327f` |

## Preserved inference and experiment scripts

| Artifact | SHA-256 |
|---|---|
| `tools/rknn/live_mobile_phone_detector_webcam.py` | `e436f9652475c7e939897544f0371caea32aa338fe9a4d7054ffb357864a3402` |
| `tools/rknn/test_mobile_phone_detector_image.py` | `8489e96f5ab69d32d0dd8c48cef53bbd359f3b9fe7588c854171e60df78dba8` |
| `tools/rknn/test_mobile_phone_detector_image.py.bak` | `f24226aa7b4bc32b83807b94d501f9e92435da031502873b1e0d0980d1471c7e` |
| `tools/rknn/test_mobile_phone_detector_image.py.phone_patch_bak` | `e8eb181a2ce6409d064a6cfe15bdd45de2868b2298a01f66aba7db84b1fbb22f` |

## Preserved test inputs/results

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `test_images/mobile_phone/phone_test.jpg` | 625,566 | `d669851340e885e6b88d5bc2ae32572ab71dfca355840ae730ff46e1d6162f00` |
| `result/phone_test.jpg` | 384,730 | `b9692060b1219e000a73e337b3aaabd5c933df72366fd820e9829badd70d87db` |

No formal baseline precision/recall/mAP report or self-contained conversion log
was found alongside these local artifacts. This absence is recorded rather
than reconstructed or guessed. The retained ONNX/RKNN models, configs, scripts,
test input, and rendered test result will remain unchanged. The planned
cabin-specific model will use separate artifact names and directories.

Runtime configuration must not switch away from this baseline until both
models are evaluated on the same final-reviewed validation images and the
cabin-specific model passes the agreed gates.
