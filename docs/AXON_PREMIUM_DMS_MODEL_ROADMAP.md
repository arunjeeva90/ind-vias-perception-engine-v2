# AXON Premium DMS Model Roadmap

This roadmap keeps the production DMS MediaPipe runtime untouched while the RKNN proof of concept matures in `tools/rknn`.

## Stage 1: Better Face Detector First

The immediate priority is replacing the Haar/center-crop live demo behavior with a real face detector. YuNet is the first candidate because OpenCV can run it directly from ONNX via `cv2.FaceDetectorYN_create`, it is small enough for a live proof of concept, and it gives a stable face crop before RKNN landmark inference.

The Stage 1 goal is not to change the landmark model. It is to stop running landmarks on invalid center crops when detection fails. The live demo should run RKNN landmarks only when a valid detector crop exists, briefly reuse a recent valid crop during short detector misses, and otherwise show a clear no-face state.

## Stage 2: 106-Point Or Better Landmark Model

After the detector is stable, the next upgrade is a stronger landmark model. A 106-point face alignment model, or another model with better coverage around eyelids, mouth, and contour points, should be evaluated for RKNN conversion and runtime performance on AXON.

The selection criteria should include output contract clarity, RKNN conversion reliability, live inference latency, landmark stability under head motion, and license terms suitable for the intended product use.

## Stage 3: Head Pose, Gaze, And Eye-State Heads

Once detection and landmarks are stable, add focused task heads for DMS signals:

- Head pose for yaw, pitch, and roll.
- Gaze direction or coarse gaze zone.
- Eye-state signals such as open, closed, blink, and possible occlusion.

These heads can be separate RKNN models at first, then consolidated later only if latency, memory, and maintenance costs justify it.

## Candidate Model Sources And License Cautions

InsightFace and SCRFD pretrained models are useful references and evaluation candidates, but many pretrained releases are research-only unless separately licensed. Treat them as non-production candidates until licensing is confirmed.

RKNN Model Zoo face detection options, including RetinaFace-style candidates, should also be evaluated because they may provide a smoother RKNN conversion path and board-specific performance baseline.
