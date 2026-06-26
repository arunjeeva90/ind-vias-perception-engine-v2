#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


PERF_LOG_FIELDS = [
    "timestamp_ms",
    "frame_id",
    "detector",
    "crop_mode",
    "face_valid",
    "landmark_valid",
    "detector_ms",
    "inference_ms",
    "total_frame_ms",
    "fps",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "landmark_min",
    "landmark_max",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Live RKNN face landmark webcam demo")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--model", default="models/dms/landmark_rk3588.rknn")
    parser.add_argument("--detector", choices=["haar", "yunet"], default=None)
    parser.add_argument("--yunet-model", default="models/dms/face_detection_yunet.onnx")
    parser.add_argument("--allow-detector-fallback", action="store_true")
    parser.add_argument("--bbox-hold-ms", type=int, default=800)
    parser.add_argument("--bbox-margin", type=float, default=1.30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--input-size", nargs=2, type=int, default=[160, 160])
    parser.add_argument("--save-snapshot", default="outputs/rknn_live_snapshot.jpg")
    parser.add_argument("--perf-log", nargs="?", const="outputs/rknn_live_perf.csv", default=None)
    return parser.parse_args()


def clamp_expand_bbox(bbox, frame_shape, margin=1.30):
    x, y, bw, bh = bbox
    h, w = frame_shape[:2]
    if bw <= 0 or bh <= 0 or w <= 0 or h <= 0:
        return None

    cx = float(x) + float(bw) * 0.5
    cy = float(y) + float(bh) * 0.5
    side = max(float(bw), float(bh)) * float(margin)
    side = max(2.0, side)

    x1 = int(round(cx - side * 0.5))
    y1 = int(round(cy - side * 0.5))
    x2 = int(round(cx + side * 0.5))
    y2 = int(round(cy + side * 0.5))

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        shift = x2 - w
        x1 = max(0, x1 - shift)
        x2 = w
    if y2 > h:
        shift = y2 - h
        y1 = max(0, y1 - shift)
        y2 = h

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def detect_face_haar(frame, face_cascade, margin):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        return clamp_expand_bbox((x, y, fw, fh), frame.shape, margin=margin)

    return None


def detect_face_yunet(frame, yunet_detector, margin):
    h, w = frame.shape[:2]
    yunet_detector.setInputSize((w, h))
    _, faces = yunet_detector.detect(frame)
    if faces is None or len(faces) == 0:
        return None

    face = max(faces, key=lambda b: float(b[2]) * float(b[3]))
    x, y, fw, fh = face[:4]
    return clamp_expand_bbox((x, y, fw, fh), frame.shape, margin=margin)


def create_yunet_detector(model_path):
    if not hasattr(cv2, "FaceDetectorYN_create"):
        raise RuntimeError("OpenCV build does not include cv2.FaceDetectorYN_create")
    return cv2.FaceDetectorYN_create(
        str(model_path),
        "",
        (320, 320),
        0.9,
        0.3,
        5000,
    )


def landmarks_are_normalized(landmarks):
    finite = np.isfinite(landmarks)
    if not np.all(finite):
        return False
    return float(np.min(landmarks)) >= -0.25 and float(np.max(landmarks)) <= 1.25


def open_perf_log(path):
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    handle = log_path.open("a", newline="")
    writer = csv.DictWriter(handle, fieldnames=PERF_LOG_FIELDS)
    if write_header:
        writer.writeheader()
        handle.flush()
    return handle, writer


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"RKNN model not found: {model_path}")

    save_path = Path(args.save_snapshot)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    input_w, input_h = args.input_size

    yunet_model_path = Path(args.yunet_model)
    detector_name = args.detector
    if detector_name is None:
        detector_name = "yunet" if yunet_model_path.exists() else "haar"

    face_cascade = None
    yunet_detector = None

    def load_haar_detector():
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(face_cascade_path)
        if cascade.empty():
            raise SystemExit("Could not load OpenCV Haar face cascade")
        return cascade

    if detector_name == "yunet":
        try:
            if not yunet_model_path.exists():
                raise FileNotFoundError(f"YuNet model not found: {yunet_model_path}")
            yunet_detector = create_yunet_detector(yunet_model_path)
        except Exception as exc:
            if not args.allow_detector_fallback:
                raise SystemExit(
                    f"YuNet detector unavailable: {exc}. "
                    "Install/use an OpenCV build with FaceDetectorYN and provide "
                    "--yunet-model, or pass --detector haar. Add "
                    "--allow-detector-fallback to continue with Haar."
                ) from exc
            print(f"[WARN] YuNet detector unavailable: {exc}")
            print("[WARN] Falling back to Haar detector because --allow-detector-fallback is set")
            detector_name = "haar"
            face_cascade = load_haar_detector()
    else:
        face_cascade = load_haar_detector()

    print("[INFO] Loading RKNN model:", model_path)
    rknn = RKNNLite()

    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise SystemExit(f"RKNNLite load_rknn failed: {ret}")

    ret = rknn.init_runtime()
    if ret != 0:
        raise SystemExit(f"RKNNLite init_runtime failed: {ret}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        rknn.release()
        raise SystemExit(f"Could not open camera index: {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))

    print("[INFO] Camera opened.")
    print("[INFO] Press q to quit, s to save snapshot.")

    perf_handle = None
    perf_writer = None
    if args.perf_log:
        perf_handle, perf_writer = open_perf_log(args.perf_log)
        print(f"[INFO] Writing performance CSV: {args.perf_log}")

    frame_count = 0
    frame_id = 0
    fps_t0 = time.time()
    fps = 0.0
    infer_ms = 0.0
    last_good_bbox = None
    last_good_bbox_t = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Failed to read frame")
                continue

            frame_t0 = time.time()
            timestamp_ms = int(round(frame_t0 * 1000.0))
            frame_id += 1
            detector_ms = 0.0
            infer_ms = 0.0
            landmark_valid = False
            landmark_min = ""
            landmark_max = ""

            display = frame.copy()
            h, w = frame.shape[:2]

            detector_t0 = time.time()
            if detector_name == "yunet":
                detected_bbox = detect_face_yunet(frame, yunet_detector, args.bbox_margin)
            else:
                detected_bbox = detect_face_haar(frame, face_cascade, args.bbox_margin)
            detector_ms = (time.time() - detector_t0) * 1000.0

            now = time.time()
            if detected_bbox is not None:
                bbox = detected_bbox
                crop_mode = "face"
                last_good_bbox = bbox
                last_good_bbox_t = now
            elif (
                last_good_bbox is not None
                and (now - last_good_bbox_t) * 1000.0 <= args.bbox_hold_ms
            ):
                bbox = last_good_bbox
                crop_mode = "hold"
            else:
                bbox = None
                crop_mode = "no-face"
                infer_ms = 0.0

            if bbox is None:
                cv2.putText(
                    display,
                    "NO VALID FACE CROP",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_AREA)

                    # Same format as successful smoke test: NHWC uint8.
                    input_tensor = np.expand_dims(crop_resized, axis=0).astype(np.uint8)

                    t0 = time.time()
                    outputs = rknn.inference(inputs=[input_tensor])
                    infer_ms = (time.time() - t0) * 1000.0

                    if outputs is not None and len(outputs) >= 3:
                        landmarks = np.asarray(outputs[2]).reshape(-1)
                        if landmarks.size > 0 and np.all(np.isfinite(landmarks)):
                            landmark_min = f"{float(np.min(landmarks)):.6f}"
                            landmark_max = f"{float(np.max(landmarks)):.6f}"

                        if landmarks.size == 136 and landmarks_are_normalized(landmarks):
                            landmark_valid = True
                            points = landmarks.reshape(68, 2)

                            crop_w = x2 - x1
                            crop_h = y2 - y1

                            for idx, (nx, ny) in enumerate(points):
                                # Outputs appear normalized 0..1 based on earlier min/max.
                                px = int(round(x1 + float(nx) * crop_w))
                                py = int(round(y1 + float(ny) * crop_h))

                                px = max(0, min(w - 1, px))
                                py = max(0, min(h - 1, py))

                                cv2.circle(display, (px, py), 2, (0, 255, 0), -1)
                        elif landmarks.size != 136:
                            cv2.putText(
                                display,
                                f"Unexpected landmark size: {landmarks.size}",
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2,
                            )
                        else:
                            cv2.putText(
                                display,
                                "Landmarks outside normalized range",
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2,
                                cv2.LINE_AA,
                            )
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.putText(
                        display,
                        "NO VALID FACE CROP",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            frame_count += 1
            dt = now - fps_t0
            if dt >= 1.0:
                fps = frame_count / dt
                frame_count = 0
                fps_t0 = now

            cv2.putText(
                display,
                (
                    f"RKNN landmarks | detector={detector_name} | crop={crop_mode} | "
                    f"FPS={fps:.1f} | infer={infer_ms:.1f} ms"
                ),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            total_frame_ms = (time.time() - frame_t0) * 1000.0

            if perf_writer is not None:
                if bbox is None:
                    bbox_x1 = bbox_y1 = bbox_x2 = bbox_y2 = ""
                else:
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

                perf_writer.writerow(
                    {
                        "timestamp_ms": timestamp_ms,
                        "frame_id": frame_id,
                        "detector": detector_name,
                        "crop_mode": crop_mode,
                        "face_valid": int(bbox is not None),
                        "landmark_valid": int(landmark_valid),
                        "detector_ms": f"{detector_ms:.3f}",
                        "inference_ms": f"{infer_ms:.3f}",
                        "total_frame_ms": f"{total_frame_ms:.3f}",
                        "fps": f"{fps:.3f}",
                        "bbox_x1": bbox_x1,
                        "bbox_y1": bbox_y1,
                        "bbox_x2": bbox_x2,
                        "bbox_y2": bbox_y2,
                        "landmark_min": landmark_min,
                        "landmark_max": landmark_max,
                    }
                )
                if frame_id % 30 == 0:
                    perf_handle.flush()

            cv2.imshow("AXON RKNN Landmark Live Demo", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                cv2.imwrite(str(save_path), display)
                print(f"[OK] Saved snapshot: {save_path}")
            elif key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        rknn.release()
        if perf_handle is not None:
            perf_handle.flush()
            perf_handle.close()
        print("[INFO] Released camera and RKNN runtime")


if __name__ == "__main__":
    main()
