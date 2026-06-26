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
    "roi_mode",
    "selected_face_count",
    "detected_face_count",
    "yaw_proxy",
    "pitch_proxy",
    "left_eye_open_proxy",
    "right_eye_open_proxy",
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
    parser.add_argument("--driver-roi", choices=["full", "left", "right"], default="full")
    parser.add_argument(
        "--driver-roi-custom",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Optional normalized driver ROI overriding --driver-roi.",
    )
    parser.add_argument("--driver-track-hold-ms", type=int, default=1500)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--input-size", nargs=2, type=int, default=[160, 160])
    parser.add_argument("--landmark-count", type=int, choices=[68, 106], default=68)
    parser.add_argument("--draw-landmark-indices", action="store_true")
    parser.add_argument("--landmark-index-step", type=int, default=1)
    parser.add_argument(
        "--landmark-coord-mode",
        choices=["auto", "zero_one", "minus_one_one"],
        default="auto",
    )
    parser.add_argument("--save-snapshot", default="outputs/rknn_live_snapshot.jpg")
    parser.add_argument("--perf-log", nargs="?", const="outputs/rknn_live_perf.csv", default=None)
    parser.add_argument("--show-dms-signals", action="store_true")
    parser.add_argument(
        "--eye-calib-log",
        nargs="?",
        const="outputs/rknn_eye_calib_landmarks.csv",
        default=None,
    )
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

    bboxes = []
    for x, y, fw, fh in faces:
        bbox = clamp_expand_bbox((x, y, fw, fh), frame.shape, margin=margin)
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes


def detect_face_yunet(frame, yunet_detector, margin):
    h, w = frame.shape[:2]
    yunet_detector.setInputSize((w, h))
    _, faces = yunet_detector.detect(frame)
    if faces is None or len(faces) == 0:
        return []

    bboxes = []
    for face in faces:
        x, y, fw, fh = face[:4]
        bbox = clamp_expand_bbox((x, y, fw, fh), frame.shape, margin=margin)
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes


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


def select_landmark_tensor(outputs, expected_size):
    if outputs is None:
        return None
    for output in outputs:
        landmarks = np.asarray(output).reshape(-1)
        if landmarks.size == expected_size:
            return landmarks
    return None


def resolve_landmark_coord_mode(raw_landmarks, requested_mode):
    if requested_mode != "auto":
        return requested_mode
    if float(np.min(raw_landmarks)) < -0.05:
        return "minus_one_one"
    return "zero_one"


def decode_landmarks(raw_landmarks, coord_mode):
    if coord_mode == "minus_one_one":
        return (raw_landmarks + 1.0) * 0.5
    return raw_landmarks


def landmarks_are_normalized(landmarks):
    if not np.all(np.isfinite(landmarks)):
        return False
    inside = np.logical_and(landmarks >= -0.1, landmarks <= 1.1)
    return float(np.count_nonzero(inside)) / float(landmarks.size) >= 0.95


def resolve_driver_roi(args, frame_shape):
    h, w = frame_shape[:2]
    if args.driver_roi_custom is not None:
        nx1, ny1, nx2, ny2 = args.driver_roi_custom
        roi_mode = "custom"
    elif args.driver_roi == "left":
        nx1, ny1, nx2, ny2 = 0.0, 0.0, 0.5, 1.0
        roi_mode = "left"
    elif args.driver_roi == "right":
        nx1, ny1, nx2, ny2 = 0.5, 0.0, 1.0, 1.0
        roi_mode = "right"
    else:
        nx1, ny1, nx2, ny2 = 0.0, 0.0, 1.0, 1.0
        roi_mode = "full"

    nx1 = max(0.0, min(1.0, nx1))
    ny1 = max(0.0, min(1.0, ny1))
    nx2 = max(0.0, min(1.0, nx2))
    ny2 = max(0.0, min(1.0, ny2))
    if nx2 <= nx1 or ny2 <= ny1:
        raise SystemExit(
            "--driver-roi-custom must define X1 Y1 X2 Y2 with X2 > X1 and Y2 > Y1"
        )

    x1 = int(round(nx1 * w))
    y1 = int(round(ny1 * h))
    x2 = int(round(nx2 * w))
    y2 = int(round(ny2 * h))
    return (x1, y1, x2, y2), roi_mode


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def center_inside_roi(bbox, roi):
    cx, cy = bbox_center(bbox)
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2


def select_driver_bbox(candidates, roi, previous_bbox, previous_t, now, hold_ms):
    roi_candidates = [bbox for bbox in candidates if center_inside_roi(bbox, roi)]
    previous_is_recent = (
        previous_bbox is not None and (now - previous_t) * 1000.0 <= hold_ms
    )

    if roi_candidates:
        if previous_is_recent:
            pcx, pcy = bbox_center(previous_bbox)
            selected = min(
                roi_candidates,
                key=lambda bbox: (bbox_center(bbox)[0] - pcx) ** 2
                + (bbox_center(bbox)[1] - pcy) ** 2,
            )
        else:
            selected = max(roi_candidates, key=bbox_area)
        return selected, "face", len(roi_candidates)

    if previous_is_recent:
        return previous_bbox, "hold", 0

    return None, "no-face", 0


def eye_open_proxy_from_indices(points, indices):
    eye = points[indices]
    eye_width = float(np.max(eye[:, 0]) - np.min(eye[:, 0]))
    if eye_width <= 1e-6:
        return 0.0
    eye_height = float(np.max(eye[:, 1]) - np.min(eye[:, 1]))
    return eye_height / eye_width


def compute_eye_open_from_group(points, indices):
    indices = np.asarray(indices, dtype=np.int32)
    if indices.size == 0 or int(np.max(indices)) >= points.shape[0]:
        return 0.0

    group = points[indices]
    group_width = float(np.max(group[:, 0]) - np.min(group[:, 0]))
    if group_width <= 1e-6:
        return 0.0
    group_height = float(np.max(group[:, 1]) - np.min(group[:, 1]))
    return group_height / group_width


def compute_eye_open_from_pair(points, idx_a, idx_b):
    if max(idx_a, idx_b) >= points.shape[0]:
        return 0.0
    return float(np.linalg.norm(points[idx_a] - points[idx_b]))


def eye_open_proxy_from_region(points, face_min, face_size, side):
    min_x, min_y = face_min
    width, height = face_size
    if width <= 1e-6 or height <= 1e-6:
        return 0.0

    center_x = min_x + width * 0.5
    if side == "left":
        x_min, x_max = min_x + width * 0.12, center_x
    else:
        x_min, x_max = center_x, min_x + width * 0.88
    y_min, y_max = min_y + height * 0.20, min_y + height * 0.58

    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )
    region = points[mask]
    if region.shape[0] < 2:
        return 0.0

    region_width = float(np.max(region[:, 0]) - np.min(region[:, 0]))
    if region_width <= 1e-6:
        return 0.0
    region_height = float(np.max(region[:, 1]) - np.min(region[:, 1]))
    return region_height / region_width


def compute_dms_signal_proxies(points, landmark_count):
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    size = max_xy - min_xy
    center = (min_xy + max_xy) * 0.5

    face_width_norm = float(size[0])
    face_height_norm = float(size[1])
    signals = {
        "face_center_x": float(center[0]),
        "face_center_y": float(center[1]),
        "face_width_norm": face_width_norm,
        "face_height_norm": face_height_norm,
        "yaw_proxy": float((center[0] - 0.5) * 2.0),
        "pitch_proxy": float((center[1] - 0.5) * 2.0),
        "left_eye_open_proxy": 0.0,
        "right_eye_open_proxy": 0.0,
    }

    if landmark_count == 106 and points.shape[0] >= 106:
        signals["left_eye_open_proxy"] = compute_eye_open_from_pair(points, 33, 40)
        signals["right_eye_open_proxy"] = compute_eye_open_from_pair(points, 87, 94)
    elif landmark_count == 68 and points.shape[0] >= 48:
        signals["left_eye_open_proxy"] = eye_open_proxy_from_indices(
            points,
            np.array([36, 37, 38, 39, 40, 41]),
        )
        signals["right_eye_open_proxy"] = eye_open_proxy_from_indices(
            points,
            np.array([42, 43, 44, 45, 46, 47]),
        )
    else:
        signals["left_eye_open_proxy"] = eye_open_proxy_from_region(
            points,
            min_xy,
            size,
            "left",
        )
        signals["right_eye_open_proxy"] = eye_open_proxy_from_region(
            points,
            min_xy,
            size,
            "right",
        )

    return signals


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


def open_eye_calib_log(path, landmark_count):
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp_ms", "label", "frame_id", "landmark_count"]
    for idx in range(landmark_count):
        fields.extend([f"x{idx}", f"y{idx}"])

    write_header = not log_path.exists() or log_path.stat().st_size == 0
    handle = log_path.open("a", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if write_header:
        writer.writeheader()
        handle.flush()
    return handle, writer


def write_eye_calib_sample(writer, timestamp_ms, label, frame_id, points):
    row = {
        "timestamp_ms": timestamp_ms,
        "label": label,
        "frame_id": frame_id,
        "landmark_count": points.shape[0],
    }
    for idx, (x, y) in enumerate(points):
        row[f"x{idx}"] = f"{float(x):.8f}"
        row[f"y{idx}"] = f"{float(y):.8f}"
    writer.writerow(row)


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"RKNN model not found: {model_path}")

    save_path = Path(args.save_snapshot)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    input_w, input_h = args.input_size
    expected_landmark_values = args.landmark_count * 2
    landmark_index_step = max(1, args.landmark_index_step)

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

    eye_calib_handle = None
    eye_calib_writer = None
    if args.eye_calib_log:
        eye_calib_handle, eye_calib_writer = open_eye_calib_log(
            args.eye_calib_log,
            args.landmark_count,
        )
        print(f"[INFO] Writing eye calibration samples: {args.eye_calib_log}")
        print("[INFO] Press o for open-eye sample, c for closed-eye sample.")

    frame_count = 0
    frame_id = 0
    fps_t0 = time.time()
    fps = 0.0
    infer_ms = 0.0
    last_driver_bbox = None
    last_driver_bbox_t = 0.0

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
            coord_mode_used = args.landmark_coord_mode
            selected_face_count = 0
            detected_face_count = 0
            dms_signals = None
            latest_valid_points = None

            display = frame.copy()
            h, w = frame.shape[:2]
            driver_roi, roi_mode = resolve_driver_roi(args, frame.shape)

            detector_t0 = time.time()
            if detector_name == "yunet":
                detected_bboxes = detect_face_yunet(frame, yunet_detector, args.bbox_margin)
            else:
                detected_bboxes = detect_face_haar(frame, face_cascade, args.bbox_margin)
            detector_ms = (time.time() - detector_t0) * 1000.0
            detected_face_count = len(detected_bboxes)

            now = time.time()
            bbox, crop_mode, _roi_face_count = select_driver_bbox(
                detected_bboxes,
                driver_roi,
                last_driver_bbox,
                last_driver_bbox_t,
                now,
                args.driver_track_hold_ms,
            )
            selected_face_count = 1 if bbox is not None else 0
            if crop_mode == "face":
                last_driver_bbox = bbox
                last_driver_bbox_t = now
            elif crop_mode == "no-face":
                infer_ms = 0.0

            cv2.rectangle(
                display,
                (driver_roi[0], driver_roi[1]),
                (driver_roi[2], driver_roi[3]),
                (255, 180, 0),
                2,
            )

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

                    landmarks = select_landmark_tensor(outputs, expected_landmark_values)
                    if landmarks is not None:
                        if np.all(np.isfinite(landmarks)):
                            coord_mode_used = resolve_landmark_coord_mode(
                                landmarks,
                                args.landmark_coord_mode,
                            )
                            decoded_landmarks = decode_landmarks(landmarks, coord_mode_used)
                            if np.all(np.isfinite(decoded_landmarks)):
                                landmark_min = f"{float(np.min(decoded_landmarks)):.6f}"
                                landmark_max = f"{float(np.max(decoded_landmarks)):.6f}"
                        else:
                            decoded_landmarks = landmarks

                        if landmarks_are_normalized(decoded_landmarks):
                            landmark_valid = True
                            points = decoded_landmarks.reshape(args.landmark_count, 2)
                            latest_valid_points = points.copy()
                            if args.show_dms_signals:
                                dms_signals = compute_dms_signal_proxies(
                                    points,
                                    args.landmark_count,
                                )

                            crop_w = x2 - x1
                            crop_h = y2 - y1

                            for idx, (nx, ny) in enumerate(points):
                                px = int(round(x1 + float(nx) * crop_w))
                                py = int(round(y1 + float(ny) * crop_h))

                                px = max(0, min(w - 1, px))
                                py = max(0, min(h - 1, py))

                                cv2.circle(display, (px, py), 2, (0, 255, 0), -1)
                                if (
                                    args.draw_landmark_indices
                                    and idx % landmark_index_step == 0
                                ):
                                    cv2.putText(
                                        display,
                                        str(idx),
                                        (px + 3, py - 3),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.28,
                                        (255, 255, 255),
                                        1,
                                        cv2.LINE_AA,
                                    )
                        else:
                            cv2.putText(
                                display,
                                f"Landmarks outside normalized range ({coord_mode_used})",
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2,
                                cv2.LINE_AA,
                            )
                    else:
                        output_sizes = []
                        if outputs is not None:
                            output_sizes = [np.asarray(output).size for output in outputs]
                        cv2.putText(
                            display,
                            f"No landmark tensor size {expected_landmark_values}: {output_sizes}",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
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
                    f"RKNN landmarks | detector={detector_name} | roi={roi_mode} | "
                    f"track={crop_mode} | "
                    f"pts={args.landmark_count} | "
                    f"lm={'valid' if landmark_valid else 'invalid'} | "
                    f"coord={coord_mode_used} | FPS={fps:.1f} | infer={infer_ms:.1f} ms"
                ),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if args.show_dms_signals and dms_signals is not None:
                cv2.putText(
                    display,
                    (
                        f"yawP={dms_signals['yaw_proxy']:.2f} "
                        f"pitchP={dms_signals['pitch_proxy']:.2f} "
                        f"eyeA={dms_signals['left_eye_open_proxy']:.2f} "
                        f"eyeB={dms_signals['right_eye_open_proxy']:.2f}"
                    ),
                    (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
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
                        "roi_mode": roi_mode,
                        "selected_face_count": selected_face_count,
                        "detected_face_count": detected_face_count,
                        "yaw_proxy": (
                            f"{dms_signals['yaw_proxy']:.6f}"
                            if dms_signals is not None
                            else ""
                        ),
                        "pitch_proxy": (
                            f"{dms_signals['pitch_proxy']:.6f}"
                            if dms_signals is not None
                            else ""
                        ),
                        "left_eye_open_proxy": (
                            f"{dms_signals['left_eye_open_proxy']:.6f}"
                            if dms_signals is not None
                            else ""
                        ),
                        "right_eye_open_proxy": (
                            f"{dms_signals['right_eye_open_proxy']:.6f}"
                            if dms_signals is not None
                            else ""
                        ),
                    }
                )
                if frame_id % 30 == 0:
                    perf_handle.flush()

            cv2.imshow("AXON RKNN Landmark Live Demo", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                cv2.imwrite(str(save_path), display)
                print(f"[OK] Saved snapshot: {save_path}")
            elif key in (ord("o"), ord("c")):
                label = "open" if key == ord("o") else "closed"
                if eye_calib_writer is None:
                    print("[WARN] Eye calibration log is not enabled. Pass --eye-calib-log.")
                elif latest_valid_points is None:
                    print(f"[WARN] No valid landmarks to save for label={label}")
                else:
                    write_eye_calib_sample(
                        eye_calib_writer,
                        timestamp_ms,
                        label,
                        frame_id,
                        latest_valid_points,
                    )
                    eye_calib_handle.flush()
                    print(
                        f"[OK] Saved eye calibration sample: "
                        f"label={label}, frame_id={frame_id}, points={args.landmark_count}"
                    )
            elif key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        rknn.release()
        if perf_handle is not None:
            perf_handle.flush()
            perf_handle.close()
        if eye_calib_handle is not None:
            eye_calib_handle.flush()
            eye_calib_handle.close()
        print("[INFO] Released camera and RKNN runtime")


if __name__ == "__main__":
    main()
