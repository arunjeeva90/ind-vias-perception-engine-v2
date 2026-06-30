#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

from ind_vias_dms.eyenetrknn.landmark_eye_crop import (
    EYE_LEFT_IMG,
    EYE_RIGHT_IMG,
    crop_eye_from_landmarks,
    draw_eye_box,
)

from ind_vias_dms.eyenetrknn.rknnlite_classifier import EyeNetRKNNLiteClassifier

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
    "eyeA_score",
    "eyeB_score",
    "eyes_closed",
    "eye_baseline_valid",
    "eye_avg_score",
    "eyes_closed_raw",
    "eye_state",
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
    parser.add_argument("--overlay-scale", type=float, default=0.55)
    parser.add_argument("--overlay-thickness", type=int, default=1)
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
    parser.add_argument("--eye-closed-score-threshold", type=float, default=0.65)
    parser.add_argument("--eye-closed-avg-threshold", type=float, default=0.78)
    parser.add_argument("--eye-state-debounce-frames", type=int, default=3)
    parser.add_argument(
        "--eye-calib-log",
        nargs="?",
        const="outputs/rknn_eye_calib_landmarks.csv",
        default=None,
    )
    parser.add_argument(
        "--eyenetrknn-model",
        default="models/eyenetrknn/eyenetrknn_mnv3s_96_int8.rknn",
    )
    parser.add_argument("--show-eye-state", action="store_true")
    parser.add_argument(
        "--save-eye-crops-root",
        default="datasets/eye_state_live/train",
        help="Folder where live EyeNet crops are saved.",
    )
    parser.add_argument(
        "--crop-save-every",
        type=int,
        default=3,
        help="Save one left/right eye crop pair every N frames during active collection.",
    )
    parser.add_argument(
        "--eye-crop-side",
        type=float,
        default=42.0,
        help="Fixed eye crop side length in pixels before resizing to 96x96. Smaller = more zoom.",
    )
    parser.add_argument(
        "--eye-crop-smooth-alpha",
        type=float,
        default=0.65,
        help="EMA smoothing alpha for eye crop center. Higher = follows current frame more.",
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


def fit_overlay_text(text, max_width, font, scale, thickness):
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        return text

    ellipsis = "..."
    while text:
        candidate = text[:-1] + ellipsis
        if cv2.getTextSize(candidate, font, scale, thickness)[0][0] <= max_width:
            return candidate
        text = text[:-1]
    return ellipsis


def draw_overlay_dashboard(frame, lines, scale, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    x = 10
    y = 24
    pad_x = 8
    pad_y = 7
    line_gap = 7
    max_text_width = max(40, w - (x + pad_x) * 2)
    thickness = max(1, int(thickness))
    scale = max(0.35, float(scale))

    fitted_lines = [
        fit_overlay_text(line, max_text_width, font, scale, thickness) for line in lines
    ]
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in fitted_lines]
    line_height = max((size[1] for size in sizes), default=12)
    panel_width = min(
        w - x * 2,
        max((size[0] for size in sizes), default=0) + pad_x * 2,
    )
    panel_height = pad_y * 2 + len(fitted_lines) * line_height + (
        max(0, len(fitted_lines) - 1) * line_gap
    )

    overlay = frame.copy()
    panel_top = max(0, y - line_height - pad_y)
    panel_bottom = min(h - 1, panel_top + panel_height)
    cv2.rectangle(
        overlay,
        (x, panel_top),
        (x + panel_width, panel_bottom),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    text_y = y
    for line in fitted_lines:
        cv2.putText(
            frame,
            line,
            (x + pad_x, text_y),
            font,
            scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        text_y += line_height + line_gap



def save_current_eye_crops(
    root_dir,
    label,
    frame_id,
    left_eye_crop,
    right_eye_crop,
):
    out_dir = Path(root_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    ts = int(round(time.time() * 1000.0))

    if left_eye_crop is not None and left_eye_crop.size > 0:
        out_path = out_dir / f"{label}_frame_{frame_id:06d}_{ts}_L.png"
        cv2.imwrite(str(out_path), left_eye_crop)
        saved += 1

    if right_eye_crop is not None and right_eye_crop.size > 0:
        out_path = out_dir / f"{label}_frame_{frame_id:06d}_{ts}_R.png"
        cv2.imwrite(str(out_path), right_eye_crop)
        saved += 1

    return saved



def fuse_eyenetrknn_states(left_state, left_conf, right_state, right_conf):
    """
    Simple EyeNet fusion.

    Priority:
    - If both eyes are useful and disagree, return one_eye_closed.
    - If one useful eye exists, trust that eye.
    - If both are bad/unavailable, return highest-confidence raw state.
    """

    left_conf = float(left_conf or 0.0)
    right_conf = float(right_conf or 0.0)

    left_valid = left_state in ("eye_open", "eye_closed") and left_conf >= 0.60
    right_valid = right_state in ("eye_open", "eye_closed") and right_conf >= 0.60

    # Both eyes useful.
    if left_valid and right_valid:
        if left_state == "eye_closed" and right_state == "eye_closed":
            return "eye_closed", max(left_conf, right_conf)

        if left_state == "eye_open" and right_state == "eye_open":
            return "eye_open", max(left_conf, right_conf)

        return "one_eye_closed", max(left_conf, right_conf)

    # Only one eye useful.
    if left_valid:
        return left_state, left_conf

    if right_valid:
        return right_state, right_conf

    # No useful eye state.
    candidates = []
    if left_state is not None:
        candidates.append(("L", left_state, left_conf))
    if right_state is not None:
        candidates.append(("R", right_state, right_conf))

    if not candidates:
        return "NO_EYE", 0.0

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][1], candidates[0][2]


def debounce_state(raw_state, stable_state, pending_state, pending_count, needed_frames=4):
    """
    Change stable state only after raw_state repeats for needed_frames.
    This prevents EyeNet FINAL from flickering frame-by-frame.
    """

    if raw_state == stable_state:
        return stable_state, None, 0

    if raw_state == pending_state:
        pending_count += 1
    else:
        pending_state = raw_state
        pending_count = 1

    if pending_count >= needed_frames:
        stable_state = raw_state
        pending_state = None
        pending_count = 0

    return stable_state, pending_state, pending_count



def update_dms_eye_event_state(
    raw_final_state,
    dms_state,
    closed_count,
    open_count,
    bad_count,
    closed_confirm_frames=6,
    open_confirm_frames=3,
    bad_confirm_frames=12,
):
    """
    DMS-level eye event state.

    Important:
    - one_eye_closed is NOT drowsiness.
    - Full eye_closed requires both eyes closed through fused FINAL state.
    - bad_crop is tolerated briefly before declaring unreliable.
    """

    if raw_final_state == "eye_closed":
        closed_count += 1
        open_count = 0
        bad_count = 0

        if closed_count >= closed_confirm_frames:
            dms_state = "eye_closed"

    elif raw_final_state in ("eye_open", "one_eye_closed"):
        open_count += 1
        closed_count = 0
        bad_count = 0

        if open_count >= open_confirm_frames:
            if raw_final_state == "one_eye_closed":
                dms_state = "one_eye_closed"
            else:
                dms_state = "eye_open"

    elif raw_final_state in ("bad_crop", "NO_EYE"):
        bad_count += 1
        closed_count = 0
        open_count = 0

        if bad_count >= bad_confirm_frames:
            dms_state = "unreliable"

    else:
        # Unknown state: keep previous DMS state.
        pass

    return dms_state, closed_count, open_count, bad_count



def recrop_eye_with_smoothed_center(
    frame,
    box,
    prev_center,
    side=42.0,
    alpha=0.65,
):
    """
    Stabilize eye crop by smoothing crop center over time.

    This reduces per-frame landmark jitter:
    current landmark box center -> EMA smoothed center -> fixed-size crop

    side:
        smaller side = more internal digital zoom
        larger side  = more context but less eye detail
    """

    if box is None:
        return None, None, prev_center

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    cx = 0.5 * (float(x1) + float(x2))
    cy = 0.5 * (float(y1) + float(y2))

    alpha = float(alpha)
    alpha = max(0.05, min(0.95, alpha))

    if prev_center is None:
        sx, sy = cx, cy
    else:
        px, py = prev_center
        sx = alpha * cx + (1.0 - alpha) * px
        sy = alpha * cy + (1.0 - alpha) * py

    side = float(side)
    side = max(28.0, min(72.0, side))

    nx1 = int(round(sx - side / 2.0))
    ny1 = int(round(sy - side / 2.0))
    nx2 = int(round(sx + side / 2.0))
    ny2 = int(round(sy + side / 2.0))

    nx1 = max(0, min(w - 1, nx1))
    ny1 = max(0, min(h - 1, ny1))
    nx2 = max(0, min(w, nx2))
    ny2 = max(0, min(h, ny2))

    if nx2 <= nx1 or ny2 <= ny1:
        return None, None, (sx, sy)

    crop = frame[ny1:ny2, nx1:nx2]

    if crop.size == 0:
        return None, None, (sx, sy)

    return crop, (nx1, ny1, nx2, ny2), (sx, sy)


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
    eyeA_baseline = None
    eyeB_baseline = None
    eye_state = "OPEN"
    raw_closed_count = 0
    raw_open_count = 0
    eye_state_debounce_frames = max(1, args.eye_state_debounce_frames)
    active_crop_label = None
    last_crop_save_frame = -999999
    eyenetrknn_stable_state = "NO_EYE"
    eyenetrknn_pending_state = None
    eyenetrknn_pending_count = 0
    eyenetrknn_stable_conf = 0.0
    dms_eye_state = "unknown"
    dms_eye_closed_count = 0
    dms_eye_open_count = 0
    dms_eye_bad_count = 0
    eye_clf = None
    if args.show_eye_state:
        eye_clf = EyeNetRKNNLiteClassifier(
            model_path=args.eyenetrknn_model,
            img_size=96,
            input_color="bgr",
        )
        print(f"[INFO] EyeNet RKNN enabled: {args.eyenetrknn_model}")

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
            eyeA_score = None
            eyeB_score = None
            eyes_closed = None
            eye_avg_score = None
            eyes_closed_raw = None
            left_eye_crop = None
            right_eye_crop = None
            left_eye_state = None
            right_eye_state = None
            left_eye_conf = 0.0
            right_eye_conf = 0.0
            left_eye_box = None
            right_eye_box = None
            eyenet_left_text = "--"
            eyenet_right_text = "--"

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

                            if args.show_eye_state and eye_clf is not None:
                                pixel_points = np.empty_like(points, dtype=np.float32)
                                pixel_points[:, 0] = x1 + points[:, 0] * crop_w
                                pixel_points[:, 1] = y1 + points[:, 1] * crop_h

                                left_eye_crop, left_eye_box = crop_eye_from_landmarks(
                                    frame,
                                    pixel_points,
                                    EYE_LEFT_IMG,
                                )
                                right_eye_crop, right_eye_box = crop_eye_from_landmarks(
                                    frame,
                                    pixel_points,
                                    EYE_RIGHT_IMG,
                                )

                                if left_eye_crop is not None and left_eye_box is not None:
                                    l_cls, l_conf, _ = eye_clf.predict(left_eye_crop)
                                    left_eye_state = l_cls
                                    left_eye_conf = float(l_conf)
                                    eyenet_left_text = f"{l_cls}:{l_conf:.2f}"
                                    draw_eye_box(
                                        display,
                                        left_eye_box,
                                        f"L {l_cls} {l_conf:.2f}",
                                        color=(0, 255, 0),
                                    )

                                if right_eye_crop is not None and right_eye_box is not None:
                                    r_cls, r_conf, _ = eye_clf.predict(right_eye_crop)
                                    right_eye_state = r_cls
                                    right_eye_conf = float(r_conf)
                                    eyenet_right_text = f"{r_cls}:{r_conf:.2f}"
                                    draw_eye_box(
                                        display,
                                        right_eye_box,
                                        f"R {r_cls} {r_conf:.2f}",
                                        color=(0, 200, 255),
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

            if dms_signals is not None:
                eyeA = dms_signals["left_eye_open_proxy"]
                eyeB = dms_signals["right_eye_open_proxy"]
                baseline_valid = (
                    eyeA_baseline is not None
                    and eyeB_baseline is not None
                    and eyeA_baseline > 0.0
                    and eyeB_baseline > 0.0
                )
                if baseline_valid:
                    eyeA_score = eyeA / eyeA_baseline
                    eyeB_score = eyeB / eyeB_baseline
                    eye_avg_score = (eyeA_score + eyeB_score) * 0.5
                    eyes_closed_raw = (
                        (
                            eyeA_score < args.eye_closed_score_threshold
                            and eyeB_score < args.eye_closed_score_threshold
                        )
                        or eye_avg_score < args.eye_closed_avg_threshold
                    )
                    if eyes_closed_raw:
                        raw_closed_count += 1
                        raw_open_count = 0
                        if raw_closed_count >= eye_state_debounce_frames:
                            eye_state = "CLOSED"
                    else:
                        raw_open_count += 1
                        raw_closed_count = 0
                        if raw_open_count >= eye_state_debounce_frames:
                            eye_state = "OPEN"
                    eyes_closed = eye_state == "CLOSED"

            if args.show_dms_signals and dms_signals is not None:
                yaw_text = f"{dms_signals['yaw_proxy']:.2f}"
                pitch_text = f"{dms_signals['pitch_proxy']:.2f}"
                eyeA_text = f"{eyeA:.2f}"
                eyeB_text = f"{eyeB:.2f}"
                if eyeA_score is not None and eyeB_score is not None:
                    eyeA_score_text = f"{eyeA_score:.2f}"
                    eyeB_score_text = f"{eyeB_score:.2f}"
                    eye_avg_score_text = f"{eye_avg_score:.2f}"
                    eyes_state = eye_state
                else:
                    eyes_state = "NO_BASE"
                    eyeA_score_text = "--"
                    eyeB_score_text = "--"
                    eye_avg_score_text = "--"
            else:
                yaw_text = "--"
                pitch_text = "--"
                eyeA_text = "--"
                eyeB_text = "--"
                eyeA_score_text = "--"
                eyeB_score_text = "--"
                eye_avg_score_text = "--"
                eyes_state = "NO_BASE"

            overlay_lines = [
                f"RKNN | det={detector_name} | roi={roi_mode} | track={crop_mode}",
                (
                    f"pts={args.landmark_count} | "
                    f"lm={'valid' if landmark_valid else 'invalid'} | "
                    f"yawP={yaw_text} | pitchP={pitch_text}"
                ),
                (
                    f"eyeA={eyeA_text} eyeB={eyeB_text} | "
                    f"sA={eyeA_score_text} sB={eyeB_score_text}"
                ),
                (
                    f"avg={eye_avg_score_text} | "
                    f"thr={args.eye_closed_score_threshold:.2f}/{args.eye_closed_avg_threshold:.2f} | "
                    f"eyes={eyes_state}"
                ),
            ]
            if args.show_eye_state:
                overlay_lines.append(f"EyeNet L={eyenet_left_text} | R={eyenet_right_text}")
                raw_eye_state, raw_eye_conf = fuse_eyenetrknn_states(
                    left_eye_state,
                    left_eye_conf,
                    right_eye_state,
                    right_eye_conf,
                )

                eyenetrknn_stable_state, eyenetrknn_pending_state, eyenetrknn_pending_count = debounce_state(
                    raw_eye_state,
                    eyenetrknn_stable_state,
                    eyenetrknn_pending_state,
                    eyenetrknn_pending_count,
                    needed_frames=6,
                )

                eyenetrknn_stable_conf = raw_eye_conf

                overlay_lines.append(
                    f"EyeNet FINAL={eyenetrknn_stable_state}:{eyenetrknn_stable_conf:.2f} "
                    f"raw={raw_eye_state}:{raw_eye_conf:.2f} "
                    f"pend={eyenetrknn_pending_state}:{eyenetrknn_pending_count}"
                )
                dms_eye_state, dms_eye_closed_count, dms_eye_open_count, dms_eye_bad_count = update_dms_eye_event_state(
                    eyenetrknn_stable_state,
                    dms_eye_state,
                    dms_eye_closed_count,
                    dms_eye_open_count,
                    dms_eye_bad_count,
                    closed_confirm_frames=6,
                    open_confirm_frames=3,
                    bad_confirm_frames=12,
                )

                overlay_lines.append(
                    f"DMS_EYE={dms_eye_state} "
                    f"closed_cnt={dms_eye_closed_count} "
                    f"open_cnt={dms_eye_open_count} "
                    f"bad_cnt={dms_eye_bad_count}"
                )
            draw_overlay_dashboard(
                display,
                overlay_lines,
                args.overlay_scale,
                args.overlay_thickness,
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
                        "eyeA_score": f"{eyeA_score:.6f}" if eyeA_score is not None else "",
                        "eyeB_score": f"{eyeB_score:.6f}" if eyeB_score is not None else "",
                        "eyes_closed": (
                            int(eyes_closed) if eyes_closed is not None else ""
                        ),
                        "eye_baseline_valid": int(
                            eyeA_baseline is not None
                            and eyeB_baseline is not None
                            and eyeA_baseline > 0.0
                            and eyeB_baseline > 0.0
                        ),
                        "eye_avg_score": (
                            f"{eye_avg_score:.6f}" if eye_avg_score is not None else ""
                        ),
                        "eyes_closed_raw": (
                            int(eyes_closed_raw) if eyes_closed_raw is not None else ""
                        ),
                        "eye_state": eye_state if eye_avg_score is not None else "",
                    }
                )
                if frame_id % 30 == 0:
                    perf_handle.flush()

            if active_crop_label is not None:
                every_n = max(1, int(args.crop_save_every))
                if frame_id - last_crop_save_frame >= every_n:
                    saved = save_current_eye_crops(
                        args.save_eye_crops_root,
                        active_crop_label,
                        frame_id,
                        left_eye_crop,
                        right_eye_crop,
                    )
                    last_crop_save_frame = frame_id

                    if saved > 0 and frame_id % 30 == 0:
                        print(
                            f"[COLLECT] label={active_crop_label}, saved={saved}, "
                            f"root={args.save_eye_crops_root}"
                        )

            cv2.imshow("AXON RKNN Landmark Live Demo", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                cv2.imwrite(str(save_path), display)
                print(f"[OK] Saved snapshot: {save_path}")
            elif key == ord("b"):
                if dms_signals is None:
                    print("[WARN] No valid eye proxy values to capture baseline")
                else:
                    eyeA = dms_signals["left_eye_open_proxy"]
                    eyeB = dms_signals["right_eye_open_proxy"]
                    if eyeA > 0.0 and eyeB > 0.0:
                        eyeA_baseline = eyeA
                        eyeB_baseline = eyeB
                        print(
                            f"[OK] Captured open-eye baseline: "
                            f"eyeA={eyeA_baseline:.6f}, eyeB={eyeB_baseline:.6f}"
                        )
                    else:
                        print(
                            f"[WARN] Eye baseline not captured; "
                            f"eyeA={eyeA:.6f}, eyeB={eyeB:.6f}"
                        )
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
            elif key in (ord("1"), ord("2"), ord("3")):
                label_map = {
                    ord("1"): "eye_open",
                    ord("2"): "eye_closed",
                    ord("3"): "bad_crop",
                }
                active_crop_label = label_map[key]
                last_crop_save_frame = -999999
                print(f"[COLLECT] Started saving: {active_crop_label}")

            elif key == ord("0"):
                print(f"[COLLECT] Stopped saving. Previous label={active_crop_label}")
                active_crop_label = None

            elif key == ord(" "):
                if active_crop_label is None:
                    print("[COLLECT] No active label. Press 1=open, 2=closed, 3=bad_crop first.")
                else:
                    saved = save_current_eye_crops(
                        args.save_eye_crops_root,
                        active_crop_label,
                        frame_id,
                        left_eye_crop,
                        right_eye_crop,
                    )
                    print(f"[COLLECT] Manual save: label={active_crop_label}, saved={saved}")

            elif key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if eye_clf is not None:
            eye_clf.release()
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
