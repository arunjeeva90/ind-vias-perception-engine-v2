from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ind_vias_dms.eyenetrknn.rknnlite_classifier import EyeNetRKNNLiteClassifier


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fourcc", type=str, default="MJPG")

    parser.add_argument(
        "--eyenetrknn-model",
        default="models/eyenetrknn/eyenetrknn_mnv3s_96_int8.rknn",
    )

    parser.add_argument(
        "--yunet-model",
        default="models/dms/face_detection_yunet.onnx",
    )

    parser.add_argument("--score-thr", type=float, default=0.60)
    parser.add_argument("--show-crops", action="store_true")

    return parser.parse_args()


def create_yunet(model_path: str, width: int, height: int, score_thr: float):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"YuNet model not found: {model_path}")

    detector = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (width, height),
        score_thr,
        0.3,
        5000,
    )

    return detector


def detect_largest_face(detector, frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    face = faces[0]

    x, y, fw, fh = face[:4].astype(int)

    x = max(0, x)
    y = max(0, y)
    fw = min(fw, w - x)
    fh = min(fh, h - y)

    if fw <= 0 or fh <= 0:
        return None

    return x, y, fw, fh


def crop_eye_regions_from_face(frame, face_box):
    """
    Approximate eye crops from face bbox.

    This is only for live pipeline validation.
    Final version should use 106 face landmarks.
    """
    x, y, w, h = face_box

    # Eye band inside face bbox.
    eye_y1 = int(y + 0.30 * h)
    eye_y2 = int(y + 0.47 * h)

    left_x1 = int(x + 0.16 * w)
    left_x2 = int(x + 0.46 * w)

    right_x1 = int(x + 0.54 * w)
    right_x2 = int(x + 0.84 * w)

    H, W = frame.shape[:2]

    def clamp_crop(x1, y1, x2, y2):
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W, x2))
        y2 = max(0, min(H, y2))

        if x2 <= x1 or y2 <= y1:
            return None, (x1, y1, x2, y2)

        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    left_crop, left_box = clamp_crop(left_x1, eye_y1, left_x2, eye_y2)
    right_crop, right_box = clamp_crop(right_x1, eye_y1, right_x2, eye_y2)

    return left_crop, right_crop, left_box, right_box


def draw_prediction(frame, box, label, conf, color):
    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {conf:.2f}"
    cv2.putText(
        frame,
        text,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if args.fourcc:
        fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    detector = create_yunet(
        model_path=args.yunet_model,
        width=args.width,
        height=args.height,
        score_thr=args.score_thr,
    )

    eye_clf = EyeNetRKNNLiteClassifier(
        model_path=args.eyenetrknn_model,
        img_size=96,
        input_color="bgr",
    )

    prev_t = time.time()
    fps = 0.0

    print("Press q to quit.")

    while True:
        ok, frame = cap.read()

        if not ok or frame is None:
            print("Failed to read frame")
            break

        face_box = detect_largest_face(detector, frame)

        if face_box is not None:
            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            left_crop, right_crop, left_box, right_box = crop_eye_regions_from_face(
                frame,
                face_box,
            )

            if left_crop is not None:
                l_cls, l_conf, _ = eye_clf.predict(left_crop)
                draw_prediction(frame, left_box, "L " + l_cls, l_conf, (0, 255, 0))

            if right_crop is not None:
                r_cls, r_conf, _ = eye_clf.predict(right_crop)
                draw_prediction(frame, right_box, "R " + r_cls, r_conf, (0, 200, 255))

            if args.show_crops:
                if left_crop is not None:
                    cv2.imshow("left_eye_crop", cv2.resize(left_crop, (192, 96)))
                if right_crop is not None:
                    cv2.imshow("right_eye_crop", cv2.resize(right_crop, (192, 96)))
        else:
            cv2.putText(
                frame,
                "No face",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        now = time.time()
        dt = now - prev_t
        prev_t = now

        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(
            frame,
            f"EyeNet-RKNN FPS: {fps:.1f}",
            (20, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("EyeNet RKNN Live", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    eye_clf.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
