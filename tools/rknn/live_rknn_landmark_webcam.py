#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def parse_args():
    parser = argparse.ArgumentParser(description="Live RKNN face landmark webcam demo")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--model", default="models/dms/landmark_rk3588.rknn")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--input-size", nargs=2, type=int, default=[160, 160])
    parser.add_argument("--save-snapshot", default="outputs/rknn_live_snapshot.jpg")
    return parser.parse_args()


def detect_face_or_center(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    h, w = frame.shape[:2]

    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])

        # Make a square-ish crop with margin.
        side = int(max(fw, fh) * 1.55)
        cx = x + fw // 2
        cy = y + fh // 2

        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)

        # Re-adjust if clipped.
        x1 = max(0, x2 - side)
        y1 = max(0, y2 - side)

        return (x1, y1, x2, y2), "face"

    # Fallback center crop.
    side = min(h, w)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    x2 = x1 + side
    y2 = y1 + side
    return (x1, y1, x2, y2), "center"


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"RKNN model not found: {model_path}")

    save_path = Path(args.save_snapshot)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    input_w, input_h = args.input_size

    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise SystemExit("Could not load OpenCV Haar face cascade")

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

    frame_count = 0
    fps_t0 = time.time()
    fps = 0.0
    infer_ms = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Failed to read frame")
                continue

            display = frame.copy()
            h, w = frame.shape[:2]

            bbox, mode = detect_face_or_center(frame, face_cascade)
            x1, y1, x2, y2 = bbox

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_AREA)

            # Same format as successful smoke test: NHWC uint8.
            input_tensor = np.expand_dims(crop_resized, axis=0).astype(np.uint8)

            t0 = time.time()
            outputs = rknn.inference(inputs=[input_tensor])
            infer_ms = (time.time() - t0) * 1000.0

            if outputs is not None and len(outputs) >= 3:
                landmarks = np.asarray(outputs[2]).reshape(-1)

                if landmarks.size == 136:
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

                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.putText(
                        display,
                        f"Unexpected landmark size: {landmarks.size}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

            frame_count += 1
            now = time.time()
            dt = now - fps_t0
            if dt >= 1.0:
                fps = frame_count / dt
                frame_count = 0
                fps_t0 = now

            cv2.putText(
                display,
                f"RKNN live landmarks | crop={mode} | FPS={fps:.1f} | infer={infer_ms:.1f} ms",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

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
        print("[INFO] Released camera and RKNN runtime")


if __name__ == "__main__":
    main()
