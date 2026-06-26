#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Open webcam, capture snap, crop face, save image.")
    parser.add_argument("--camera", type=int, default=1, help="Camera index, usually 0 or 1")
    parser.add_argument("--output", default="examples/dms_face_crop.jpg", help="Output crop path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fourcc", default="MJPG")
    return parser.parse_args()


def crop_face_or_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    h, w = frame.shape[:2]

    if len(faces) > 0:
        # Pick largest face
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])

        # Add margin around face
        margin = int(0.35 * max(fw, fh))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + fw + margin)
        y2 = min(h, y + fh + margin)

        crop = frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2), "face"

    # Fallback: center square crop
    side = min(h, w)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    x2 = x1 + side
    y2 = y1 + side
    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2), "center"


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index: {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if args.fourcc:
        fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    print("[INFO] Camera opened.")
    print("[INFO] Press 's' to save face crop.")
    print("[INFO] Press 'q' to quit.")

    last_frame = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Failed to read frame.")
            continue

        last_frame = frame.copy()
        preview = frame.copy()

        crop, bbox, mode = crop_face_or_center(frame)
        x1, y1, x2, y2 = bbox

        color = (0, 255, 0) if mode == "face" else (0, 255, 255)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            preview,
            f"crop mode: {mode} | press s=save q=quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Capture DMS Face Crop", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            crop, bbox, mode = crop_face_or_center(last_frame)

            # Resize to 192x192 for RKNN landmark PoC input convenience.
            crop_192 = cv2.resize(crop, (192, 192), interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(output_path), crop_192)
            print(f"[OK] Saved {mode} crop to: {output_path}")
            print(f"[OK] Output size: 192x192")
            break

        if key == ord("q"):
            print("[INFO] Quit without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
