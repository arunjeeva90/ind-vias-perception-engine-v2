from __future__ import annotations

from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


# Image-left eye in display, usually driver's right eye.
EYE_LEFT_IMG = [
    33, 34, 35, 36, 37, 38, 39, 40,
]

# Image-right eye in display, usually driver's left eye.
EYE_RIGHT_IMG = [
    87, 88, 89, 90, 94, 95, 96, 98, 99, 100,
]


def _remove_outliers(pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] < 6:
        return pts

    med = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - med, axis=1)

    keep_count = max(4, int(round(pts.shape[0] * 0.75)))
    keep_idx = np.argsort(dist)[:keep_count]

    return pts[keep_idx]


def crop_eye_from_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray,
    indices: Iterable[int],
    pad_x: float = 0.85,
    pad_y: float = 1.10,
    min_size: int = 8,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    h, w = frame.shape[:2]

    pts = []
    for idx in indices:
        if idx < 0 or idx >= len(landmarks):
            continue

        x, y = landmarks[idx]
        if np.isfinite(x) and np.isfinite(y):
            pts.append([float(x), float(y)])

    if len(pts) < 4:
        return None, None

    pts = np.asarray(pts, dtype=np.float32)
    pts = _remove_outliers(pts)

    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))

    bw = x2 - x1
    bh = y2 - y1

    if bw < min_size or bh < 3:
        return None, None

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    # Fixed crop size for current 640x480 DMS setup.
    # This makes left and right eye boxes same size.
    # Increase to 48 if too tight. Reduce to 36 if too much face/glasses is included.
    side = 42.0

    sx1 = int(round(cx - side / 2.0))
    sy1 = int(round(cy - side / 2.0))
    sx2 = int(round(cx + side / 2.0))
    sy2 = int(round(cy + side / 2.0))

    sx1 = max(0, min(w - 1, sx1))
    sy1 = max(0, min(h - 1, sy1))
    sx2 = max(0, min(w, sx2))
    sy2 = max(0, min(h, sy2))

    if sx2 <= sx1 or sy2 <= sy1:
        return None, None

    crop = frame[sy1:sy2, sx1:sx2]

    if crop.size == 0:
        return None, None

    return crop, (sx1, sy1, sx2, sy2)


def draw_eye_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    text: str,
    color=(0, 255, 0),
):
    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

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
