from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Sequence

import cv2
import numpy as np


# InsightFace 2d106det layout, expressed in image order.  Each eye has ten
# points; the older PoC incorrectly omitted 41/42 and mixed 98/99/100 eyebrow
# points into the image-right eye.
LANDMARK_106_IMAGE_LEFT_EYE = tuple(range(33, 43))
LANDMARK_106_IMAGE_RIGHT_EYE = tuple(range(87, 97))
LANDMARK_106_IMAGE_LEFT_EYEBROW = tuple(range(43, 52))
LANDMARK_106_IMAGE_RIGHT_EYEBROW = tuple(range(97, 106))


@dataclass(frozen=True)
class EyeCropObservation:
    image: np.ndarray | None
    valid: bool
    reason: str
    eye_width: float = 0.0
    rotation_degrees: float = 0.0
    padding_fraction: float = 1.0
    blur_score: float = 0.0
    brightness_score: float = 0.0
    dark_pixel_fraction: float = 1.0
    bright_pixel_fraction: float = 0.0


def aligned_eye_crop(
    frame: np.ndarray,
    corner_a: Sequence[float],
    corner_b: Sequence[float],
    *,
    image_size: int = 96,
    context_scale: float = 1.65,
    eyebrow_shift: float = 0.10,
    min_eye_width: float = 18.0,
    max_padding_fraction: float = 0.25,
    min_blur: float = 12.0,
    min_brightness: float = 18.0,
    max_brightness: float = 235.0,
    max_dark_fraction: float = 0.60,
    max_bright_fraction: float = 0.55,
) -> EyeCropObservation:
    """Reproduce the reviewed handoff's aligned 96x96 eye-crop contract.

    The validity checks are runtime abstention gates.  They do not relabel an
    image and they do not substitute geometric eye openness for the classifier.
    """

    if (
        frame is None
        or frame.size == 0
        or frame.ndim not in (2, 3)
        or image_size <= 0
        or context_scale <= 0
    ):
        return EyeCropObservation(None, False, "INVALID_FRAME")

    a = np.asarray(corner_a, dtype=np.float64).reshape(-1)
    b = np.asarray(corner_b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2 or not np.isfinite(a[:2]).all() or not np.isfinite(b[:2]).all():
        return EyeCropObservation(None, False, "INVALID_CORNERS")

    eye_width = float(hypot(float(a[0] - b[0]), float(a[1] - b[1])))
    if eye_width < min_eye_width:
        return EyeCropObservation(
            None,
            False,
            "EYE_TOO_SMALL",
            eye_width=eye_width,
        )

    center_x = float((a[0] + b[0]) * 0.5)
    center_y = float((a[1] + b[1]) * 0.5)
    angle = degrees(atan2(float(b[1] - a[1]), float(b[0] - a[0])))
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0

    side = max(12.0, eye_width * context_scale)
    shifted_center_y = center_y - eye_width * eyebrow_shift
    matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    aligned = cv2.warpAffine(
        frame,
        matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    shifted_center = matrix @ np.asarray(
        [center_x, shifted_center_y, 1.0],
        dtype=np.float64,
    )
    native, padding_fraction = _square_crop_with_reflect_padding(
        aligned,
        float(shifted_center[0]),
        float(shifted_center[1]),
        side,
    )
    if native is None:
        return EyeCropObservation(
            None,
            False,
            "CROP_FAILED",
            eye_width=eye_width,
            rotation_degrees=angle,
        )

    interpolation = cv2.INTER_AREA if native.shape[0] >= image_size else cv2.INTER_CUBIC
    image = cv2.resize(native, (image_size, image_size), interpolation=interpolation)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    metrics = eye_crop_quality(image)
    reason = "OK"
    if padding_fraction > max_padding_fraction:
        reason = "EXCESSIVE_PADDING"
    elif metrics["blur_score"] < min_blur:
        reason = "TOO_BLURRY"
    elif (
        metrics["brightness_score"] < min_brightness
        or metrics["dark_pixel_fraction"] > max_dark_fraction
    ):
        reason = "TOO_DARK"
    elif (
        metrics["brightness_score"] > max_brightness
        or metrics["bright_pixel_fraction"] > max_bright_fraction
    ):
        reason = "OVEREXPOSED"

    return EyeCropObservation(
        image=image,
        valid=reason == "OK",
        reason=reason,
        eye_width=eye_width,
        rotation_degrees=angle,
        padding_fraction=padding_fraction,
        blur_score=metrics["blur_score"],
        brightness_score=metrics["brightness_score"],
        dark_pixel_fraction=metrics["dark_pixel_fraction"],
        bright_pixel_fraction=metrics["bright_pixel_fraction"],
    )


def eye_corners_from_106(
    landmarks: np.ndarray,
    indices: Sequence[int],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return robust horizontal eye corners from one 2d106det eye group."""

    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return None
    selected = []
    for index in indices:
        if 0 <= index < len(points) and np.isfinite(points[index]).all():
            selected.append(points[index])
    if len(selected) < 6:
        return None
    eye = np.asarray(selected, dtype=np.float32)
    # The ten-point groups include two pupil/centre duplicates.  Horizontal
    # extrema remain the stable anatomical corners without relying on ordering.
    left = eye[int(np.argmin(eye[:, 0]))]
    right = eye[int(np.argmax(eye[:, 0]))]
    return (float(left[0]), float(left[1])), (float(right[0]), float(right[1]))


def eye_crop_quality(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return {
        "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness_score": float(np.mean(gray)),
        "dark_pixel_fraction": float(np.mean(gray <= 12)),
        "bright_pixel_fraction": float(np.mean(gray >= 245)),
    }


def _square_crop_with_reflect_padding(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    side: float,
) -> tuple[np.ndarray | None, float]:
    side_px = max(1, int(round(side)))
    x1 = int(round(center_x - side_px * 0.5))
    y1 = int(round(center_y - side_px * 0.5))
    x2 = x1 + side_px
    y2 = y1 + side_px
    height, width = image.shape[:2]
    ix1, iy1 = max(0, x1), max(0, y1)
    ix2, iy2 = min(width, x2), min(height, y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None, 1.0
    crop = image[iy1:iy2, ix1:ix2]
    top, bottom = max(0, -y1), max(0, y2 - height)
    left, right = max(0, -x1), max(0, x2 - width)
    if top or bottom or left or right:
        crop = cv2.copyMakeBorder(
            crop,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REFLECT_101,
        )
    if crop.shape[:2] != (side_px, side_px):
        return None, 1.0
    visible_area = float((ix2 - ix1) * (iy2 - iy1))
    padding_fraction = 1.0 - visible_area / float(side_px * side_px)
    return crop, max(0.0, min(1.0, padding_fraction))
