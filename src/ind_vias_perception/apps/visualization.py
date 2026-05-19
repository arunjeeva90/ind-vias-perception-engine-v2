from __future__ import annotations

import cv2
import numpy as np

from ind_vias_perception.common.types import BBox2D, PerceptionOutput


_BOX_COLOR = (0, 220, 0)
_TEXT_COLOR = (255, 255, 255)
_TEXT_BG_COLOR = (0, 0, 0)
_STATUS_BG_COLOR = (32, 32, 32)


def draw_perception_output(frame: np.ndarray, output: PerceptionOutput) -> np.ndarray:
    annotated = frame.copy()

    for det in output.detections:
        x1, y1, x2, y2 = _clamp_bbox(det.bbox, annotated.shape)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        label = f"{det.label.value} {det.confidence:.2f}"
        if det.distance_m is not None:
            label += f" {det.distance_m:.1f}m"
        if det.ttc_s is not None:
            label += f" TTC {det.ttc_s:.1f}s"
        _draw_label(annotated, label, (x1, max(0, y1 - 8)))

    payload = output.safety_payload
    status_parts = [
        f"warning: {payload.get('warning_level', 'unknown')}",
        f"sentinel: {payload.get('sentinel_state', 'unknown')}",
        f"cais: {payload.get('cais_mode', output.mode)}",
    ]
    _draw_status(annotated, " | ".join(status_parts))
    return annotated


def _clamp_bbox(bbox: BBox2D, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x1 = int(max(0, min(width - 1, round(bbox.x1))))
    y1 = int(max(0, min(height - 1, round(bbox.y1))))
    x2 = int(max(0, min(width - 1, round(bbox.x2))))
    y2 = int(max(0, min(height - 1, round(bbox.y2))))
    return x1, y1, x2, y2


def _draw_label(frame: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - text_h - baseline - 4)
    right = min(frame.shape[1] - 1, x + text_w + 8)
    cv2.rectangle(frame, (x, top), (right, y + baseline), _TEXT_BG_COLOR, -1)
    cv2.putText(frame, text, (x + 4, y - 4), font, scale, _TEXT_COLOR, thickness, cv2.LINE_AA)


def _draw_status(frame: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1] - 1, text_w + 16), text_h + baseline + 14), _STATUS_BG_COLOR, -1)
    cv2.putText(frame, text, (8, text_h + 6), font, scale, _TEXT_COLOR, thickness, cv2.LINE_AA)
