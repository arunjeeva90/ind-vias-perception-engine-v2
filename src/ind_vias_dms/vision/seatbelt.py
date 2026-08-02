from __future__ import annotations

import cv2
import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import SeatbeltAuthenticity
from ind_vias_dms.vision.onnx_classifier import ONNXImageClassifier


class SeatbeltDetectionPlaceholder:
    """Optional explicit torso classifier with UNKNOWN-safe fallback.

    The historical class name is retained for import compatibility. When the
    backend is disabled or unavailable it behaves exactly like the original
    placeholder.
    """

    def __init__(self, config: DMSConfig | None = None) -> None:
        cfg = dict(config.seatbelt_detection or {}) if config is not None else {}
        self.classifier = ONNXImageClassifier(cfg)
        self.min_confidence = float(cfg.get("min_confidence", 0.72))
        self.min_blur = float(cfg.get("min_blur", 24.0))
        self.min_brightness = float(cfg.get("min_brightness", 25.0))
        self.max_brightness = float(cfg.get("max_brightness", 235.0))
        self.confirm_ms = int(cfg.get("confirm_ms", 1200))
        self.candidate_label: str | None = None
        self.candidate_since_ms: int | None = None
        self.confirmed_label: str | None = None
        self.last_status = self.classifier.backend_status
        self.last_torso_bbox: tuple[int, int, int, int] | None = None

    def process(
        self,
        frame: object,
        face_bbox: tuple[int, int, int, int] | None = None,
        timestamp_ms: int = 0,
    ) -> SeatbeltAuthenticity:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return self._unknown("INVALID_FRAME")
        if not self.classifier.ready:
            return self._unknown(self.classifier.backend_status)
        torso_box = face_to_torso_box(face_bbox, frame.shape) if face_bbox else None
        self.last_torso_bbox = torso_box
        if torso_box is None:
            return self._unknown("INVALID_TORSO_ROI")
        x1, y1, x2, y2 = torso_box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or not self._quality_ok(crop):
            return self._unknown("LOW_QUALITY_TORSO_ROI")

        prediction = self.classifier.predict(crop)
        self.last_status = prediction.backend_status
        if prediction.backend_status != "OK" or prediction.confidence < self.min_confidence:
            return self._unknown("LOW_CLASSIFIER_CONFIDENCE")
        if prediction.label not in {"no_seat_belt", "seat_belt_on"}:
            return self._unknown("UNSUPPORTED_CLASS")

        if prediction.label != self.candidate_label:
            self.candidate_label = prediction.label
            self.candidate_since_ms = timestamp_ms
        elapsed = (
            timestamp_ms - self.candidate_since_ms
            if self.candidate_since_ms is not None
            else 0
        )
        if elapsed < self.confirm_ms:
            self.last_status = "TEMPORAL_CONFIRMING"
            return SeatbeltAuthenticity(
                visual_belt_path="CANDIDATE",
                final_state="UNKNOWN",
                confidence=prediction.confidence,
            )

        self.confirmed_label = prediction.label
        self.last_status = "OK"
        if prediction.label == "seat_belt_on":
            return SeatbeltAuthenticity(
                visual_belt_path="WORN",
                final_state="WORN",
                confidence=prediction.confidence,
            )
        return SeatbeltAuthenticity(
            visual_belt_path="NOT_WORN",
            final_state="NOT_WORN",
            confidence=prediction.confidence,
        )

    def _unknown(self, status: str) -> SeatbeltAuthenticity:
        self.last_status = status
        self.candidate_label = None
        self.candidate_since_ms = None
        return SeatbeltAuthenticity()

    def _quality_ok(self, crop: np.ndarray) -> bool:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        return (
            blur >= self.min_blur
            and self.min_brightness <= brightness <= self.max_brightness
        )


def face_to_torso_box(
    face_bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    """Convert the reviewed handoff face geometry into a driver-torso ROI."""

    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = [float(value) for value in face_bbox]
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    center_x = (x1 + x2) * 0.5
    torso_width = max(face_width * 3.2, frame_width * 0.18)
    torso = (
        center_x - torso_width * 0.5,
        y1 + face_height * 0.52,
        center_x + torso_width * 0.5,
        y2 + face_height * 4.25,
    )
    tx1 = max(0, min(frame_width, int(round(torso[0]))))
    ty1 = max(0, min(frame_height, int(round(torso[1]))))
    tx2 = max(0, min(frame_width, int(round(torso[2]))))
    ty2 = max(0, min(frame_height, int(round(torso[3]))))
    if tx2 - tx1 < 96 or ty2 - ty1 < 120:
        return None
    aspect = (tx2 - tx1) / max(1, ty2 - ty1)
    if aspect < 0.42 or aspect > 1.45:
        return None
    return tx1, ty1, tx2, ty2
