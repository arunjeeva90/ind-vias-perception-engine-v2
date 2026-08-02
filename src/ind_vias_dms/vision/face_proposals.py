from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.utils.mediapipe_loader import load_mediapipe_solutions


@dataclass
class FaceProposal:
    bbox: tuple[int, int, int, int]
    confidence: float
    backend: str
    roi_name: str = "FULL"

    @property
    def area(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])


class FaceProposalDetector:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.backend = "none"
        self._detector: Any | None = None
        self._haar: cv2.CascadeClassifier | None = None
        if config.face_proposal_backend == "mediapipe_face_detection":
            try:
                mp = load_mediapipe_solutions()

                self._detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=config.face_proposal_model_selection,
                    min_detection_confidence=config.face_proposal_min_confidence,
                )
                self.backend = "mediapipe_face_detection"
            except Exception:
                self._detector = None
        if self._detector is None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(str(cascade_path))
            self.backend = "opencv_haar" if not self._haar.empty() else "none"

    def detect(self, frame_bgr: np.ndarray) -> list[FaceProposal]:
        if self.backend == "none":
            return []
        proposals: list[FaceProposal] = []
        for name, roi in self._ordered_rois():
            proposals.extend(self._detect_in_roi(frame_bgr, roi, name))
        return suppress_duplicate_proposals(
            proposals,
            iou_threshold=self.config.duplicate_face_iou_threshold,
            center_distance_threshold=self.config.duplicate_face_center_distance_threshold,
        )[: self.config.max_num_faces]

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()

    def _detect_in_roi(
        self,
        frame_bgr: np.ndarray,
        roi_norm: tuple[float, float, float, float],
        roi_name: str,
    ) -> list[FaceProposal]:
        height, width = frame_bgr.shape[:2]
        rx1, ry1, rx2, ry2 = _norm_to_box(roi_norm, width, height)
        crop = frame_bgr[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return []
        if self.config.face_detection_upscale > 1.0:
            crop_for_detection = cv2.resize(
                crop,
                None,
                fx=self.config.face_detection_upscale,
                fy=self.config.face_detection_upscale,
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            crop_for_detection = crop
        if self._detector is not None:
            return self._detect_mediapipe(crop_for_detection, (rx1, ry1), roi_name)
        if self._haar is not None and not self._haar.empty():
            return self._detect_haar(crop_for_detection, (rx1, ry1), roi_name)
        return []

    def _detect_mediapipe(
        self,
        crop_bgr: np.ndarray,
        offset: tuple[int, int],
        roi_name: str,
    ) -> list[FaceProposal]:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.process(rgb)
        if not result.detections:
            return []
        height, width = crop_bgr.shape[:2]
        scale = max(1.0, self.config.face_detection_upscale)
        proposals = []
        for detection in result.detections:
            rel = detection.location_data.relative_bounding_box
            x1 = int((rel.xmin * width) / scale) + offset[0]
            y1 = int((rel.ymin * height) / scale) + offset[1]
            x2 = int(((rel.xmin + rel.width) * width) / scale) + offset[0]
            y2 = int(((rel.ymin + rel.height) * height) / scale) + offset[1]
            confidence = float(detection.score[0]) if detection.score else 0.0
            proposals.append(FaceProposal((x1, y1, x2, y2), confidence, self.backend, roi_name))
        return proposals

    def _detect_haar(
        self,
        crop_bgr: np.ndarray,
        offset: tuple[int, int],
        roi_name: str,
    ) -> list[FaceProposal]:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        scale = max(1.0, self.config.face_detection_upscale)
        detections = self._haar.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(40, 40))
        proposals = []
        for x, y, w, h in detections:
            x1 = int(x / scale) + offset[0]
            y1 = int(y / scale) + offset[1]
            x2 = int((x + w) / scale) + offset[0]
            y2 = int((y + h) / scale) + offset[1]
            proposals.append(FaceProposal((x1, y1, x2, y2), 0.45, self.backend, roi_name))
        return proposals

    def _ordered_rois(self) -> list[tuple[str, tuple[float, float, float, float]]]:
        return [
            ("DRIVER_ROI", _roi(self.config, "driver_roi_norm")),
            ("FRONT_CABIN", (0.0, 0.08, 1.0, 0.95)),
            ("FRONT_PASSENGER_ROI", _roi(self.config, "front_passenger_roi_norm")),
            ("REAR_LEFT_ROI", _roi(self.config, "rear_left_roi_norm")),
            ("REAR_CENTER_ROI", _roi(self.config, "rear_center_roi_norm")),
            ("REAR_RIGHT_ROI", _roi(self.config, "rear_right_roi_norm")),
        ]


def suppress_duplicate_proposals(
    proposals: list[FaceProposal],
    iou_threshold: float,
    center_distance_threshold: float,
) -> list[FaceProposal]:
    kept: list[FaceProposal] = []
    for proposal in sorted(proposals, key=lambda item: (item.confidence, item.area), reverse=True):
        duplicate = False
        for existing in kept:
            if _iou(proposal.bbox, existing.bbox) > iou_threshold:
                duplicate = True
                break
            if _center_distance_norm(proposal.bbox, existing.bbox) < center_distance_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(proposal)
    return kept


def expand_box(
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
    margin: float,
) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    return (
        max(0, int(x1 - bw * margin)),
        max(0, int(y1 - bh * margin)),
        min(width - 1, int(x2 + bw * margin)),
        min(height - 1, int(y2 + bh * margin)),
    )


def _roi(config: DMSConfig, name: str) -> tuple[float, float, float, float]:
    if config.auto_generate_rois_from_layout:
        return _generated_roi(config, name)
    value = getattr(config, name, None)
    if value:
        return (value["x_min"], value["y_min"], value["x_max"], value["y_max"])
    return _generated_roi(config, name)


def _generated_roi(config: DMSConfig, name: str) -> tuple[float, float, float, float]:
    driver_right = config.driver_image_side.upper() == "RIGHT"
    if config.mirror_input:
        driver_right = not driver_right
    if name == "driver_roi_norm":
        return (0.45, 0.10, 1.00, 0.95) if driver_right else (0.00, 0.10, 0.55, 0.95)
    if name == "front_passenger_roi_norm":
        return (0.00, 0.10, 0.55, 0.95) if driver_right else (0.45, 0.10, 1.00, 0.95)
    if name == "rear_left_roi_norm":
        return (0.00, 0.00, 0.33, 0.70)
    if name == "rear_right_roi_norm":
        return (0.66, 0.00, 1.00, 0.70)
    if name == "rear_center_roi_norm":
        return (0.30, 0.00, 0.70, 0.70)
    return (0.0, 0.0, 1.0, 1.0)


def _norm_to_box(
    roi: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, min(width - 1, int(roi[0] * width))),
        max(0, min(height - 1, int(roi[1] * height))),
        max(0, min(width, int(roi[2] * width))),
        max(0, min(height, int(roi[3] * height))),
    )


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


def _center_distance_norm(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ac = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    bc = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    scale = max(1.0, float(max(a + b)))
    return (((ac[0] - bc[0]) / scale) ** 2 + ((ac[1] - bc[1]) / scale) ** 2) ** 0.5
