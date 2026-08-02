from __future__ import annotations

from dataclasses import dataclass
from math import dist

import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.vision.eye_crop import EyeCropObservation, aligned_eye_crop
from ind_vias_dms.vision.landmark_106 import (
    compare_eye_geometry,
    create_landmark_106_backend,
)
from ind_vias_dms.vision.onnx_classifier import ONNXImageClassifier


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


@dataclass
class EyeState:
    openness: float = 0.0
    is_closed: bool = False
    confidence: float = 0.0
    left_eye_points: list[tuple[int, int]] | None = None
    right_eye_points: list[tuple[int, int]] | None = None
    classification_source: str = "LANDMARK_EAR"
    model_confidence: float = 0.0
    model_backend_status: str = "DISABLED"
    left_eye_state: str = "UNKNOWN"
    right_eye_state: str = "UNKNOWN"
    left_eye_confidence: float = 0.0
    right_eye_confidence: float = 0.0
    left_eye_quality_status: str = "NOT_EVALUATED"
    right_eye_quality_status: str = "NOT_EVALUATED"
    landmark_106_status: str = "DISABLED"
    landmark_geometry_agreement: bool | None = None
    landmark_106_inference_ms: float = 0.0


class EyeStateEstimator:
    def __init__(self, config: DMSConfig | float) -> None:
        if isinstance(config, DMSConfig):
            self.closed_threshold = config.eye_closed_threshold
            classifier_config = config.eye_state_classifier
        else:
            self.closed_threshold = float(config)
            classifier_config = None
        self.classifier = ONNXImageClassifier(classifier_config)
        cfg = dict(classifier_config or {})
        self.landmark_106 = create_landmark_106_backend(classifier_config)
        self.landmark_106_require_agreement = bool(
            cfg.get("landmark_106_require_agreement", True)
        )
        self.landmark_106_advisory_only = bool(
            cfg.get("landmark_106_advisory_only", True)
        )
        self.landmark_106_max_normalized_error = float(
            cfg.get("landmark_106_max_normalized_error", 0.35)
        )
        self.model_min_confidence = float(cfg.get("min_confidence", 0.70))
        self.crop_mode = str(cfg.get("crop_mode", "aligned_reviewed_v1"))
        self.eye_crop_padding_x = float(cfg.get("crop_padding_x", 0.45))
        self.eye_crop_padding_y = float(cfg.get("crop_padding_y", 1.10))
        self.eye_crop_context_scale = float(cfg.get("crop_context_scale", 1.65))
        self.eye_crop_eyebrow_shift = float(cfg.get("crop_eyebrow_shift", 0.10))
        self.eye_crop_min_width = float(cfg.get("crop_min_eye_width", 18.0))
        self.eye_crop_min_blur = float(cfg.get("crop_min_blur", 12.0))
        self.eye_crop_min_brightness = float(cfg.get("crop_min_brightness", 18.0))
        self.eye_crop_max_brightness = float(cfg.get("crop_max_brightness", 235.0))
        self.eye_crop_max_padding_fraction = float(
            cfg.get("crop_max_padding_fraction", 0.25)
        )
        self.geometry_agreement_enabled = bool(
            cfg.get("geometry_agreement_enabled", True)
        )
        self.geometry_open_margin = float(cfg.get("geometry_open_margin", 1.25))

    def estimate(
        self,
        landmarks_px: dict[int, tuple[float, float]] | None,
        frame: np.ndarray | None = None,
        face_bbox: tuple[int, int, int, int] | None = None,
    ) -> EyeState:
        if not landmarks_px:
            return EyeState(model_backend_status=self.classifier.backend_status)
        try:
            left = [landmarks_px[i] for i in LEFT_EYE]
            right = [landmarks_px[i] for i in RIGHT_EYE]
        except KeyError:
            return EyeState(model_backend_status=self.classifier.backend_status)
        left_ear = _ear(left)
        right_ear = _ear(right)
        openness = (left_ear + right_ear) / 2.0
        state = EyeState(
            openness=openness,
            is_closed=openness < self.closed_threshold,
            confidence=0.85,
            left_eye_points=[(int(x), int(y)) for x, y in left],
            right_eye_points=[(int(x), int(y)) for x, y in right],
            model_backend_status=self.classifier.backend_status,
            landmark_106_status=self.landmark_106.backend_status,
        )
        if frame is not None and self.landmark_106.enabled:
            geometry_106 = self.landmark_106.infer(frame, face_bbox)
            state.landmark_106_status = (
                f"{self.landmark_106.backend_name}_{geometry_106.backend_status}"
            )
            state.landmark_106_inference_ms = geometry_106.inference_ms
        else:
            geometry_106 = None

        if geometry_106 is not None and geometry_106.backend_status == "OK":
            agreement = compare_eye_geometry(
                (right[0], right[3]),  # MediaPipe anatomical left/image-right.
                (left[0], left[3]),  # MediaPipe anatomical right/image-left.
                geometry_106.points_px,
                max_normalized_error=self.landmark_106_max_normalized_error,
            )
            state.landmark_geometry_agreement = agreement.valid
            if agreement.valid:
                state.confidence = max(state.confidence, 0.90)
                state.classification_source = (
                    f"LANDMARK_EAR_{self.landmark_106.backend_name}_AGREEMENT"
                )
            elif (
                not self.landmark_106_advisory_only
                and self.landmark_106_require_agreement
            ):
                state.classification_source = "UNKNOWN_LANDMARK_GEOMETRY_DISAGREEMENT"
                state.confidence = 0.0
                return state

        if frame is None or not self.classifier.ready:
            return state

        per_eye = []
        for side, points, ear in (
            ("left", left, left_ear),
            ("right", right, right_ear),
        ):
            observation = self._eye_crop_observation(frame, points)
            if side == "left":
                state.left_eye_quality_status = observation.reason
            else:
                state.right_eye_quality_status = observation.reason
            if not observation.valid or observation.image is None:
                per_eye.append((side, "UNKNOWN", 0.0, ear))
                continue
            prediction = self.classifier.predict(observation.image)
            eye_label, eye_confidence = self._fuse_single_eye(prediction, ear)
            per_eye.append((side, eye_label, eye_confidence, ear))
            if side == "left":
                state.left_eye_state = eye_label
                state.left_eye_confidence = eye_confidence
            else:
                state.right_eye_state = eye_label
                state.right_eye_confidence = eye_confidence

        usable = [item for item in per_eye if item[1] in {"OPEN", "CLOSED"}]
        if not usable:
            state.model_backend_status = self.classifier.backend_status
            state.classification_source = "UNKNOWN_NO_VALID_EYE_CROP"
            state.confidence = 0.0
            return state

        labels = {item[1] for item in usable}
        state.model_confidence = float(np.mean([item[2] for item in usable]))
        state.model_backend_status = "OK"
        if len(labels) > 1:
            state.classification_source = "UNKNOWN_BILATERAL_DISAGREEMENT"
            state.confidence = 0.0
            return state
        if len(usable) == 1:
            state.classification_source = "ONNX_SINGLE_EYE_QUALITY_GATED"
            # A single visible eye is useful, but must not carry the same
            # confidence as bilateral agreement.
            state.confidence = usable[0][2] * 0.80
        else:
            state.classification_source = "ONNX_BILATERAL_GEOMETRY_AGREEMENT"
            state.confidence = min(item[2] for item in usable)
        state.is_closed = next(iter(labels)) == "CLOSED"
        if state.is_closed:
            state.openness = min(state.openness, self.closed_threshold * 0.70)
        else:
            state.openness = max(state.openness, self.closed_threshold * 1.30)
        return state

    def close(self) -> None:
        self.landmark_106.close()

    def _eye_crop_observation(
        self,
        frame: np.ndarray,
        points: list[tuple[float, float]],
    ) -> EyeCropObservation:
        if self.crop_mode == "legacy_axis_aligned":
            crop = _crop_eye(
                frame,
                points,
                padding_x=self.eye_crop_padding_x,
                padding_y=self.eye_crop_padding_y,
            )
            if crop is None:
                return EyeCropObservation(None, False, "CROP_FAILED")
            return EyeCropObservation(crop, True, "OK")
        return aligned_eye_crop(
            frame,
            points[0],
            points[3],
            image_size=self.classifier.input_width,
            context_scale=self.eye_crop_context_scale,
            eyebrow_shift=self.eye_crop_eyebrow_shift,
            min_eye_width=self.eye_crop_min_width,
            max_padding_fraction=self.eye_crop_max_padding_fraction,
            min_blur=self.eye_crop_min_blur,
            min_brightness=self.eye_crop_min_brightness,
            max_brightness=self.eye_crop_max_brightness,
        )

    def _fuse_single_eye(self, prediction, ear: float) -> tuple[str, float]:
        if prediction.backend_status != "OK" or not prediction.probabilities:
            return "UNKNOWN", 0.0
        closed_probability = float(
            prediction.probabilities.get("eye_closed", 0.0)
        )
        open_probability = float(prediction.probabilities.get("eye_open", 0.0))
        confidence = max(closed_probability, open_probability)
        if confidence < self.model_min_confidence:
            return "UNKNOWN", 0.0
        model_label = "CLOSED" if closed_probability > open_probability else "OPEN"
        if not self.geometry_agreement_enabled:
            return model_label, confidence

        geometry_label = "AMBIGUOUS"
        if ear < self.closed_threshold:
            geometry_label = "CLOSED"
        elif ear > self.closed_threshold * self.geometry_open_margin:
            geometry_label = "OPEN"
        if geometry_label != "AMBIGUOUS" and geometry_label != model_label:
            return "UNKNOWN", 0.0
        geometry_bonus = 0.04 if geometry_label == model_label else 0.0
        return model_label, min(0.99, confidence + geometry_bonus)


def _ear(points: list[tuple[float, float]]) -> float:
    vertical = dist(points[1], points[5]) + dist(points[2], points[4])
    horizontal = 2.0 * dist(points[0], points[3])
    if horizontal <= 1e-6:
        return 0.0
    return vertical / horizontal


def _crop_eye(
    frame: np.ndarray,
    points: list[tuple[float, float]],
    padding_x: float,
    padding_y: float,
) -> np.ndarray | None:
    coordinates = np.asarray(points, dtype=np.float32)
    if coordinates.shape != (6, 2) or not np.isfinite(coordinates).all():
        return None
    height, width = frame.shape[:2]
    x1, y1 = np.min(coordinates, axis=0)
    x2, y2 = np.max(coordinates, axis=0)
    eye_width = float(x2 - x1)
    eye_height = float(y2 - y1)
    if eye_width < 6.0 or eye_height < 2.0:
        return None
    x1 -= eye_width * padding_x
    x2 += eye_width * padding_x
    y1 -= eye_height * padding_y
    y2 += eye_height * padding_y
    ix1 = max(0, min(width - 1, int(np.floor(x1))))
    iy1 = max(0, min(height - 1, int(np.floor(y1))))
    ix2 = max(ix1 + 1, min(width, int(np.ceil(x2))))
    iy2 = max(iy1 + 1, min(height, int(np.ceil(y2))))
    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return None
    return crop
