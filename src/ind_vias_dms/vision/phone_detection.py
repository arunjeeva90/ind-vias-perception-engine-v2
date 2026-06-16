from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import dist
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import GazeZone, PlaceholderState
from ind_vias_dms.vision.face_landmarks import FaceLandmarkResult


class MobileDistractionState(str, Enum):
    NO_MOBILE_DISTRACTION = "NO_PHONE"
    SELF_TOUCH_TRANSIENT = "SELF_TOUCH_TRANSIENT"
    EAR_SCRATCH_GESTURE = "EAR_SCRATCH_GESTURE"
    FACE_TOUCH_GROOMING = "FACE_TOUCH_GROOMING"
    PHONE_TO_EAR_CANDIDATE = "PHONE_TO_EAR_CANDIDATE"
    PHONE_TO_EAR_SUSPECTED = "PHONE_TO_EAR_SUSPECTED"
    PHONE_TO_EAR_CONFIRMED = "PHONE_TO_EAR_CONFIRMED"
    PHONE_DOWN_SUSPECTED = "PHONE_DOWN_SUSPECTED"
    TEXTING_SUSPECTED = "TEXTING_SUSPECTED"
    HAND_NEAR_FACE = "HAND_NEAR_FACE"
    UNKNOWN = "UNKNOWN"


@dataclass
class HandContext:
    near_face: bool = False
    near_ear: bool = False
    lower_region: bool = False
    ear_side: str = "UNKNOWN"
    hand_bbox_norm: tuple[float, float, float, float] | None = None
    confidence: float = 0.0


@dataclass
class PhoneObjectResult:
    detected: bool = False
    bbox_norm: tuple[float, float, float, float] | None = None
    confidence: float = 0.0
    region: str = "UNKNOWN"
    backend_status: str = "NOT_CONFIGURED"


class MobileDistractionEstimator:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.down_since_ms: int | None = None
        self.phone_to_ear_since_ms: int | None = None
        self.texting_since_ms: int | None = None
        self._last_ear_hand_center: tuple[float, float] | None = None
        self._last_ear_motion_px: float = 0.0
        self._hands: Any | None = None
        self.last_phone_object = PhoneObjectResult()
        self._phone_object_cfg = config.phone_object_detection or {}
        self._phone_object_enabled = bool(self._phone_object_cfg.get("enabled", False))
        self._phone_object_model_path = str(self._phone_object_cfg.get("model_path", ""))
        self._phone_object_allow_missing = bool(self._phone_object_cfg.get("allow_missing_model", True))
        if self._phone_object_enabled:
            self._init_phone_object_backend()
        if not config.mobile_distraction_enabled:
            return
        try:
            import mediapipe as mp  # type: ignore
        except ImportError:
            return
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    def process(
        self,
        frame: np.ndarray,
        face_bbox: tuple[int, int, int, int] | None,
        face_landmarks: dict[int, tuple[float, float]] | None,
        gaze_zone: GazeZone,
        timestamp_ms: int,
    ) -> PlaceholderState:
        if not self.config.mobile_distraction_enabled:
            return PlaceholderState(state=MobileDistractionState.UNKNOWN.value, confidence=0.0)

        hand = self._estimate_hand_context(frame, face_bbox, face_landmarks)
        gaze_down = gaze_zone in {GazeZone.DOWN, GazeZone.PHONE_DOWN}
        self._update_timer("down_since_ms", timestamp_ms, gaze_down)
        self._update_timer(
            "phone_to_ear_since_ms",
            timestamp_ms,
            hand.near_ear and hand.confidence >= self.config.phone_to_ear_min_hand_confidence,
        )
        self._update_timer("texting_since_ms", timestamp_ms, gaze_down and hand.lower_region)

        return self._classify_from_context(hand, gaze_down, timestamp_ms)

    def _classify_from_context(
        self,
        hand: HandContext,
        gaze_down: bool,
        timestamp_ms: int,
    ) -> PlaceholderState:
        if self._elapsed(self.texting_since_ms, timestamp_ms) >= self.config.texting_sustain_ms:
            return PlaceholderState(MobileDistractionState.TEXTING_SUSPECTED.value, 0.8)
        phone_to_ear_ms = self._elapsed(self.phone_to_ear_since_ms, timestamp_ms)
        if hand.near_ear and phone_to_ear_ms < self.config.hand_to_ear_transient_ms:
            return PlaceholderState(MobileDistractionState.SELF_TOUCH_TRANSIENT.value, 0.45)
        if (
            hand.near_ear
            and self.config.ear_scratch_motion_filter_enabled
            and phone_to_ear_ms < self.config.ear_scratch_max_warning_ms
            and self.config.ear_scratch_local_motion_min_px
            <= getattr(self, "_last_ear_motion_px", 0.0)
            <= self.config.ear_scratch_local_motion_max_px
        ):
            return PlaceholderState(MobileDistractionState.EAR_SCRATCH_GESTURE.value, 0.5)
        if (
            phone_to_ear_ms >= self.config.phone_to_ear_confirmed_ms
            and self.last_phone_object.detected
            and self.last_phone_object.region == "DRIVER_FACE_NEAR_EAR"
        ):
            return PlaceholderState(MobileDistractionState.PHONE_TO_EAR_CONFIRMED.value, 0.92)
        if (
            phone_to_ear_ms >= self.config.phone_to_ear_suspected_ms
            and (
                self.last_phone_object.detected
                or not self.config.phone_to_ear_requires_phone_object_for_confirmed
            )
        ):
            return PlaceholderState(
                MobileDistractionState.PHONE_TO_EAR_SUSPECTED.value,
                max(0.78, min(0.95, hand.confidence)),
            )
        if phone_to_ear_ms >= self.config.phone_to_ear_candidate_ms:
            return PlaceholderState(MobileDistractionState.PHONE_TO_EAR_CANDIDATE.value, 0.58)
        if self._elapsed(self.down_since_ms, timestamp_ms) >= self.config.phone_down_suspect_ms:
            return PlaceholderState(MobileDistractionState.PHONE_DOWN_SUSPECTED.value, 0.65)
        if hand.near_face:
            return PlaceholderState(MobileDistractionState.HAND_NEAR_FACE.value, 0.45)
        return PlaceholderState(MobileDistractionState.NO_MOBILE_DISTRACTION.value, 0.6)

    def process_cabin(
        self,
        frame: np.ndarray,
        faces: list[tuple[int, str, FaceLandmarkResult | None]],
        driver_track_id: int | None,
        gaze_zone: GazeZone,
        timestamp_ms: int,
    ) -> tuple[PlaceholderState, list[str]]:
        driver_state = PlaceholderState(MobileDistractionState.UNKNOWN.value, 0.0)
        cabin_events: list[str] = []
        self.last_phone_object = self._detect_phone_object(frame)
        ordered_faces = sorted(faces, key=lambda item: item[0] != driver_track_id)
        for track_id, zone, face in ordered_faces:
            if face is None:
                continue
            state = self.process(
                frame,
                face.bbox,
                face.landmarks_px,
                gaze_zone if track_id == driver_track_id else GazeZone.UNKNOWN,
                timestamp_ms,
            )
            if track_id == driver_track_id:
                driver_state = state
            elif state.state == MobileDistractionState.PHONE_TO_EAR_SUSPECTED.value:
                cabin_events.append("PASSENGER_PHONE_TO_EAR")
            elif state.state in {
                MobileDistractionState.TEXTING_SUSPECTED.value,
                MobileDistractionState.PHONE_DOWN_SUSPECTED.value,
            }:
                cabin_events.append(f"{zone}_{state.state}")
        return driver_state, cabin_events

    def _init_phone_object_backend(self) -> None:
        if not self._phone_object_model_path:
            self.last_phone_object = PhoneObjectResult(backend_status="MODEL_MISSING")
            return
        if not Path(self._phone_object_model_path).exists():
            if self._phone_object_allow_missing:
                self.last_phone_object = PhoneObjectResult(backend_status="MODEL_MISSING")
                return
            raise RuntimeError(f"Phone object detector model not found: {self._phone_object_model_path}")
        # v0.2.3 keeps the object backend optional; posture detection continues until
        # a reviewed ONNX phone detector is supplied.
        self.last_phone_object = PhoneObjectResult(backend_status="MODEL_PRESENT_NOT_LOADED")

    def _detect_phone_object(self, frame: np.ndarray) -> PhoneObjectResult:
        if not getattr(self, "_phone_object_enabled", False):
            return PhoneObjectResult(backend_status="NOT_CONFIGURED")
        last = getattr(self, "last_phone_object", PhoneObjectResult())
        if last.backend_status in {"MODEL_MISSING", "MODEL_PRESENT_NOT_LOADED"}:
            return PhoneObjectResult(backend_status=last.backend_status)
        return PhoneObjectResult(backend_status="NOT_IMPLEMENTED")

    def close(self) -> None:
        if getattr(self, "_hands", None) is not None:
            self._hands.close()

    def _estimate_hand_context(
        self,
        frame: np.ndarray,
        face_bbox: tuple[int, int, int, int] | None,
        face_landmarks: dict[int, tuple[float, float]] | None,
    ) -> HandContext:
        if self._hands is None or face_bbox is None:
            return HandContext()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return HandContext()
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        threshold = max(face_w, face_h) * self.config.hand_near_face_distance_ratio
        ear_threshold_px = max(width, height) * self.config.phone_to_ear_hand_distance_threshold_norm
        anchors = self._face_anchors(face_bbox, face_landmarks)
        side_rois = self._ear_side_rois(face_bbox, width, height)
        context = HandContext()
        for hand_landmarks in result.multi_hand_landmarks:
            points = [(lm.x * width, lm.y * height) for lm in hand_landmarks.landmark]
            wrist = points[0]
            index_tip = points[8]
            hand_center = (
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            )
            hx1, hy1 = min(point[0] for point in points), min(point[1] for point in points)
            hx2, hy2 = max(point[0] for point in points), max(point[1] for point in points)
            hand_bbox_norm = (hx1 / width, hy1 / height, hx2 / width, hy2 / height)
            if min(dist(hand_center, anchor) for anchor in anchors["face"]) <= threshold:
                context.near_face = True
            ear_distance = min(dist(hand_center, anchor) for anchor in anchors["ear"])
            side_hit = self._point_in_roi(hand_center, side_rois["left"]) or self._point_in_roi(hand_center, side_rois["right"])
            wrist_side_hit = self._point_in_roi(wrist, side_rois["left"]) or self._point_in_roi(wrist, side_rois["right"])
            index_side_hit = self._point_in_roi(index_tip, side_rois["left"]) or self._point_in_roi(index_tip, side_rois["right"])
            if ear_distance <= min(threshold, ear_threshold_px) or side_hit or wrist_side_hit or index_side_hit:
                context.near_ear = True
                context.ear_side = "LEFT" if self._point_in_roi(hand_center, side_rois["left"]) else "RIGHT"
                if self._last_ear_hand_center is not None:
                    self._last_ear_motion_px = dist(hand_center, self._last_ear_hand_center)
                self._last_ear_hand_center = hand_center
            if wrist[1] > y2 and index_tip[1] > y1 + face_h * 0.5:
                context.lower_region = True
            context.hand_bbox_norm = hand_bbox_norm
            context.confidence = max(context.confidence, 0.65)
            if context.near_ear:
                context.confidence = max(context.confidence, 0.82)
        if not context.near_ear:
            self._last_ear_hand_center = None
            self._last_ear_motion_px = 0.0
        return context

    def _face_anchors(
        self,
        face_bbox: tuple[int, int, int, int],
        face_landmarks: dict[int, tuple[float, float]] | None,
    ) -> dict[str, list[tuple[float, float]]]:
        x1, y1, x2, y2 = face_bbox
        face_h = y2 - y1
        left_ear = (float(x1), y1 + face_h * 0.45)
        right_ear = (float(x2), y1 + face_h * 0.45)
        mouth = ((x1 + x2) / 2.0, y1 + face_h * 0.72)
        nose = ((x1 + x2) / 2.0, y1 + face_h * 0.48)
        if face_landmarks:
            mouth = face_landmarks.get(13, mouth)
            nose = face_landmarks.get(1, nose)
        return {"face": [left_ear, right_ear, mouth, nose], "ear": [left_ear, right_ear]}

    def _ear_side_rois(
        self,
        face_bbox: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> dict[str, tuple[float, float, float, float]]:
        x1, y1, x2, y2 = face_bbox
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        expand_x = face_w * self.config.phone_to_ear_face_side_roi_expand
        y_top = max(0.0, y1 + face_h * 0.10)
        y_bottom = min(float(height - 1), y1 + face_h * 0.78)
        return {
            "left": (max(0.0, x1 - expand_x), y_top, min(float(width - 1), x1 + face_w * 0.18), y_bottom),
            "right": (max(0.0, x2 - face_w * 0.18), y_top, min(float(width - 1), x2 + expand_x), y_bottom),
        }

    @staticmethod
    def _point_in_roi(point: tuple[float, float], roi: tuple[float, float, float, float]) -> bool:
        return roi[0] <= point[0] <= roi[2] and roi[1] <= point[1] <= roi[3]

    def _update_timer(self, attr: str, timestamp_ms: int, active: bool) -> None:
        if active and getattr(self, attr) is None:
            setattr(self, attr, timestamp_ms)
        elif not active:
            setattr(self, attr, None)

    @staticmethod
    def _elapsed(since_ms: int | None, timestamp_ms: int) -> int:
        return 0 if since_ms is None else max(0, timestamp_ms - since_ms)


PhoneDetectionPlaceholder = MobileDistractionEstimator
