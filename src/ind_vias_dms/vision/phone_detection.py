from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import dist
from typing import Any

import cv2
import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import GazeZone, PlaceholderState


class MobileDistractionState(str, Enum):
    NO_MOBILE_DISTRACTION = "NO_PHONE"
    PHONE_TO_EAR_SUSPECTED = "PHONE_TO_EAR_SUSPECTED"
    PHONE_DOWN_SUSPECTED = "PHONE_DOWN_SUSPECTED"
    TEXTING_SUSPECTED = "TEXTING_SUSPECTED"
    HAND_NEAR_FACE = "HAND_NEAR_FACE"
    UNKNOWN = "UNKNOWN"


@dataclass
class HandContext:
    near_face: bool = False
    near_ear: bool = False
    lower_region: bool = False
    confidence: float = 0.0


class MobileDistractionEstimator:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.down_since_ms: int | None = None
        self.phone_to_ear_since_ms: int | None = None
        self.texting_since_ms: int | None = None
        self._hands: Any | None = None
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
        self._update_timer("phone_to_ear_since_ms", timestamp_ms, hand.near_ear)
        self._update_timer("texting_since_ms", timestamp_ms, gaze_down and hand.lower_region)

        if self._elapsed(self.texting_since_ms, timestamp_ms) >= self.config.texting_sustain_ms:
            return PlaceholderState(MobileDistractionState.TEXTING_SUSPECTED.value, 0.8)
        if (
            self._elapsed(self.phone_to_ear_since_ms, timestamp_ms)
            >= self.config.phone_to_ear_sustain_ms
        ):
            return PlaceholderState(MobileDistractionState.PHONE_TO_EAR_SUSPECTED.value, 0.78)
        if self._elapsed(self.down_since_ms, timestamp_ms) >= self.config.phone_down_sustain_ms:
            return PlaceholderState(MobileDistractionState.PHONE_DOWN_SUSPECTED.value, 0.65)
        if hand.near_face:
            return PlaceholderState(MobileDistractionState.HAND_NEAR_FACE.value, 0.45)
        return PlaceholderState(MobileDistractionState.NO_MOBILE_DISTRACTION.value, 0.6)

    def close(self) -> None:
        if self._hands is not None:
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
        anchors = self._face_anchors(face_bbox, face_landmarks)
        context = HandContext()
        for hand_landmarks in result.multi_hand_landmarks:
            points = [(lm.x * width, lm.y * height) for lm in hand_landmarks.landmark]
            wrist = points[0]
            index_tip = points[8]
            hand_center = (
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            )
            if min(dist(hand_center, anchor) for anchor in anchors["face"]) <= threshold:
                context.near_face = True
            if min(dist(hand_center, anchor) for anchor in anchors["ear"]) <= threshold:
                context.near_ear = True
            if wrist[1] > y2 and index_tip[1] > y1 + face_h * 0.5:
                context.lower_region = True
            context.confidence = max(context.confidence, 0.65)
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

    def _update_timer(self, attr: str, timestamp_ms: int, active: bool) -> None:
        if active and getattr(self, attr) is None:
            setattr(self, attr, timestamp_ms)
        elif not active:
            setattr(self, attr, None)

    @staticmethod
    def _elapsed(since_ms: int | None, timestamp_ms: int) -> int:
        return 0 if since_ms is None else max(0, timestamp_ms - since_ms)


PhoneDetectionPlaceholder = MobileDistractionEstimator
