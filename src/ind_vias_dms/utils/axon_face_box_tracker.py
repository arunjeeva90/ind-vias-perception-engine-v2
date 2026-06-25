from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TrackedBox:
    bbox: tuple[int, int, int, int]
    confidence: float
    age_ms: int
    source: str


class AxonFaceBoxTracker:
    """Tiny optical-flow bbox tracker for AXON live display.

    This is not a replacement for DMS inference. It only keeps the visual face
    box alive between slow FaceMesh detections.
    """

    def __init__(
        self,
        *,
        max_hold_ms: int = 1200,
        min_points: int = 6,
        max_jump_ratio: float = 0.35,
    ) -> None:
        self.max_hold_ms = max_hold_ms
        self.min_points = min_points
        self.max_jump_ratio = max_jump_ratio
        self.prev_gray: np.ndarray | None = None
        self.prev_points: np.ndarray | None = None
        self.bbox: tuple[int, int, int, int] | None = None
        self.last_update_ms: int | None = None

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_points = None
        self.bbox = None
        self.last_update_ms = None

    def update_from_detection(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
        timestamp_ms: int,
    ) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if bbox is None:
            if self.prev_gray is None:
                self.prev_gray = gray
            return

        x1, y1, x2, y2 = self._clip_bbox(bbox, frame.shape)
        if x2 <= x1 + 8 or y2 <= y1 + 8:
            self.reset()
            self.prev_gray = gray
            return

        self.bbox = (x1, y1, x2, y2)
        self.last_update_ms = timestamp_ms
        self.prev_gray = gray
        self.prev_points = self._points_from_bbox(gray, self.bbox)

    def track(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
    ) -> TrackedBox | None:
        if self.bbox is None or self.prev_gray is None or self.last_update_ms is None:
            return None

        age_ms = timestamp_ms - self.last_update_ms
        if age_ms < 0 or age_ms > self.max_hold_ms:
            self.reset()
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_points is None or len(self.prev_points) < self.min_points:
            self.prev_points = self._points_from_bbox(self.prev_gray, self.bbox)

        if self.prev_points is None or len(self.prev_points) < self.min_points:
            self.prev_gray = gray
            return TrackedBox(self.bbox, 0.35, age_ms, "hold")

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None,
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
        )

        if next_points is None or status is None:
            self.prev_gray = gray
            return TrackedBox(self.bbox, 0.30, age_ms, "hold")

        good_prev = self.prev_points[status.reshape(-1) == 1]
        good_next = next_points[status.reshape(-1) == 1]

        if len(good_next) < self.min_points:
            self.prev_gray = gray
            return TrackedBox(self.bbox, 0.30, age_ms, "hold")

        motion = good_next.reshape(-1, 2) - good_prev.reshape(-1, 2)
        dx, dy = np.median(motion, axis=0)

        x1, y1, x2, y2 = self.bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        if abs(dx) > self.max_jump_ratio * w or abs(dy) > self.max_jump_ratio * h:
            self.prev_gray = gray
            return TrackedBox(self.bbox, 0.25, age_ms, "hold")

        new_bbox = (
            int(round(x1 + dx)),
            int(round(y1 + dy)),
            int(round(x2 + dx)),
            int(round(y2 + dy)),
        )
        self.bbox = self._clip_bbox(new_bbox, frame.shape)
        self.prev_gray = gray
        self.prev_points = self._points_from_bbox(gray, self.bbox)

        confidence = max(0.25, 0.75 * (1.0 - age_ms / max(1, self.max_hold_ms)))
        return TrackedBox(self.bbox, confidence, age_ms, "flow")

    def _points_from_bbox(
        self,
        gray: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        points = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=40,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=5,
        )
        if points is None:
            return None

        points[:, 0, 0] += x1
        points[:, 0, 1] += y1
        return points.astype(np.float32)

    def _clip_bbox(
        self,
        bbox: tuple[int, int, int, int],
        shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        height, width = shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width - 1, int(x2)))
        y2 = max(0, min(height - 1, int(y2)))
        return x1, y1, x2, y2
