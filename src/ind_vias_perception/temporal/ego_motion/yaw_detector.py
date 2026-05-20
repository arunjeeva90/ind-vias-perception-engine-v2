from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class YawDetectionResult:
    turning_detected: bool
    yaw_score: float
    median_dx: float
    flow_points: int
    ego_motion_state: str = "straight"
    yaw_confidence: float = 0.0
    turning_confirmation_count: int = 0


class OpticalFlowYawDetector:
    def __init__(
        self,
        min_flow_points: int = 25,
        median_dx_threshold: float = 2.0,
        yaw_score_threshold: float = 0.55,
        smoothing_window: int = 5,
        required_turning_frames: int = 3,
    ):
        self.min_flow_points = min_flow_points
        self.median_dx_threshold = median_dx_threshold
        self.yaw_score_threshold = yaw_score_threshold
        self.smoothing_window = smoothing_window
        self.required_turning_frames = required_turning_frames
        self._history: deque[YawDetectionResult] = deque(maxlen=smoothing_window)
        self._prev_gray: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> YawDetectionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return self._smooth(YawDetectionResult(False, 0.0, 0.0, 0))

        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=7,
        )
        if prev_pts is None:
            self._prev_gray = gray
            return self._smooth(YawDetectionResult(False, 0.0, 0.0, 0))

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, prev_pts, None)
        self._prev_gray = gray
        if next_pts is None or status is None:
            return self._smooth(YawDetectionResult(False, 0.0, 0.0, 0))
        measurement = analyze_flow(
            prev_pts,
            next_pts,
            status,
            min_flow_points=self.min_flow_points,
            median_dx_threshold=self.median_dx_threshold,
            yaw_score_threshold=self.yaw_score_threshold,
        )
        return self._smooth(measurement)

    def update_from_measurement(self, measurement: YawDetectionResult) -> YawDetectionResult:
        return self._smooth(measurement)

    def _smooth(self, measurement: YawDetectionResult) -> YawDetectionResult:
        self._history.append(measurement)
        turning_count = sum(1 for item in self._history if item.turning_detected)
        recent = list(self._history)
        stable_signs = [np.sign(item.median_dx) for item in recent if item.turning_detected]
        stable = len(set(stable_signs)) <= 1 if stable_signs else False
        confidence = turning_count / max(self.required_turning_frames, 1)
        confidence = max(0.0, min(1.0, confidence))

        if measurement.flow_points < self.min_flow_points:
            state = "uncertain"
            confidence = 0.0
        elif turning_count >= self.required_turning_frames and stable:
            state = "turning"
        elif any(item.turning_detected for item in recent):
            state = "uncertain"
        else:
            state = "straight"

        return YawDetectionResult(
            turning_detected=state == "turning",
            yaw_score=measurement.yaw_score,
            median_dx=measurement.median_dx,
            flow_points=measurement.flow_points,
            ego_motion_state=state,
            yaw_confidence=confidence,
            turning_confirmation_count=turning_count,
        )


def analyze_flow(
    prev_pts: np.ndarray,
    next_pts: np.ndarray,
    status: np.ndarray,
    min_flow_points: int = 25,
    median_dx_threshold: float = 2.0,
    yaw_score_threshold: float = 0.55,
) -> YawDetectionResult:
    valid = status.reshape(-1).astype(bool)
    if not np.any(valid):
        return YawDetectionResult(False, 0.0, 0.0, 0)
    prev = prev_pts.reshape(-1, 2)[valid]
    nxt = next_pts.reshape(-1, 2)[valid]
    dx = nxt[:, 0] - prev[:, 0]
    flow_points = int(dx.size)
    if flow_points == 0:
        return YawDetectionResult(False, 0.0, 0.0, 0)

    median_dx = float(np.median(dx))
    dominant_sign = 1.0 if median_dx >= 0 else -1.0
    same_direction = np.sign(dx) == dominant_sign
    strong_motion = np.abs(dx) >= median_dx_threshold
    yaw_score = float(np.mean(same_direction & strong_motion))
    turning = (
        flow_points >= min_flow_points
        and abs(median_dx) >= median_dx_threshold
        and yaw_score >= yaw_score_threshold
    )
    return YawDetectionResult(turning, yaw_score, median_dx, flow_points)
