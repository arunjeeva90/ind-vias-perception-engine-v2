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
    reason_codes: str = "ok"
    feature_count: int = 0
    roi_shape: str = ""
    downscale_factor: float = 1.0


class OpticalFlowYawDetector:
    def __init__(
        self,
        min_flow_points: int = 25,
        median_dx_threshold: float = 2.0,
        yaw_score_threshold: float = 0.55,
        smoothing_window: int = 5,
        required_turning_frames: int = 3,
        max_feature_width: int = 640,
        max_feature_height: int = 640,
        max_corners: int = 120,
        quality_level: float = 0.01,
        min_distance: float = 12.0,
        block_size: int = 7,
        roi_top_ratio: float = 0.35,
        roi_bottom_ratio: float = 0.90,
    ):
        self.min_flow_points = min_flow_points
        self.median_dx_threshold = median_dx_threshold
        self.yaw_score_threshold = yaw_score_threshold
        self.smoothing_window = smoothing_window
        self.required_turning_frames = required_turning_frames
        self.max_feature_width = max_feature_width
        self.max_feature_height = max_feature_height
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.roi_top_ratio = roi_top_ratio
        self.roi_bottom_ratio = roi_bottom_ratio
        self._history: deque[YawDetectionResult] = deque(maxlen=smoothing_window)
        self._prev_gray: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> YawDetectionResult:
        gray, roi_shape, downscale = preprocess_frame_for_flow(
            frame,
            self.max_feature_width,
            self.max_feature_height,
            self.roi_top_ratio,
            self.roi_bottom_ratio,
        )
        if gray is None:
            self._prev_gray = None
            return self._smooth(
                YawDetectionResult(
                    False,
                    0.0,
                    0.0,
                    0,
                    ego_motion_state="uncertain",
                    reason_codes="ego_motion_feature_failure",
                    roi_shape=roi_shape,
                    downscale_factor=downscale,
                )
            )
        if self._prev_gray is None:
            self._prev_gray = gray
            return self._smooth(
                YawDetectionResult(
                    False,
                    0.0,
                    0.0,
                    0,
                    reason_codes="initial_frame",
                    roi_shape=roi_shape,
                    downscale_factor=downscale,
                )
            )
        if self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return self._smooth(
                YawDetectionResult(
                    False,
                    0.0,
                    0.0,
                    0,
                    ego_motion_state="uncertain",
                    reason_codes="ego_motion_feature_failure",
                    roi_shape=roi_shape,
                    downscale_factor=downscale,
                )
            )

        try:
            prev_pts = cv2.goodFeaturesToTrack(
                self._prev_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
            )
        except cv2.error:
            self._prev_gray = gray
            return self._failure("opencv_memory_error", roi_shape, downscale)
        if prev_pts is None:
            self._prev_gray = gray
            return self._failure("ego_motion_feature_failure", roi_shape, downscale)

        feature_count = int(prev_pts.shape[0])
        try:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, prev_pts, None)
        except cv2.error:
            self._prev_gray = gray
            return self._failure("ego_motion_feature_failure", roi_shape, downscale, feature_count)
        self._prev_gray = gray
        if next_pts is None or status is None:
            return self._failure("ego_motion_feature_failure", roi_shape, downscale, feature_count)
        measurement = analyze_flow(
            prev_pts,
            next_pts,
            status,
            min_flow_points=self.min_flow_points,
            median_dx_threshold=self.median_dx_threshold,
            yaw_score_threshold=self.yaw_score_threshold,
        )
        measurement = YawDetectionResult(
            measurement.turning_detected,
            measurement.yaw_score,
            measurement.median_dx,
            measurement.flow_points,
            reason_codes=measurement.reason_codes,
            feature_count=feature_count,
            roi_shape=roi_shape,
            downscale_factor=downscale,
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
            reason_codes=measurement.reason_codes,
            feature_count=measurement.feature_count,
            roi_shape=measurement.roi_shape,
            downscale_factor=measurement.downscale_factor,
        )

    def _failure(
        self,
        reason: str,
        roi_shape: str,
        downscale: float,
        feature_count: int = 0,
    ) -> YawDetectionResult:
        return self._smooth(
            YawDetectionResult(
                False,
                0.0,
                0.0,
                0,
                ego_motion_state="uncertain",
                yaw_confidence=0.0,
                reason_codes=reason,
                feature_count=feature_count,
                roi_shape=roi_shape,
                downscale_factor=downscale,
            )
        )


def preprocess_frame_for_flow(
    frame: np.ndarray,
    max_width: int,
    max_height: int,
    roi_top_ratio: float,
    roi_bottom_ratio: float,
) -> tuple[np.ndarray | None, str, float]:
    if frame is None or frame.size == 0:
        return None, "", 1.0
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        return None, "", 1.0
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    height, width = gray.shape[:2]
    top = int(max(0, min(height - 1, round(height * roi_top_ratio))))
    bottom = int(max(top + 1, min(height, round(height * roi_bottom_ratio))))
    roi = gray[top:bottom, :]
    if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
        return None, f"{roi.shape[0]}x{roi.shape[1]}" if roi.ndim == 2 else "", 1.0
    scale = min(max_width / roi.shape[1], max_height / roi.shape[0], 1.0)
    if scale < 1.0:
        roi = cv2.resize(
            roi,
            (max(1, int(round(roi.shape[1] * scale))), max(1, int(round(roi.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    roi = np.ascontiguousarray(roi, dtype=np.uint8)
    return roi, f"{roi.shape[0]}x{roi.shape[1]}", scale


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
