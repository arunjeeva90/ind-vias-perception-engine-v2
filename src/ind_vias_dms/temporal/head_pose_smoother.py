from __future__ import annotations

from ind_vias_dms.vision.head_pose import HeadPose


class HeadPoseSmoother:
    def __init__(
        self,
        alpha: float,
        outlier_threshold_deg: float,
        min_confidence: float,
    ) -> None:
        self.alpha = min(1.0, max(0.0, alpha))
        self.outlier_threshold_deg = outlier_threshold_deg
        self.min_confidence = min_confidence
        self._yaw: float | None = None
        self._pitch: float | None = None
        self._roll: float | None = None

    def reset(self) -> None:
        self._yaw = None
        self._pitch = None
        self._roll = None

    def update(self, pose: HeadPose) -> HeadPose:
        if pose.confidence < self.min_confidence:
            return HeadPose(confidence=0.0)
        if self._is_outlier(pose):
            return self._current_pose_like(pose, confidence=min(pose.confidence, 0.2))
        if self._yaw is None or self._pitch is None or self._roll is None:
            self._yaw = pose.yaw_deg
            self._pitch = pose.pitch_deg
            self._roll = pose.roll_deg
        else:
            self._yaw = self._ema(self._yaw, pose.yaw_deg)
            self._pitch = self._ema(self._pitch, pose.pitch_deg)
            self._roll = self._ema(self._roll, pose.roll_deg)
        return self._current_pose_like(pose, confidence=pose.confidence)

    def _ema(self, previous: float, current: float) -> float:
        return (self.alpha * current) + ((1.0 - self.alpha) * previous)

    def _is_outlier(self, pose: HeadPose) -> bool:
        values = (pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
        if any(abs(value) > self.outlier_threshold_deg for value in values):
            return True
        if self._yaw is None or self._pitch is None or self._roll is None:
            return False
        deltas = (
            abs(pose.yaw_deg - self._yaw),
            abs(pose.pitch_deg - self._pitch),
            abs(pose.roll_deg - self._roll),
        )
        return any(delta > self.outlier_threshold_deg for delta in deltas)

    def _current_pose_like(self, pose: HeadPose, confidence: float) -> HeadPose:
        if self._yaw is None or self._pitch is None or self._roll is None:
            return HeadPose(confidence=0.0)
        return HeadPose(
            yaw_deg=self._yaw,
            pitch_deg=self._pitch,
            roll_deg=self._roll,
            rvec=pose.rvec,
            tvec=pose.tvec,
            camera_matrix=pose.camera_matrix,
            dist_coeffs=pose.dist_coeffs,
            confidence=confidence,
        )
