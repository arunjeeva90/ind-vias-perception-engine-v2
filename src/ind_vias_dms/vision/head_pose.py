from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HeadPose:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    confidence: float = 0.0


def normalize_angle_deg(angle: float) -> float:
    normalized = (angle + 180.0) % 360.0 - 180.0
    if normalized == -180.0:
        return 180.0
    return normalized


def _fold_frontal_angle(angle: float) -> tuple[float, bool]:
    normalized = normalize_angle_deg(angle)
    folded = False
    if normalized > 90.0:
        normalized -= 180.0
        folded = True
    elif normalized < -90.0:
        normalized += 180.0
        folded = True
    return normalized, folded


class HeadPoseEstimator:
    LANDMARK_IDS = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 33,
        "right_eye_outer": 263,
        "left_mouth": 61,
        "right_mouth": 291,
    }

    MODEL_POINTS = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )

    def estimate(
        self,
        landmarks_px: dict[int, tuple[float, float]] | None,
        frame_shape: tuple[int, int, int],
    ) -> HeadPose:
        if not landmarks_px:
            return HeadPose()
        try:
            image_points = np.array(
                [landmarks_px[idx] for idx in self.LANDMARK_IDS.values()],
                dtype=np.float64,
            )
        except KeyError:
            return HeadPose()
        height, width = frame_shape[:2]
        focal = float(width)
        camera_matrix = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return HeadPose()
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
        raw_pitch, raw_yaw, raw_roll = (float(a) for a in angles)
        pitch, pitch_folded = _fold_frontal_angle(raw_pitch)
        yaw, yaw_folded = _fold_frontal_angle(raw_yaw)
        roll, roll_folded = _fold_frontal_angle(raw_roll)
        confidence = 0.8
        if pitch_folded or yaw_folded or roll_folded:
            confidence = 0.55
        if abs(pitch) > 90.0 or abs(roll) > 90.0:
            pitch = max(-90.0, min(90.0, pitch))
            roll = max(-90.0, min(90.0, roll))
            confidence = min(confidence, 0.25)
        return HeadPose(
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            confidence=confidence,
        )
