from __future__ import annotations

from dataclasses import dataclass
from math import dist


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


@dataclass
class EyeState:
    openness: float = 0.0
    is_closed: bool = False
    confidence: float = 0.0
    left_eye_points: list[tuple[int, int]] | None = None
    right_eye_points: list[tuple[int, int]] | None = None


class EyeStateEstimator:
    def __init__(self, closed_threshold: float) -> None:
        self.closed_threshold = closed_threshold

    def estimate(self, landmarks_px: dict[int, tuple[float, float]] | None) -> EyeState:
        if not landmarks_px:
            return EyeState()
        try:
            left = [landmarks_px[i] for i in LEFT_EYE]
            right = [landmarks_px[i] for i in RIGHT_EYE]
        except KeyError:
            return EyeState()
        left_ear = _ear(left)
        right_ear = _ear(right)
        openness = (left_ear + right_ear) / 2.0
        return EyeState(
            openness=openness,
            is_closed=openness < self.closed_threshold,
            confidence=0.85,
            left_eye_points=[(int(x), int(y)) for x, y in left],
            right_eye_points=[(int(x), int(y)) for x, y in right],
        )


def _ear(points: list[tuple[float, float]]) -> float:
    vertical = dist(points[1], points[5]) + dist(points[2], points[4])
    horizontal = 2.0 * dist(points[0], points[3])
    if horizontal <= 1e-6:
        return 0.0
    return vertical / horizontal
