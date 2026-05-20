from __future__ import annotations

import math

from ind_vias_perception.common.types import CameraCalibration


def robust_fuse_distance_m(
    ground_m: float,
    semantic_m: float,
    cal: CameraCalibration,
    prefer_semantic: bool = False,
) -> tuple[float, str]:
    ground_ok = math.isfinite(ground_m) and ground_m > 0
    semantic_ok = math.isfinite(semantic_m) and semantic_m > 0
    if ground_ok and semantic_ok:
        ratio = ground_m / semantic_m
        if 0.5 <= ratio <= 2.0:
            return _clamp((ground_m + semantic_m) * 0.5, cal), "fused"
        if prefer_semantic:
            return _clamp(semantic_m, cal), "semantic"
        return _clamp(semantic_m, cal), "semantic"
    if semantic_ok:
        return _clamp(semantic_m, cal), "semantic"
    if ground_ok:
        return _clamp(ground_m, cal), "ground"
    return cal.max_distance_m, "fallback"


def _clamp(value: float, cal: CameraCalibration) -> float:
    return min(cal.max_distance_m, max(cal.min_distance_m, value))
