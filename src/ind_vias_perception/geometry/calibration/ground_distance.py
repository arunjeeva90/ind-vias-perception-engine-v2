from __future__ import annotations

import math
from ind_vias_perception.common.types import CameraCalibration


def ground_contact_distance_m(v_px: float, cal: CameraCalibration) -> float:
    raw = raw_ground_contact_distance_m(v_px, cal)
    if not math.isfinite(raw):
        return float("inf")
    return min(cal.max_distance_m, max(cal.min_distance_m, raw))


def raw_ground_contact_distance_m(v_px: float, cal: CameraCalibration) -> float:
    if v_px <= cal.horizon_v_px:
        return float("inf")
    return (cal.fy_px * cal.height_m) / (v_px - cal.horizon_v_px)


def pitch_corrected_distance_m(v_px: float, cal: CameraCalibration) -> float:
    theta = math.radians(cal.pitch_deg)
    denom = (v_px - cal.cy_px) * math.cos(theta) + cal.fy_px * math.sin(theta)
    if denom <= 1e-3:
        return float("inf")
    return cal.height_m * cal.fy_px / denom
