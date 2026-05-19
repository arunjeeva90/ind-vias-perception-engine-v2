from __future__ import annotations

import math
from ind_vias_perception.common.types import CameraCalibration


def ground_contact_distance_m(v_px: float, cal: CameraCalibration) -> float:
    denom = max(1.0, v_px - cal.horizon_v_px)
    return (cal.fy_px * cal.height_m) / denom


def pitch_corrected_distance_m(v_px: float, cal: CameraCalibration) -> float:
    theta = math.radians(cal.pitch_deg)
    denom = (v_px - cal.cy_px) * math.cos(theta) + cal.fy_px * math.sin(theta)
    if denom <= 1e-3:
        return float("inf")
    return cal.height_m * cal.fy_px / denom
