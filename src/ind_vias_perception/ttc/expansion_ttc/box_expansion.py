from __future__ import annotations

def ttc_from_area(area_now: float, area_prev: float | None, dt: float) -> float | None:
    if area_prev is None or dt <= 0:
        return None
    d_area = area_now - area_prev
    if d_area <= 1e-6:
        return None
    return area_now / (d_area / dt)
