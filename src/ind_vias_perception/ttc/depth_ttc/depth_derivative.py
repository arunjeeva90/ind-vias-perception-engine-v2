from __future__ import annotations

def ttc_from_depth(distance_m: float | None, relative_velocity_mps: float | None) -> float | None:
    if distance_m is None or relative_velocity_mps is None or relative_velocity_mps >= -1e-3:
        return None
    return max(0.0, -distance_m / relative_velocity_mps)
