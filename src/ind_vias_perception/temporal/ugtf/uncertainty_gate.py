from __future__ import annotations

from ind_vias_perception.common.types import Detection


def temporal_alpha_from_uncertainty(det: Detection) -> float:
    return max(0.1, min(0.95, 1.0 / (1.0 + det.sigma_depth)))
