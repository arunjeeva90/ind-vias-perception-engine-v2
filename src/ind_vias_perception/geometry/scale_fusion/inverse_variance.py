from __future__ import annotations

from collections.abc import Iterable
from ind_vias_perception.geometry.scale_anchors.geometric_anchor import ScaleAnchor


def fuse_inverse_variance(anchors: Iterable[ScaleAnchor]) -> tuple[float, float]:
    valid = [a for a in anchors if a is not None and a.scale_or_distance_m > 0]
    if not valid:
        return float("inf"), 1e9
    weights = [1.0 / max(a.sigma, 1e-3) ** 2 for a in valid]
    value = sum(w * a.scale_or_distance_m for w, a in zip(weights, valid)) / sum(weights)
    sigma = (1.0 / sum(weights)) ** 0.5
    return value, sigma
