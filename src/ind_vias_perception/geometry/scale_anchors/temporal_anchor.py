from __future__ import annotations

from ind_vias_perception.common.types import Detection
from ind_vias_perception.geometry.scale_anchors.geometric_anchor import ScaleAnchor


class TemporalTrackAnchor:
    name = "temporal_track_anchor"

    def estimate(self, det: Detection) -> ScaleAnchor | None:
        if det.distance_m is None:
            return None
        return ScaleAnchor(self.name, det.distance_m, sigma=max(0.2, det.sigma_depth * 0.8))
