from __future__ import annotations

from ind_vias_perception.common.types import Detection, ObjectClass
from ind_vias_perception.geometry.scale_anchors.geometric_anchor import ScaleAnchor

_PRIOR_HEIGHT_M = {
    ObjectClass.CAR: 1.50, ObjectClass.TRUCK: 3.0, ObjectClass.BUS: 3.1,
    ObjectClass.MOTORCYCLE: 1.25, ObjectClass.AUTO_RICKSHAW: 1.75,
    ObjectClass.BICYCLE: 1.45, ObjectClass.PEDESTRIAN: 1.70, ObjectClass.ANIMAL: 1.20,
}


class SemanticObjectSizeAnchor:
    name = "semantic_object_size_anchor"

    def estimate(self, det: Detection, fy_px: float) -> ScaleAnchor:
        h_prior = _PRIOR_HEIGHT_M.get(det.label, 1.6)
        distance = (h_prior * fy_px) / max(det.bbox.height, 1.0)
        return ScaleAnchor(self.name, distance, sigma=0.75)
