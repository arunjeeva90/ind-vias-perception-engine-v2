from __future__ import annotations

import math
from ind_vias_perception.common.types import Detection, ObjectClass
from ind_vias_perception.geometry.scale_anchors.geometric_anchor import ScaleAnchor

_DEFAULT_PRIORS = {
    "car": {"width_m": 1.75, "height_m": 1.50},
    "truck": {"width_m": 2.50, "height_m": 2.80},
    "bus": {"width_m": 2.50, "height_m": 3.00},
    "motorcycle": {"width_m": 0.80, "height_m": 1.40},
    "auto_rickshaw": {"width_m": 1.30, "height_m": 1.75},
    "bicycle": {"height_m": 1.70},
    "cyclist": {"height_m": 1.70},
    "pedestrian": {"height_m": 1.70},
    "animal": {"height_m": 1.20},
}


class SemanticObjectSizeAnchor:
    name = "semantic_object_size_anchor"

    def __init__(self, priors: dict[str, dict[str, float]] | None = None):
        self.priors = _DEFAULT_PRIORS | (priors or {})

    def estimate(self, det: Detection, fy_px: float, fx_px: float | None = None) -> ScaleAnchor:
        fx = fy_px if fx_px is None else fx_px
        prior = self._prior_for(det.label)
        estimates = []
        if "width_m" in prior and det.bbox.width > 0:
            estimates.append((fx * float(prior["width_m"])) / det.bbox.width)
        if "height_m" in prior and det.bbox.height > 0:
            estimates.append((fy_px * float(prior["height_m"])) / det.bbox.height)
        finite = [value for value in estimates if math.isfinite(value) and value > 0]
        distance = sum(finite) / len(finite) if finite else float("inf")
        det.metadata["distance_semantic_m"] = float(distance)
        return ScaleAnchor(self.name, distance, sigma=0.75)

    def _prior_for(self, label: ObjectClass) -> dict[str, float]:
        if label == ObjectClass.BICYCLE and "cyclist" in self.priors:
            return self.priors["cyclist"]
        return self.priors.get(label.value, {})
