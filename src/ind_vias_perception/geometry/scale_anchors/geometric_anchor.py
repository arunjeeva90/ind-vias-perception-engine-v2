from __future__ import annotations

from dataclasses import dataclass
from ind_vias_perception.common.types import CameraCalibration, Detection
from ind_vias_perception.geometry.calibration.ground_distance import (
    ground_contact_distance_m,
    raw_ground_contact_distance_m,
)


@dataclass(frozen=True)
class ScaleAnchor:
    name: str
    scale_or_distance_m: float
    sigma: float


class GeometricGroundContactAnchor:
    name = "geometric_ground_contact_anchor"

    def estimate(self, det: Detection, cal: CameraCalibration) -> ScaleAnchor:
        u, v = det.ground_contact or det.bbox.bottom_center
        raw_distance = raw_ground_contact_distance_m(v, cal)
        distance = ground_contact_distance_m(v, cal)
        det.metadata["u_gc"] = float(u)
        det.metadata["v_gc"] = float(v)
        det.metadata["raw_distance_m"] = float(raw_distance)
        det.metadata["filtered_distance_m"] = float(distance)
        return ScaleAnchor(self.name, distance, max(0.15, det.sigma_depth))
