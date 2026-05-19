from __future__ import annotations

from dataclasses import dataclass
from ind_vias_perception.common.types import CameraCalibration, Detection
from ind_vias_perception.geometry.calibration.ground_distance import ground_contact_distance_m


@dataclass(frozen=True)
class ScaleAnchor:
    name: str
    scale_or_distance_m: float
    sigma: float


class GeometricGroundContactAnchor:
    name = "geometric_ground_contact_anchor"

    def estimate(self, det: Detection, cal: CameraCalibration) -> ScaleAnchor:
        _, v = det.ground_contact or det.bbox.bottom_center
        distance = ground_contact_distance_m(v, cal)
        return ScaleAnchor(self.name, distance, max(0.15, det.sigma_depth))
