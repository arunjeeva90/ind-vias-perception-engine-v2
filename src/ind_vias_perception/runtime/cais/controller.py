from __future__ import annotations

from dataclasses import dataclass
from ind_vias_perception.common.types import Detection, SceneQuality


@dataclass(frozen=True)
class CAISDecision:
    mode: str
    target_fps: int
    roi_depth_enabled: bool
    reason: str


class CAISController:
    def __init__(self, high_uncertainty: float = 0.55, critical_ttc_s: float = 3.0, high_complexity: float = 0.70):
        self.high_uncertainty = high_uncertainty
        self.critical_ttc_s = critical_ttc_s
        self.high_complexity = high_complexity

    def decide(self, detections: list[Detection], scene: SceneQuality) -> CAISDecision:
        min_ttc = min([d.ttc_s for d in detections if d.ttc_s is not None] or [99.0])
        max_unc = max([d.sigma_depth for d in detections] or [0.0])
        if min_ttc < self.critical_ttc_s:
            return CAISDecision("critical", 30, True, "TTC below critical threshold")
        if max_unc > self.high_uncertainty or scene.complexity > self.high_complexity:
            return CAISDecision("enhanced", 30, True, "uncertainty or scene complexity high")
        if scene.degraded_score > 0.6:
            return CAISDecision("degraded", 15, False, "sensor quality degraded")
        return CAISDecision("nominal", 25, False, "nominal scene")
