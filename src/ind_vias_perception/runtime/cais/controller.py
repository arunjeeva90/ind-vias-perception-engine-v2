from __future__ import annotations

from dataclasses import dataclass
from ind_vias_perception.common.types import Detection, SceneQuality


@dataclass(frozen=True)
class CAISDecision:
    mode: str
    target_fps: int
    roi_depth_enabled: bool
    reason: str
    score: float = 0.0
    reason_codes: str = "nominal"


class CAISController:
    def __init__(
        self,
        high_uncertainty: float = 0.55,
        critical_ttc_s: float = 3.0,
        high_complexity: float = 0.70,
        enhanced_score_threshold: float = 0.45,
        critical_score_threshold: float = 0.75,
        ignore_invalid_side_objects: bool = True,
    ):
        self.high_uncertainty = high_uncertainty
        self.critical_ttc_s = critical_ttc_s
        self.high_complexity = high_complexity
        self.enhanced_score_threshold = enhanced_score_threshold
        self.critical_score_threshold = critical_score_threshold
        self.ignore_invalid_side_objects = ignore_invalid_side_objects

    def decide(self, detections: list[Detection], scene: SceneQuality) -> CAISDecision:
        relevant = [d for d in detections if self._include_detection(d)]
        reason_codes: list[str] = []
        scores: list[float] = []

        valid_ttc = [
            d.ttc_s
            for d in relevant
            if d.ttc_s is not None and d.metadata.get("ttc_valid_for_safety", False)
        ]
        if valid_ttc:
            min_ttc = min(valid_ttc)
            if min_ttc < self.critical_ttc_s:
                scores.append(min(1.0, (self.critical_ttc_s - min_ttc) / self.critical_ttc_s + 0.55))
                reason_codes.append("valid_ttc_below_threshold")

        high_uncertainty = [
            d
            for d in relevant
            if d.metadata.get("distance_valid_for_safety", True)
            and d.sigma_depth > self.high_uncertainty
        ]
        if high_uncertainty:
            scores.append(0.55)
            reason_codes.append("safety_relevant_uncertainty_high")

        if scene.complexity > self.high_complexity:
            scores.append(0.45)
            reason_codes.append("scene_complexity_high")

        score = max(scores or [0.0])
        if score >= self.critical_score_threshold:
            return CAISDecision(
                "critical",
                30,
                True,
                "CAIS critical score threshold reached",
                score,
                ",".join(reason_codes),
            )
        if score >= self.enhanced_score_threshold:
            return CAISDecision(
                "enhanced",
                30,
                True,
                "CAIS enhanced score threshold reached",
                score,
                ",".join(reason_codes),
            )
        if scene.degraded_score > 0.6:
            return CAISDecision("degraded", 15, False, "sensor quality degraded", 0.4, "sensor_degraded")
        return CAISDecision("nominal", 25, False, "nominal scene", score, ",".join(reason_codes or ["nominal"]))

    def _include_detection(self, det: Detection) -> bool:
        if not self.ignore_invalid_side_objects:
            return True
        if det.metadata.get("distance_valid_for_safety", True):
            return True
        return bool(det.metadata.get("in_ego_corridor", False))
