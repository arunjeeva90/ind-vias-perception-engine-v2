from __future__ import annotations

from dataclasses import dataclass
import math
from ind_vias_perception.common.types import Detection, SceneQuality


@dataclass(frozen=True)
class CAISDecision:
    mode: str
    target_fps: int
    roi_depth_enabled: bool
    reason: str
    score: float = 0.0
    reason_codes: str = "nominal"
    ttc_used_s: float | None = None
    ttc_threshold_s: float = 3.0
    ttc_source_track_id: int | None = None


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

    def decide(
        self,
        detections: list[Detection],
        scene: SceneQuality,
        selected_target_payload: dict[str, object] | None = None,
    ) -> CAISDecision:
        relevant = [d for d in detections if self._include_detection(d)]
        reason_codes: list[str] = []
        scores: list[float] = []
        ttc_used = None
        ttc_source_track_id = None

        selected_target_payload = selected_target_payload or {}
        target_ttc = selected_target_payload.get("target_ttc_s")
        target_valid = bool(selected_target_payload.get("selected_target_valid_for_safety", False))
        ttc_valid = bool(selected_target_payload.get("ttc_valid_for_safety", False))
        if target_valid and ttc_valid and _finite_number(target_ttc):
            ttc_used = float(target_ttc)
            ttc_source_track_id = _maybe_int(selected_target_payload.get("target_track_id"))
            if ttc_used <= self.critical_ttc_s:
                scores.append(min(1.0, (self.critical_ttc_s - ttc_used) / self.critical_ttc_s + 0.55))
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
                ttc_used,
                self.critical_ttc_s,
                ttc_source_track_id,
            )
        if score >= self.enhanced_score_threshold:
            return CAISDecision(
                "enhanced",
                30,
                True,
                "CAIS enhanced score threshold reached",
                score,
                ",".join(reason_codes),
                ttc_used,
                self.critical_ttc_s,
                ttc_source_track_id,
            )
        if scene.degraded_score > 0.6:
            return CAISDecision(
                "degraded",
                15,
                False,
                "sensor quality degraded",
                0.4,
                "sensor_degraded",
                ttc_used,
                self.critical_ttc_s,
                ttc_source_track_id,
            )
        return CAISDecision(
            "nominal",
            25,
            False,
            "nominal scene",
            score,
            ",".join(reason_codes or ["nominal"]),
            ttc_used,
            self.critical_ttc_s,
            ttc_source_track_id,
        )

    def _include_detection(self, det: Detection) -> bool:
        if not self.ignore_invalid_side_objects:
            return True
        if det.metadata.get("distance_valid_for_safety", True):
            return True
        return bool(det.metadata.get("in_ego_corridor", False))


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _maybe_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
