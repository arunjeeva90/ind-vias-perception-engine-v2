from __future__ import annotations

from ind_vias_perception.common.types import Detection
from ind_vias_perception.safety.sentinel_fsm.fsm import SentinelState


class SafetyGate:
    def __init__(self, ego_corridor: dict[str, object] | None = None):
        self.ego_corridor = ego_corridor or {}

    def evaluate(self, detections: list[Detection], sentinel_state: SentinelState) -> dict[str, object]:
        valid = [d for d in detections if _safety_distance_m(d) < 1e9]
        candidates = _ranked_candidates(valid)
        target = candidates[0] if candidates else None
        if target is None:
            return {"warning_level": "none", "aeb_ready": False, "reason": "no target"}
        conf = target.confidence * (1.0 - min(0.9, target.sigma_depth))
        ttc = target.ttc_s
        warning = "none"
        if sentinel_state == SentinelState.NOMINAL and ttc is not None:
            if ttc < 2.0 and conf > 0.75:
                warning = "strong"
            elif ttc < 3.5 and conf > 0.55:
                warning = "visual"
            elif ttc < 5.0 and conf > 0.35:
                warning = "advisory"
        if target.metadata.get("ego_motion_state") == "turning" and warning == "strong" and conf < 0.9:
            warning = "visual"
        return {
            "warning_level": warning,
            "aeb_ready": warning == "strong",
            "target_track_id": target.track_id,
            "target_distance_m": _safety_distance_m(target),
            "target_ttc_s": ttc,
            "target_in_ego_corridor": bool(target.metadata.get("in_ego_corridor", False)),
            "target_relevance": float(target.metadata.get("target_relevance", 0.0)),
            "target_distance_valid_for_safety": bool(
                target.metadata.get("distance_valid_for_safety", True)
            ),
            "ego_motion_state": target.metadata.get("ego_motion_state", "straight"),
            "sentinel_state": sentinel_state.value,
        }


def _safety_distance_m(det: Detection) -> float:
    distance = det.metadata.get("distance_bumper_m", det.distance_m)
    if distance is None:
        return 1e9
    return float(distance)


def _ranked_candidates(detections: list[Detection]) -> list[Detection]:
    ego_valid = [
        d
        for d in detections
        if d.metadata.get("in_ego_corridor", False)
        and d.metadata.get("distance_valid_for_safety", True)
    ]
    if ego_valid:
        return sorted(ego_valid, key=_target_sort_key)

    ego_any = [d for d in detections if d.metadata.get("in_ego_corridor", False)]
    if ego_any:
        return sorted(ego_any, key=_target_sort_key)

    valid_any = [d for d in detections if d.metadata.get("distance_valid_for_safety", True)]
    if valid_any:
        return sorted(valid_any, key=_target_sort_key)

    return sorted(detections, key=_target_sort_key)


def _target_sort_key(det: Detection) -> tuple[float, float]:
    return (-float(det.metadata.get("target_relevance", 0.0)), _safety_distance_m(det))
