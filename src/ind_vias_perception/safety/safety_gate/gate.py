from __future__ import annotations

from ind_vias_perception.common.types import Detection
from ind_vias_perception.safety.sentinel_fsm.fsm import SentinelState


class SafetyGate:
    def evaluate(self, detections: list[Detection], sentinel_state: SentinelState) -> dict[str, object]:
        valid = [d for d in detections if d.track_id is not None]
        target = min(valid, key=lambda d: d.distance_m or 1e9) if valid else None
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
        return {
            "warning_level": warning,
            "aeb_ready": warning == "strong",
            "target_track_id": target.track_id,
            "target_distance_m": target.distance_m,
            "target_ttc_s": ttc,
            "sentinel_state": sentinel_state.value,
        }
