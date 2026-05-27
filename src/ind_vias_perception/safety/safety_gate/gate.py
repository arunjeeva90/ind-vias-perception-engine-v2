from __future__ import annotations

from ind_vias_perception.common.types import Detection
from ind_vias_perception.safety.safety_gate.confirmation import WarningConfirmationGate
from ind_vias_perception.safety.sentinel_fsm.fsm import SentinelState


class SafetyGate:
    def __init__(
        self,
        ego_corridor: dict[str, object] | None = None,
        confirmation_cfg: dict[str, object] | None = None,
        safety_gate_cfg: dict[str, object] | None = None,
    ):
        self.ego_corridor = ego_corridor or {}
        safety_gate_cfg = safety_gate_cfg or {}
        self.min_relevance_for_fcw_warning = float(
            safety_gate_cfg.get("min_relevance_for_fcw_warning", 0.0)
        )
        self.allow_side_target_fcw_warning = bool(
            safety_gate_cfg.get("allow_side_target_fcw_warning", True)
        )
        confirmation_cfg = confirmation_cfg or {}
        self.confirmation = WarningConfirmationGate(
            enabled=bool(confirmation_cfg.get("enabled", False)),
            required_frames=confirmation_cfg.get("required_frames", {}),
        )

    def evaluate(self, detections: list[Detection], sentinel_state: SentinelState) -> dict[str, object]:
        candidates = [d for d in detections if _safety_distance_m(d) < 1e9]
        debug_target = _best_debug_target(candidates)
        safety_candidates = [d for d in candidates if _distance_valid_for_safety(d)]
        target = _best_safety_target(safety_candidates)
        if debug_target is None:
            return {"warning_level": "none", "aeb_ready": False, "reason": "no target"}
        if target is None:
            confirmation = self.confirmation.update(None, "none", aeb_candidate=False)
            return self._payload(
                target=debug_target,
                ttc=debug_target.ttc_s,
                raw_warning="none",
                confirmed_warning="none",
                confirmation=confirmation,
                sentinel_state=sentinel_state,
                warning_suppressed_reason="no_valid_safety_target",
                selected_target=debug_target,
                selected_target_reason="no_valid_safety_target",
                debug_target=debug_target,
            )
        conf = target.confidence * (1.0 - min(0.9, target.sigma_depth))
        ttc = target.ttc_s
        raw_warning = "none"
        ttc_valid = bool(target.metadata.get("ttc_valid_for_safety", True))
        if sentinel_state == SentinelState.NOMINAL and ttc is not None and ttc_valid:
            if ttc < 2.0 and conf > 0.75:
                raw_warning = "strong"
            elif ttc < 3.5 and conf > 0.55:
                raw_warning = "visual"
            elif ttc < 5.0 and conf > 0.35:
                raw_warning = "advisory"
        cutin_target = _best_cutin_target(detections)
        if raw_warning == "none" and cutin_target is not None:
            target = cutin_target
            ttc = target.ttc_s
            raw_warning = "cut_in_risk"
        high_conf_turning = (
            target.metadata.get("ego_motion_state") == "turning"
            and float(target.metadata.get("yaw_confidence", 0.0)) >= 0.8
        )
        if high_conf_turning and raw_warning == "strong" and conf < 0.9:
            raw_warning = "visual"
        if _predicted_without_recent_confirmation(target) and raw_warning == "strong":
            raw_warning = "visual"
        if raw_warning != "none" and self._suppress_side_target_warning(target):
            raw_warning = "none"
        confirmation = self.confirmation.update(
            target.track_id,
            raw_warning,
            aeb_candidate=raw_warning == "strong",
        )
        confirmed_warning = confirmation.confirmed_warning_level
        return self._payload(
            target=target,
            ttc=ttc,
            raw_warning=raw_warning,
            confirmed_warning=confirmed_warning,
            confirmation=confirmation,
            sentinel_state=sentinel_state,
            selected_target=target,
            selected_target_reason="valid_safety_target",
            debug_target=debug_target,
        )

    def _payload(
        self,
        target: Detection,
        ttc: float | None,
        raw_warning: str,
        confirmed_warning: str,
        confirmation,
        sentinel_state: SentinelState,
        warning_suppressed_reason: str | None = None,
        selected_target: Detection | None = None,
        selected_target_reason: str = "valid_safety_target",
        debug_target: Detection | None = None,
    ) -> dict[str, object]:
        selected_target = selected_target or target
        debug_target = debug_target or target
        return {
            "warning_level": confirmed_warning,
            "raw_warning_level": raw_warning,
            "confirmed_warning_level": confirmed_warning,
            "aeb_ready": confirmed_warning == "strong",
            "warning_candidate": confirmation.warning_candidate,
            "confirmation_count": confirmation.confirmation_count,
            "confirmation_required": confirmation.confirmation_required,
            "target_track_id": target.track_id,
            "target_distance_m": _safety_distance_m(target),
            "target_ttc_s": ttc,
            "ttc_valid_for_safety": bool(target.metadata.get("ttc_valid_for_safety", False)),
            "ttc_reason_codes": target.metadata.get("ttc_reason_codes", "n/a"),
            "target_in_ego_corridor": bool(target.metadata.get("in_ego_corridor", False)),
            "target_relevance": float(target.metadata.get("target_relevance", 0.0)),
            "target_distance_valid_for_safety": bool(
                target.metadata.get("distance_valid_for_safety", True)
            ),
            "selected_target_valid_for_safety": _distance_valid_for_safety(selected_target),
            "selected_target_reason": selected_target_reason,
            "debug_target_track_id": debug_target.track_id,
            "debug_target_distance_valid_for_safety": _distance_valid_for_safety(debug_target),
            "ego_motion_state": target.metadata.get("ego_motion_state", "straight"),
            "yaw_confidence": float(target.metadata.get("yaw_confidence", 0.0)),
            "warning_suppressed_reason": warning_suppressed_reason,
            "sentinel_state": sentinel_state.value,
            "cutin_warning_candidate": (
                confirmation.warning_candidate if confirmation.warning_candidate == "cut_in_risk" else "none"
            ),
            "cutin_warning_confirmed": (
                confirmed_warning if confirmed_warning == "cut_in_risk" else "none"
            ),
            "cutin_target_track_id": (
                target.track_id
                if target.metadata.get("cutin_valid_for_safety", False)
                and target.metadata.get("cutin_warning_candidate") == "cut_in_risk"
                and target.metadata.get("cutin_warning_eligible", False)
                else None
            ),
            "side_state": target.metadata.get("side_state", "n/a"),
            "cutin_state": target.metadata.get("cutin_state", "NONE"),
            "ttc_lateral_s": target.metadata.get("ttc_lateral_s"),
            "cutin_confidence": float(target.metadata.get("cutin_confidence", 0.0)),
            "cutin_valid_for_safety": bool(target.metadata.get("cutin_valid_for_safety", False)),
            "cutin_reason_codes": target.metadata.get("cutin_reason_codes", "n/a"),
            "lateral_velocity_px_s": float(target.metadata.get("lateral_velocity_px_s", 0.0)),
            "lateral_history_count": int(float(target.metadata.get("lateral_history_count", 0.0))),
            "corridor_overlap_ratio": float(target.metadata.get("corridor_overlap_ratio", 0.0)),
            "corridor_overlap_delta": float(target.metadata.get("corridor_overlap_delta", 0.0)),
            "corridor_entry_confirmed": bool(
                target.metadata.get("corridor_entry_confirmed", False)
            ),
            "lateral_motion_stable": bool(target.metadata.get("lateral_motion_stable", False)),
            "lateral_center_history_count": int(
                float(target.metadata.get("lateral_center_history_count", 0.0))
            ),
            "lateral_velocity_px_s_smoothed": float(
                target.metadata.get("lateral_velocity_px_s_smoothed", 0.0)
            ),
            "cutin_crossing_trend": bool(target.metadata.get("cutin_crossing_trend", False)),
            "cutin_entry_side": target.metadata.get("cutin_entry_side", "UNKNOWN"),
            "cutin_warning_eligible": bool(target.metadata.get("cutin_warning_eligible", False)),
            "crossing_state": target.metadata.get("crossing_state", "none"),
            "crossing_confidence": float(target.metadata.get("crossing_confidence", 0.0)),
            "crossing_history_count": int(float(target.metadata.get("crossing_history_count", 0.0))),
            "crossing_valid_for_safety": bool(
                target.metadata.get("crossing_valid_for_safety", False)
            ),
            "crossing_reason_codes": target.metadata.get("crossing_reason_codes", "n/a"),
            "crossing_lateral_displacement_px": float(
                target.metadata.get("crossing_lateral_displacement_px", 0.0)
            ),
            "crossing_corridor_approach": bool(
                target.metadata.get("crossing_corridor_approach", False)
            ),
            "crossing_boundary_suppressed": bool(
                target.metadata.get("crossing_boundary_suppressed", False)
            ),
            "crossing_tiny_object_suppressed": bool(
                target.metadata.get("crossing_tiny_object_suppressed", False)
            ),
        }

    def _suppress_side_target_warning(self, target: Detection) -> bool:
        if target.metadata.get("in_ego_corridor", False):
            return False
        if self.allow_side_target_fcw_warning:
            return False
        if bool(target.metadata.get("cut_in_risk", False)):
            return False
        return float(target.metadata.get("target_relevance", 0.0)) < self.min_relevance_for_fcw_warning


def _safety_distance_m(det: Detection) -> float:
    distance = det.metadata.get("distance_bumper_m", det.distance_m)
    if distance is None:
        return 1e9
    return float(distance)


def _best_safety_target(detections: list[Detection]) -> Detection | None:
    if not detections:
        return None
    return sorted(detections, key=_target_sort_key)[0]


def _best_debug_target(detections: list[Detection]) -> Detection | None:
    if not detections:
        return None
    return sorted(detections, key=_debug_sort_key)[0]


def _best_cutin_target(detections: list[Detection]) -> Detection | None:
    cutin = [
        det
        for det in detections
        if det.metadata.get("cutin_valid_for_safety", False)
        and det.metadata.get("cutin_warning_candidate") == "cut_in_risk"
        and det.metadata.get("cut_in_risk", False)
        and det.metadata.get("cutin_warning_eligible", False)
        and det.metadata.get("corridor_entry_confirmed", False)
        and det.metadata.get("lateral_motion_stable", False)
        and det.metadata.get("distance_valid_for_safety", True)
        and det.metadata.get("ego_motion_state", "straight") == "straight"
        and _valid_lateral_ttc(det)
        and not det.metadata.get("cutin_warning_suppressed", False)
    ]
    if not cutin:
        return None
    return sorted(
        cutin,
        key=lambda det: (
            -float(det.metadata.get("cutin_confidence", 0.0)),
            float(det.metadata.get("ttc_lateral_s") or 1e9),
            _safety_distance_m(det),
        ),
    )[0]


def _target_sort_key(det: Detection) -> tuple[int, int, float, float, int]:
    return (
        0 if det.metadata.get("in_ego_corridor", False) else 1,
        0 if _distance_valid_for_safety(det) else 1,
        -float(det.metadata.get("target_relevance", 0.0)),
        1 if det.metadata.get("track_predicted", False) else 0,
        _safety_distance_m(det),
    )


def _debug_sort_key(det: Detection) -> tuple[int, float, float, int]:
    return (
        0 if det.metadata.get("in_ego_corridor", False) else 1,
        -float(det.metadata.get("target_relevance", 0.0)),
        _safety_distance_m(det),
        1 if det.metadata.get("track_predicted", False) else 0,
    )


def _distance_valid_for_safety(det: Detection) -> bool:
    return bool(det.metadata.get("distance_valid_for_safety", True))


def _predicted_without_recent_confirmation(det: Detection) -> bool:
    if not det.metadata.get("track_predicted", False):
        return False
    return float(det.metadata.get("confirmation_count", 0.0)) < 1.0


def _valid_lateral_ttc(det: Detection) -> bool:
    ttc = det.metadata.get("ttc_lateral_s")
    if ttc is None:
        return False
    ttc = float(ttc)
    return 0.4 <= ttc <= 4.0
