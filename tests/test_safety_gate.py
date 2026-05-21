from __future__ import annotations

from ind_vias_perception.common.types import BBox2D, Detection, ObjectClass
from ind_vias_perception.safety.safety_gate.gate import SafetyGate
from ind_vias_perception.safety.sentinel_fsm.fsm import SentinelState


def _warning_det(ttc_s: float = 3.0) -> Detection:
    det = Detection(BBox2D(0, 0, 10, 10), ObjectClass.CAR, 0.9)
    det.track_id = 7
    det.ttc_s = ttc_s
    det.sigma_depth = 0.1
    det.metadata["distance_bumper_m"] = 5.0
    return det


def _invalid_warning_det(ttc_s: float = 1.0) -> Detection:
    det = _warning_det(ttc_s)
    det.metadata["distance_valid_for_safety"] = False
    det.metadata["in_ego_corridor"] = True
    return det


def test_safety_gate_uses_bumper_distance_when_available():
    det = Detection(BBox2D(0, 0, 10, 10), ObjectClass.CAR, 0.9, distance_m=20.0)
    det.track_id = 1
    det.metadata["distance_bumper_m"] = 18.55

    payload = SafetyGate().evaluate([det], SentinelState.NOMINAL)

    assert payload["target_distance_m"] == 18.55


def test_ego_corridor_target_preferred_over_side_lane_target():
    side = Detection(BBox2D(0, 0, 10, 10), ObjectClass.CAR, 0.9)
    side.track_id = 1
    side.metadata["distance_bumper_m"] = 5.0
    side.metadata["in_ego_corridor"] = False
    side.metadata["target_relevance"] = 0.5
    center = Detection(BBox2D(0, 0, 10, 10), ObjectClass.CAR, 0.9)
    center.track_id = 2
    center.metadata["distance_bumper_m"] = 20.0
    center.metadata["in_ego_corridor"] = True
    center.metadata["target_relevance"] = 0.9

    payload = SafetyGate({"enabled": True}).evaluate([side, center], SentinelState.NOMINAL)

    assert payload["target_track_id"] == 2
    assert payload["target_distance_m"] == 20.0
    assert payload["target_in_ego_corridor"] is True


def test_target_selection_falls_back_when_no_object_is_inside_corridor():
    side_near = Detection(BBox2D(0, 0, 10, 10), ObjectClass.CAR, 0.9)
    side_near.track_id = 1
    side_near.metadata["distance_bumper_m"] = 5.0
    side_near.metadata["in_ego_corridor"] = False
    side_near.metadata["target_relevance"] = 0.9
    side_far = Detection(BBox2D(0, 0, 10, 10), ObjectClass.CAR, 0.9)
    side_far.track_id = 2
    side_far.metadata["distance_bumper_m"] = 20.0
    side_far.metadata["in_ego_corridor"] = False
    side_far.metadata["target_relevance"] = 0.5

    payload = SafetyGate({"enabled": True}).evaluate([side_far, side_near], SentinelState.NOMINAL)

    assert payload["target_track_id"] == 1
    assert payload["target_distance_m"] == 5.0
    assert payload["target_in_ego_corridor"] is False


def test_safety_gate_ignores_invalid_side_object_when_valid_ego_object_exists():
    side = Detection(BBox2D(0, 0, 10, 10), ObjectClass.CAR, 0.9)
    side.track_id = 1
    side.metadata["distance_bumper_m"] = 2.0
    side.metadata["in_ego_corridor"] = False
    side.metadata["distance_valid_for_safety"] = False
    side.metadata["target_relevance"] = 0.95
    ego = Detection(BBox2D(0, 0, 10, 10), ObjectClass.CAR, 0.9)
    ego.track_id = 2
    ego.metadata["distance_bumper_m"] = 15.0
    ego.metadata["in_ego_corridor"] = True
    ego.metadata["distance_valid_for_safety"] = True
    ego.metadata["target_relevance"] = 0.8

    payload = SafetyGate({"enabled": True}).evaluate([side, ego], SentinelState.NOMINAL)

    assert payload["target_track_id"] == 2
    assert payload["target_distance_valid_for_safety"] is True


def test_turning_suppresses_strong_warning_when_confidence_is_not_high():
    det = Detection(BBox2D(0, 0, 10, 10), ObjectClass.CAR, 0.9)
    det.track_id = 1
    det.ttc_s = 1.0
    det.sigma_depth = 0.1
    det.metadata["distance_bumper_m"] = 5.0
    det.metadata["ego_motion_state"] = "turning"
    det.metadata["yaw_confidence"] = 0.9

    payload = SafetyGate().evaluate([det], SentinelState.NOMINAL)

    assert payload["warning_level"] == "visual"
    assert payload["aeb_ready"] is False
    assert payload["ego_motion_state"] == "turning"


def test_warning_not_confirmed_on_first_frame():
    gate = SafetyGate(
        confirmation_cfg={"enabled": True, "required_frames": {"warning": 2}}
    )

    payload = gate.evaluate([_warning_det(ttc_s=3.0)], SentinelState.NOMINAL)

    assert payload["raw_warning_level"] == "visual"
    assert payload["confirmed_warning_level"] == "none"
    assert payload["warning_level"] == "none"
    assert payload["confirmation_count"] == 1
    assert payload["confirmation_required"] == 2


def test_warning_confirmed_after_required_frames():
    gate = SafetyGate(
        confirmation_cfg={"enabled": True, "required_frames": {"warning": 2}}
    )

    gate.evaluate([_warning_det(ttc_s=3.0)], SentinelState.NOMINAL)
    payload = gate.evaluate([_warning_det(ttc_s=3.0)], SentinelState.NOMINAL)

    assert payload["raw_warning_level"] == "visual"
    assert payload["confirmed_warning_level"] == "visual"
    assert payload["warning_level"] == "visual"
    assert payload["confirmation_count"] == 2


def test_warning_confirmation_resets_when_condition_disappears():
    gate = SafetyGate(
        confirmation_cfg={"enabled": True, "required_frames": {"warning": 2}}
    )

    gate.evaluate([_warning_det(ttc_s=3.0)], SentinelState.NOMINAL)
    clear = _warning_det(ttc_s=9.0)
    payload = gate.evaluate([clear], SentinelState.NOMINAL)

    assert payload["raw_warning_level"] == "none"
    assert payload["confirmed_warning_level"] == "none"
    assert payload["confirmation_count"] == 0


def test_disabled_confirmation_preserves_old_behavior():
    gate = SafetyGate(
        confirmation_cfg={"enabled": False, "required_frames": {"warning": 2}}
    )

    payload = gate.evaluate([_warning_det(ttc_s=3.0)], SentinelState.NOMINAL)

    assert payload["raw_warning_level"] == "visual"
    assert payload["confirmed_warning_level"] == "visual"
    assert payload["warning_level"] == "visual"
    assert payload["confirmation_count"] == 1
    assert payload["confirmation_required"] == 1


def test_predicted_only_track_is_not_used_for_new_strong_warning():
    gate = SafetyGate()
    det = _warning_det(ttc_s=1.0)
    det.metadata["track_predicted"] = True

    payload = gate.evaluate([det], SentinelState.NOMINAL)

    assert payload["raw_warning_level"] == "visual"
    assert payload["warning_level"] == "visual"
    assert payload["aeb_ready"] is False


def test_invalid_distance_target_with_low_ttc_does_not_produce_advisory():
    payload = SafetyGate().evaluate([_invalid_warning_det(ttc_s=1.0)], SentinelState.NOMINAL)

    assert payload["target_distance_valid_for_safety"] is False
    assert payload["raw_warning_level"] == "none"
    assert payload["confirmed_warning_level"] == "none"
    assert payload["warning_candidate"] == "none"
    assert payload["aeb_ready"] is False
    assert payload["warning_suppressed_reason"] == "no_valid_safety_target"


def test_valid_distance_target_with_low_ttc_still_produces_warning_candidate():
    payload = SafetyGate().evaluate([_warning_det(ttc_s=3.0)], SentinelState.NOMINAL)

    assert payload["target_distance_valid_for_safety"] is True
    assert payload["raw_warning_level"] == "visual"
    assert payload["warning_candidate"] == "visual"


def test_confirmation_does_not_advance_for_invalid_distance_target():
    gate = SafetyGate(
        confirmation_cfg={"enabled": True, "required_frames": {"warning": 2, "aeb_ready": 3}}
    )

    first = gate.evaluate([_invalid_warning_det(ttc_s=1.0)], SentinelState.NOMINAL)
    second = gate.evaluate([_invalid_warning_det(ttc_s=1.0)], SentinelState.NOMINAL)

    assert first["confirmation_count"] == 0
    assert second["confirmation_count"] == 0
    assert second["confirmed_warning_level"] == "none"


def test_invalid_ttc_is_not_used_for_warning():
    det = _warning_det(ttc_s=1.0)
    det.metadata["ttc_valid_for_safety"] = False
    det.metadata["ttc_reason_codes"] = "predicted_track"

    payload = SafetyGate().evaluate([det], SentinelState.NOMINAL)

    assert payload["ttc_valid_for_safety"] is False
    assert payload["ttc_reason_codes"] == "predicted_track"
    assert payload["raw_warning_level"] == "none"
    assert payload["confirmed_warning_level"] == "none"


def test_side_low_relevance_target_does_not_produce_advisory_fcw_warning():
    det = _warning_det(ttc_s=3.0)
    det.metadata["in_ego_corridor"] = False
    det.metadata["target_relevance"] = 0.2
    det.metadata["ttc_valid_for_safety"] = True

    payload = SafetyGate(
        safety_gate_cfg={
            "min_relevance_for_fcw_warning": 0.5,
            "allow_side_target_fcw_warning": False,
        }
    ).evaluate([det], SentinelState.NOMINAL)

    assert payload["raw_warning_level"] == "none"
    assert payload["confirmed_warning_level"] == "none"


def test_valid_ego_corridor_target_preferred_over_invalid_nearer_target():
    invalid_near = _invalid_warning_det(ttc_s=1.0)
    invalid_near.track_id = 1
    invalid_near.metadata["distance_bumper_m"] = 2.0
    invalid_near.metadata["target_relevance"] = 1.0
    valid_far = _warning_det(ttc_s=3.0)
    valid_far.track_id = 2
    valid_far.metadata["distance_bumper_m"] = 15.0
    valid_far.metadata["in_ego_corridor"] = True
    valid_far.metadata["distance_valid_for_safety"] = True
    valid_far.metadata["target_relevance"] = 0.8

    payload = SafetyGate().evaluate([invalid_near, valid_far], SentinelState.NOMINAL)

    assert payload["target_track_id"] == 2
    assert payload["selected_target_valid_for_safety"] is True
    assert payload["debug_target_track_id"] == 1
    assert payload["debug_target_distance_valid_for_safety"] is False


def test_invalid_target_only_has_debug_target_and_no_warning():
    invalid = _invalid_warning_det(ttc_s=1.0)
    invalid.track_id = 11

    payload = SafetyGate().evaluate([invalid], SentinelState.NOMINAL)

    assert payload["target_track_id"] == 11
    assert payload["debug_target_track_id"] == 11
    assert payload["selected_target_valid_for_safety"] is False
    assert payload["debug_target_distance_valid_for_safety"] is False
    assert payload["raw_warning_level"] == "none"
    assert payload["confirmed_warning_level"] == "none"
    assert payload["aeb_ready"] is False
    assert payload["warning_suppressed_reason"] == "no_valid_safety_target"


def test_predicted_valid_target_loses_priority_to_stable_valid_target():
    predicted = _warning_det(ttc_s=1.0)
    predicted.track_id = 1
    predicted.metadata["distance_bumper_m"] = 3.0
    predicted.metadata["in_ego_corridor"] = True
    predicted.metadata["distance_valid_for_safety"] = True
    predicted.metadata["target_relevance"] = 0.9
    predicted.metadata["track_predicted"] = True
    stable = _warning_det(ttc_s=3.0)
    stable.track_id = 2
    stable.metadata["distance_bumper_m"] = 12.0
    stable.metadata["in_ego_corridor"] = True
    stable.metadata["distance_valid_for_safety"] = True
    stable.metadata["target_relevance"] = 0.9
    stable.metadata["track_predicted"] = False

    payload = SafetyGate().evaluate([predicted, stable], SentinelState.NOMINAL)

    assert payload["target_track_id"] == 2
    assert payload["selected_target_reason"] == "valid_safety_target"
