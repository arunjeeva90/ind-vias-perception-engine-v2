from __future__ import annotations

from ind_vias_perception.common.types import BBox2D, Detection, ObjectClass
from ind_vias_perception.safety.safety_gate.gate import SafetyGate
from ind_vias_perception.safety.sentinel_fsm.fsm import SentinelState


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

    payload = SafetyGate().evaluate([det], SentinelState.NOMINAL)

    assert payload["warning_level"] == "visual"
    assert payload["aeb_ready"] is False
    assert payload["ego_motion_state"] == "turning"
