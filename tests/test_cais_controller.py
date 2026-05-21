from __future__ import annotations

from ind_vias_perception.common.types import BBox2D, Detection, ObjectClass, SceneQuality
from ind_vias_perception.runtime.cais.controller import CAISController


def _det(ttc_s: float | None, valid: bool, in_corridor: bool, sigma: float = 0.9) -> Detection:
    det = Detection(BBox2D(0, 0, 10, 10), ObjectClass.CAR, 0.9)
    det.ttc_s = ttc_s
    det.sigma_depth = sigma
    det.metadata["distance_valid_for_safety"] = valid
    det.metadata["in_ego_corridor"] = in_corridor
    det.metadata["ttc_valid_for_safety"] = bool(valid and ttc_s is not None)
    return det


def test_missing_ttc_does_not_produce_cais_valid_ttc_below_threshold():
    controller = CAISController(ignore_invalid_side_objects=True)

    decision = controller.decide(
        [_det(None, valid=True, in_corridor=True, sigma=0.2)],
        SceneQuality(),
        {
            "selected_target_valid_for_safety": True,
            "ttc_valid_for_safety": False,
            "target_ttc_s": None,
            "target_track_id": 1,
        },
    )

    assert decision.mode == "nominal"
    assert "valid_ttc_below_threshold" not in decision.reason_codes
    assert decision.ttc_used_s is None


def test_high_ttc_does_not_produce_cais_valid_ttc_below_threshold():
    controller = CAISController(critical_ttc_s=3.0)

    decision = controller.decide(
        [_det(48.0, valid=True, in_corridor=True, sigma=0.2)],
        SceneQuality(),
        {
            "selected_target_valid_for_safety": True,
            "ttc_valid_for_safety": True,
            "target_ttc_s": 48.0,
            "target_track_id": 1,
        },
    )

    assert decision.mode == "nominal"
    assert "valid_ttc_below_threshold" not in decision.reason_codes
    assert decision.ttc_used_s == 48.0


def test_valid_low_ttc_target_can_force_cais_critical():
    controller = CAISController(critical_ttc_s=3.0, critical_score_threshold=0.75)

    decision = controller.decide(
        [_det(1.0, valid=True, in_corridor=True, sigma=0.2)],
        SceneQuality(),
        {
            "selected_target_valid_for_safety": True,
            "ttc_valid_for_safety": True,
            "target_ttc_s": 1.0,
            "target_track_id": 7,
        },
    )

    assert decision.mode == "critical"
    assert "valid_ttc_below_threshold" in decision.reason_codes
    assert decision.score >= 0.75
    assert decision.ttc_used_s == 1.0
    assert decision.ttc_source_track_id == 7


def test_low_ttc_from_non_selected_target_does_not_affect_cais():
    controller = CAISController(critical_ttc_s=3.0)

    decision = controller.decide(
        [
            _det(1.0, valid=True, in_corridor=False, sigma=0.2),
            _det(48.0, valid=True, in_corridor=True, sigma=0.2),
        ],
        SceneQuality(),
        {
            "selected_target_valid_for_safety": True,
            "ttc_valid_for_safety": True,
            "target_ttc_s": 48.0,
            "target_track_id": 2,
        },
    )

    assert decision.mode == "nominal"
    assert "valid_ttc_below_threshold" not in decision.reason_codes
    assert decision.ttc_used_s == 48.0
