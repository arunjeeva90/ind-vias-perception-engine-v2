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


def test_invalid_side_objects_do_not_force_cais_critical():
    controller = CAISController(ignore_invalid_side_objects=True)

    decision = controller.decide([_det(1.0, valid=False, in_corridor=False)], SceneQuality())

    assert decision.mode == "nominal"
    assert decision.score == 0.0
    assert decision.reason_codes == "nominal"


def test_valid_low_ttc_target_can_force_cais_critical():
    controller = CAISController(critical_ttc_s=3.0, critical_score_threshold=0.75)

    decision = controller.decide([_det(1.0, valid=True, in_corridor=True, sigma=0.2)], SceneQuality())

    assert decision.mode == "critical"
    assert "valid_ttc_below_threshold" in decision.reason_codes
    assert decision.score >= 0.75
