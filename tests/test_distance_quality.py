from __future__ import annotations

from ind_vias_perception.common.types import BBox2D, Detection, ObjectClass
from ind_vias_perception.geometry.scale_fusion.distance_quality import evaluate_distance_quality


_CFG = {
    "min_bbox_area_ratio": {"car": 0.002},
    "near_horizon_margin_px": 40,
    "side_object_confidence_scale": 0.5,
    "disagreement_ratio_limit": 2.0,
}


def _det(bbox: BBox2D, in_corridor: bool = True) -> Detection:
    det = Detection(bbox, ObjectClass.CAR, 0.9)
    det.metadata["in_ego_corridor"] = in_corridor
    det.metadata["v_gc"] = bbox.y2
    det.metadata["distance_ground_m"] = 20.0
    det.metadata["distance_semantic_m"] = 22.0
    return det


def test_ego_object_gets_higher_confidence_than_side_object():
    ego = _det(BBox2D(500, 700, 760, 1000), True)
    side = _det(BBox2D(50, 700, 310, 1000), False)

    ego_conf, ego_rel, ego_valid, _ = evaluate_distance_quality(ego, 1440, 1440, 640, _CFG)
    side_conf, side_rel, side_valid, _ = evaluate_distance_quality(side, 1440, 1440, 640, _CFG)

    assert ego_conf > side_conf
    assert ego_rel > side_rel
    assert ego_valid is True
    assert side_valid is True


def test_near_horizon_object_gets_low_confidence():
    det = _det(BBox2D(500, 600, 760, 670), True)

    confidence, _, valid, reasons = evaluate_distance_quality(det, 1440, 1440, 640, _CFG)

    assert confidence < 0.5
    assert valid is False
    assert "near_horizon" in reasons
    assert "near_horizon" in det.metadata["distance_reason_codes"]
    assert det.metadata["ground_contact_row"] == 670.0
    assert det.metadata["horizon_y"] == 640.0


def test_tiny_bbox_gets_low_confidence():
    det = _det(BBox2D(700, 900, 725, 930), True)

    confidence, _, valid, reasons = evaluate_distance_quality(det, 1440, 1440, 640, _CFG)

    assert confidence < 0.5
    assert valid is False
    assert "tiny_bbox" in reasons
    assert det.metadata["is_tiny_bbox"] is True
    assert "tiny_bbox" in det.metadata["distance_reason_codes"]


def test_strong_ground_semantic_disagreement_lowers_confidence():
    det = _det(BBox2D(500, 700, 760, 1000), True)
    det.metadata["distance_ground_m"] = 80.0
    det.metadata["distance_semantic_m"] = 20.0

    confidence, _, valid, reasons = evaluate_distance_quality(det, 1440, 1440, 640, _CFG)

    assert confidence < 1.0
    assert valid is True
    assert "distance_disagreement" in reasons
    assert det.metadata["ground_semantic_ratio"] == 4.0


def test_side_object_includes_side_object_reason():
    det = _det(BBox2D(50, 700, 310, 1000), False)

    _, _, valid, reasons = evaluate_distance_quality(det, 1440, 1440, 640, _CFG)

    assert valid is True
    assert "side_object" in reasons
    assert det.metadata["is_side_object"] is True


def test_non_finite_distances_include_diagnostic_reasons_without_invalidating_by_itself():
    det = _det(BBox2D(500, 700, 760, 1000), True)
    det.metadata["distance_ground_m"] = float("inf")
    det.metadata["distance_semantic_m"] = float("inf")

    _, _, valid, reasons = evaluate_distance_quality(det, 1440, 1440, 640, _CFG)

    assert valid is True
    assert "non_finite_ground_distance" in reasons
    assert "non_finite_semantic_distance" in reasons
