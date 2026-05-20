from __future__ import annotations

from ind_vias_perception.common.types import BBox2D, CameraCalibration, Detection, ObjectClass
from ind_vias_perception.geometry.scale_anchors.semantic_anchor import SemanticObjectSizeAnchor
from ind_vias_perception.geometry.scale_fusion.robust_distance import robust_fuse_distance_m


def test_semantic_distance_for_car_width():
    det = Detection(BBox2D(100, 100, 300, 250), ObjectClass.CAR, 0.9)

    anchor = SemanticObjectSizeAnchor({"car": {"width_m": 1.75}}).estimate(det, fy_px=1100, fx_px=1000)

    assert anchor.scale_or_distance_m == 8.75
    assert det.metadata["distance_semantic_m"] == 8.75


def test_fusion_averages_when_ground_and_semantic_agree():
    cal = CameraCalibration(1100, 1100, 720, 720, 1.25, 0.0, 980)

    distance, source = robust_fuse_distance_m(10.0, 12.0, cal)

    assert distance == 11.0
    assert source == "fused"


def test_fusion_prefers_semantic_when_ground_explodes_near_horizon():
    cal = CameraCalibration(1100, 1100, 720, 720, 1.25, 0.0, 980)

    distance, source = robust_fuse_distance_m(80.0, 12.0, cal, prefer_semantic=True)

    assert distance == 12.0
    assert source == "semantic"
