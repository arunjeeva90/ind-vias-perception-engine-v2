from __future__ import annotations

from ind_vias_perception.common.types import BBox2D, Detection, ObjectClass
from ind_vias_perception.geometry.scale_anchors.semantic_anchor import SemanticObjectSizeAnchor
from ind_vias_perception.perception.postprocess.compound_agents import group_compound_agents


_CFG = {
    "enable_two_wheeler_grouping": True,
    "motorcycle_expand_scale": 1.6,
    "min_iou": 0.02,
}


def test_rider_and_motorcycle_overlap_creates_two_wheeler_agent():
    rider = Detection(BBox2D(100, 100, 160, 260), ObjectClass.PEDESTRIAN, 0.8)
    motorcycle = Detection(BBox2D(90, 210, 190, 310), ObjectClass.MOTORCYCLE, 0.9)

    grouped = group_compound_agents([rider, motorcycle], _CFG)

    assert len(grouped) == 1
    assert grouped[0].label == ObjectClass.TWO_WHEELER_AGENT
    assert grouped[0].bbox == BBox2D(90, 100, 190, 310)
    assert grouped[0].metadata["compound_source"] == "rider_motorcycle"
    assert grouped[0].metadata["grouped_from"] == ["pedestrian", "motorcycle"]


def test_unrelated_pedestrian_is_not_grouped():
    rider = Detection(BBox2D(10, 10, 60, 120), ObjectClass.PEDESTRIAN, 0.8)
    motorcycle = Detection(BBox2D(300, 210, 400, 310), ObjectClass.MOTORCYCLE, 0.9)

    grouped = group_compound_agents([rider, motorcycle], _CFG)

    assert grouped == [rider, motorcycle]


def test_motorcycle_without_rider_remains_motorcycle():
    motorcycle = Detection(BBox2D(90, 210, 190, 310), ObjectClass.MOTORCYCLE, 0.9)

    grouped = group_compound_agents([motorcycle], _CFG)

    assert grouped == [motorcycle]


def test_grouped_agent_receives_semantic_distance():
    rider = Detection(BBox2D(100, 100, 160, 260), ObjectClass.PEDESTRIAN, 0.8)
    motorcycle = Detection(BBox2D(90, 210, 190, 310), ObjectClass.MOTORCYCLE, 0.9)
    grouped = group_compound_agents([rider, motorcycle], _CFG)

    anchor = SemanticObjectSizeAnchor(
        {"two_wheeler_agent": {"width_m": 0.90, "height_m": 1.65}}
    ).estimate(grouped[0], fy_px=1100, fx_px=1100)

    expected_width_distance = 1100 * 0.90 / grouped[0].bbox.width
    expected_height_distance = 1100 * 1.65 / grouped[0].bbox.height
    assert anchor.scale_or_distance_m == (expected_width_distance + expected_height_distance) * 0.5
    assert grouped[0].metadata["distance_semantic_m"] == anchor.scale_or_distance_m
