from __future__ import annotations

from ind_vias_perception.common.types import BBox2D, Detection, ObjectClass, SceneQuality
from ind_vias_perception.ttc.cutin.lateral_cutin import LateralCutInDetector


EGO_CORRIDOR = {
    "enabled": True,
    "center_x_norm": 0.5,
    "bottom_width_norm": 0.45,
    "top_width_norm": 0.18,
    "top_y_norm": 0.45,
    "bottom_y_norm": 1.0,
}


def _det(
    track_id: int,
    u_gc: float,
    distance_m: float = 12.0,
    relevance: float = 0.5,
    distance_valid: bool = True,
    label: ObjectClass = ObjectClass.CAR,
    bbox_width: float = 120.0,
) -> Detection:
    det = Detection(BBox2D(u_gc - bbox_width * 0.5, 700, u_gc + bbox_width * 0.5, 820), label, 0.9)
    det.track_id = track_id
    det.distance_m = distance_m
    det.metadata["u_gc"] = u_gc
    det.metadata["v_gc"] = 820.0
    det.metadata["distance_bumper_m"] = distance_m
    det.metadata["distance_valid_for_safety"] = distance_valid
    det.metadata["target_relevance"] = relevance
    return det


def _detector() -> LateralCutInDetector:
    return LateralCutInDetector(
        enabled=True,
        history_size=10,
        min_history=5,
        lateral_velocity_threshold_px_s=25.0,
        max_relevant_distance_m=22.0,
        lateral_ttc_threshold_s=2.8,
        crossing_cfg={
            "enabled": True,
            "min_history_count": 8,
            "min_lateral_displacement_px": 60,
            "max_distance_m": 25.0,
            "min_confidence": 0.7,
        },
        ego_corridor=EGO_CORRIDOR,
    )


def _run_sequence(
    detector: LateralCutInDetector,
    xs: list[float],
    distance_m: float = 12.0,
    relevance: float = 0.5,
    distance_valid: bool = True,
    scene: SceneQuality | None = None,
    label: ObjectClass = ObjectClass.CAR,
    timestamp_step_s: float = 0.1,
    bbox_width: float = 120.0,
):
    result = []
    for idx, x in enumerate(xs):
        result = detector.update(
            [_det(1, x, distance_m, relevance, distance_valid, label, bbox_width)],
            timestamp_s=idx * timestamp_step_s,
            image_width=1000,
            image_height=1000,
            scene=scene or SceneQuality(),
        )
    return result[0]


def test_object_moving_from_right_toward_corridor_becomes_right_cut_in():
    det = _run_sequence(_detector(), [975, 925, 875, 825, 775, 725], timestamp_step_s=1.0, bbox_width=200.0)

    assert det.metadata["side_state"] == "RIGHT"
    assert det.metadata["cutin_state"] == "RIGHT_CUT_IN"
    assert det.metadata["cut_in_risk"] is True
    assert det.metadata["cutin_valid_for_safety"] is True
    assert det.metadata["ttc_lateral_s"] < 2.8


def test_object_moving_from_left_toward_corridor_becomes_left_cut_in():
    det = _run_sequence(_detector(), [25, 75, 125, 175, 225, 275], timestamp_step_s=1.0, bbox_width=200.0)

    assert det.metadata["side_state"] == "LEFT"
    assert det.metadata["cutin_state"] == "LEFT_CUT_IN"
    assert det.metadata["cut_in_risk"] is True


def test_object_already_in_corridor_becomes_in_path():
    det = _run_sequence(_detector(), [500, 502, 504, 506, 508])

    assert det.metadata["side_state"] == "IN"
    assert det.metadata["cutin_state"] == "IN_PATH"
    assert det.metadata["cut_in_risk"] is False
    assert det.metadata["cutin_valid_for_safety"] is False
    assert "in_path_longitudinal_only" in det.metadata["cutin_reason_codes"]


def test_far_object_beyond_max_distance_does_not_become_cut_in_risk():
    det = _run_sequence(_detector(), [930, 900, 870, 840, 810], distance_m=35.0)

    assert det.metadata["side_state"] == "RIGHT"
    assert det.metadata["cutin_state"] == "NONE"
    assert det.metadata["cut_in_risk"] is False


def test_turning_suppresses_or_lowers_cut_in_confidence():
    detector = _detector()
    result = []
    scene = SceneQuality(ego_motion_state="turning", yaw_confidence=0.9)
    for idx, x in enumerate([930, 900, 870, 840, 810]):
        result = detector.update(
            [_det(1, x)],
            timestamp_s=idx * 0.1,
            image_width=1000,
            image_height=1000,
            scene=scene,
        )

    det = result[0]
    assert det.metadata["cutin_confidence"] < 0.5
    assert det.metadata["cut_in_risk"] is False


def test_low_relevance_side_object_does_not_create_cut_in_risk():
    det = _run_sequence(_detector(), [930, 900, 870, 840, 810], relevance=0.1, bbox_width=40.0)

    assert det.metadata["cutin_state"] == "RIGHT_CUT_IN"
    assert det.metadata["cutin_valid_for_safety"] is False
    assert det.metadata["cut_in_risk"] is False
    assert "low_relevance_no_crossing_trend" in det.metadata["cutin_reason_codes"]


def test_invalid_distance_suppresses_cut_in_risk():
    det = _run_sequence(
        _detector(),
        [975, 925, 875, 825, 775, 725],
        relevance=0.8,
        distance_valid=False,
        timestamp_step_s=1.0,
        bbox_width=200.0,
    )

    assert det.metadata["cutin_valid_for_safety"] is False
    assert det.metadata["cut_in_risk"] is False
    assert "invalid_distance_for_safety" in det.metadata["cutin_reason_codes"]


def test_near_image_boundary_suppresses_cut_in_risk():
    det = _run_sequence(_detector(), [1110, 1080, 1050, 1020, 990], relevance=0.8)

    assert det.metadata["cutin_valid_for_safety"] is False
    assert det.metadata["cut_in_risk"] is False
    assert "near_image_boundary" in det.metadata["cutin_reason_codes"]


def test_valid_high_confidence_side_object_with_corridor_overlap_creates_cut_in_risk():
    det = _run_sequence(
        _detector(),
        [975, 925, 875, 825, 775, 725],
        relevance=0.1,
        timestamp_step_s=1.0,
        bbox_width=200.0,
    )

    assert det.metadata["cutin_state"] == "RIGHT_CUT_IN"
    assert det.metadata["corridor_overlap_ratio"] >= 0.15
    assert det.metadata["cutin_crossing_trend"] is True
    assert det.metadata["cutin_valid_for_safety"] is True
    assert det.metadata["cut_in_risk"] is True
    assert det.metadata["cutin_reason_codes"] == "eligible_cut_in"


def test_uncertain_ego_motion_suppresses_cut_in_risk():
    det = _run_sequence(
        _detector(),
        [975, 925, 875, 825, 775, 725],
        relevance=0.8,
        timestamp_step_s=1.0,
        bbox_width=200.0,
        scene=SceneQuality(ego_motion_state="uncertain", yaw_confidence=0.4),
    )

    assert det.metadata["cutin_valid_for_safety"] is False
    assert det.metadata["cut_in_risk"] is False
    assert "ego_not_straight" in det.metadata["cutin_reason_codes"]


def test_low_relevance_overlap_without_crossing_trend_does_not_create_cut_in_risk():
    det = _run_sequence(_detector(), [690, 690, 690, 690, 690], relevance=0.1)

    assert det.metadata["corridor_overlap_ratio"] >= 0.15
    assert det.metadata["cutin_crossing_trend"] is False
    assert det.metadata["cutin_valid_for_safety"] is False
    assert det.metadata["cut_in_risk"] is False
    assert "low_relevance_no_crossing_trend" in det.metadata["cutin_reason_codes"]


def test_low_relevance_real_side_to_in_crossing_trend_can_create_cut_in_risk():
    det = _run_sequence(
        _detector(),
        [975, 925, 875, 825, 775, 725],
        relevance=0.1,
        timestamp_step_s=1.0,
        bbox_width=200.0,
    )

    assert det.metadata["cutin_entry_side"] == "RIGHT"
    assert det.metadata["cutin_crossing_trend"] is True
    assert det.metadata["cutin_warning_eligible"] is True
    assert det.metadata["cut_in_risk"] is True


def test_high_relevance_side_object_can_still_create_cut_in_risk():
    det = _run_sequence(
        _detector(),
        [975, 925, 875, 825, 775, 725],
        relevance=0.8,
        timestamp_step_s=1.0,
        bbox_width=200.0,
    )

    assert det.metadata["cutin_valid_for_safety"] is True
    assert det.metadata["cut_in_risk"] is True


def test_in_path_remains_longitudinal_only():
    det = _run_sequence(_detector(), [500, 502, 504, 506, 508], relevance=0.8)

    assert det.metadata["cutin_state"] == "IN_PATH"
    assert det.metadata["cutin_warning_eligible"] is False
    assert det.metadata["cut_in_risk"] is False


def test_side_object_with_lateral_bbox_jitter_does_not_become_cut_in_risk():
    det = _run_sequence(_detector(), [900, 905, 895, 902, 898, 901], relevance=0.8)

    assert det.metadata["lateral_motion_stable"] is False
    assert det.metadata["cutin_valid_for_safety"] is False
    assert det.metadata["cut_in_risk"] is False


def test_tiny_lateral_ttc_from_jitter_is_rejected():
    det = _run_sequence(_detector(), [760, 720, 680, 640, 600], relevance=0.8, timestamp_step_s=0.1, bbox_width=200.0)

    assert det.metadata["ttc_lateral_s"] is None or det.metadata["ttc_lateral_s"] < 0.4
    assert det.metadata["cutin_valid_for_safety"] is False
    assert det.metadata["cut_in_risk"] is False


def test_pedestrian_left_to_right_crossing_is_not_vehicle_cut_in():
    det = _run_sequence(
        _detector(),
        [240, 290, 340, 390, 440, 490, 540, 590],
        relevance=0.8,
        label=ObjectClass.PEDESTRIAN,
        timestamp_step_s=1.0,
        bbox_width=60.0,
    )

    assert det.metadata["crossing_state"] == "left_to_right"
    assert det.metadata["crossing_valid_for_safety"] is True
    assert det.metadata["crossing_confidence"] > 0.0
    assert det.metadata["cut_in_risk"] is False


def test_pedestrian_right_to_left_crossing_is_not_vehicle_cut_in():
    det = _run_sequence(
        _detector(),
        [760, 710, 660, 610, 560, 510, 460, 410],
        relevance=0.8,
        label=ObjectClass.PEDESTRIAN,
        timestamp_step_s=1.0,
        bbox_width=60.0,
    )

    assert det.metadata["crossing_state"] == "right_to_left"
    assert det.metadata["crossing_valid_for_safety"] is True
    assert det.metadata["crossing_confidence"] > 0.0
    assert det.metadata["cut_in_risk"] is False


def test_pedestrian_with_short_history_remains_uncertain_or_none():
    det = _run_sequence(
        _detector(),
        [240, 290, 340, 390, 440],
        label=ObjectClass.PEDESTRIAN,
        timestamp_step_s=1.0,
        bbox_width=60.0,
    )

    assert det.metadata["crossing_state"] in {"uncertain", "none"}
    assert det.metadata["crossing_valid_for_safety"] is False
    assert "insufficient_history" in det.metadata["crossing_reason_codes"]


def test_pedestrian_with_small_jitter_remains_none():
    det = _run_sequence(
        _detector(),
        [430, 434, 429, 433, 431, 435, 430, 432],
        label=ObjectClass.PEDESTRIAN,
        timestamp_step_s=1.0,
        bbox_width=60.0,
    )

    assert det.metadata["crossing_state"] == "none"
    assert det.metadata["crossing_valid_for_safety"] is False
    assert "low_lateral_displacement" in det.metadata["crossing_reason_codes"]


def test_side_parallel_two_wheeler_remains_parallel_not_crossing():
    det = _run_sequence(
        _detector(),
        [740, 770, 800, 830, 860, 890, 920, 950],
        label=ObjectClass.MOTORCYCLE,
        timestamp_step_s=1.0,
        bbox_width=80.0,
    )

    assert det.metadata["crossing_state"] == "parallel"
    assert det.metadata["crossing_valid_for_safety"] is False
    assert "not_approaching_corridor" in det.metadata["crossing_reason_codes"]


def test_far_pedestrian_crossing_is_not_valid_for_safety():
    det = _run_sequence(
        _detector(),
        [240, 290, 340, 390, 440, 490, 540, 590],
        distance_m=40.0,
        label=ObjectClass.PEDESTRIAN,
        timestamp_step_s=1.0,
        bbox_width=60.0,
    )

    assert det.metadata["crossing_state"] == "left_to_right"
    assert det.metadata["crossing_valid_for_safety"] is False
    assert "too_far" in det.metadata["crossing_reason_codes"]


def test_near_boundary_pedestrian_crossing_is_suppressed():
    det = _run_sequence(
        _detector(),
        [600, 650, 700, 750, 800, 850, 900, 970],
        label=ObjectClass.PEDESTRIAN,
        timestamp_step_s=1.0,
        bbox_width=60.0,
    )

    assert det.metadata["crossing_state"] == "left_to_right"
    assert det.metadata["crossing_valid_for_safety"] is False
    assert det.metadata["crossing_boundary_suppressed"] is True
    assert "near_boundary" in det.metadata["crossing_reason_codes"]


def test_car_lateral_movement_is_not_classified_as_vru_crossing():
    det = _run_sequence(
        _detector(),
        [240, 290, 340, 390, 440, 490, 540, 590],
        label=ObjectClass.CAR,
        timestamp_step_s=1.0,
        bbox_width=200.0,
    )

    assert det.metadata["crossing_state"] == "none"
    assert det.metadata["crossing_valid_for_safety"] is False
    assert "non_vru_class" in det.metadata["crossing_reason_codes"]
