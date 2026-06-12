from __future__ import annotations

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.occupancy import CabinOccupancyManager
from ind_vias_dms.core.occupant_manager import CabinOccupantManager, OccupantSelection, TrackedFace
from ind_vias_dms.core.pipeline import DMSPipeline
from ind_vias_dms.vision.face_landmarks import FaceLandmarkResult, FaceQualityResult


def _face(box: tuple[int, int, int, int], confidence: float = 0.8) -> FaceLandmarkResult:
    x1, y1, x2, y2 = box
    area_norm = ((x2 - x1) / 1000.0) * ((y2 - y1) / 1000.0)
    return FaceLandmarkResult(
        face_found=True,
        bbox=box,
        confidence=confidence,
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        area=float((x2 - x1) * (y2 - y1)),
        box_norm=(x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0),
        quality=FaceQualityResult(
            proposal_confidence=confidence,
            landmark_count=160,
            landmark_coverage_score=0.85,
            face_box_area_norm=area_norm,
            face_aspect_ratio=(x2 - x1) / max(1.0, float(y2 - y1)),
            left_eye_visible=True,
            right_eye_visible=True,
            nose_visible=True,
            mouth_visible=True,
            chin_visible=True,
            both_eyes_available=True,
            face_completeness_score=0.95,
            is_valid_driver_face=True,
            validation_state="VALIDATED_FULL_FACE",
            rejection_reason_codes=[],
        ),
    )


def _selection(
    faces: list[TrackedFace],
    rejected: list[dict[str, object]] | None = None,
) -> OccupantSelection:
    driver = next((face for face in faces if face.zone == "DRIVER"), None)
    return OccupantSelection(
        faces=faces,
        driver=driver,
        proposal_count=len(faces) + len(rejected or []),
        unconfirmed_proposal_count=len(rejected or []),
        rejected_proposals=rejected or [],
    )


def test_rhd_driver_image_side_left_driver_roi_is_image_left():
    manager = CabinOccupantManager(DMSConfig(driver_image_side="LEFT", auto_generate_rois_from_layout=True))

    assert manager._roi("driver_roi_norm")[0] == 0.0
    assert manager._roi("front_passenger_roi_norm")[0] >= 0.45


def test_generated_seat_rois_are_available():
    manager = CabinOccupantManager(DMSConfig(driver_image_side="LEFT", auto_generate_rois_from_layout=True))

    assert manager._roi("rear_left_roi_norm")[2] <= 0.36
    assert manager._roi("rear_center_roi_norm")[0] < 0.5 < manager._roi("rear_center_roi_norm")[2]
    assert manager._roi("rear_right_roi_norm")[0] >= 0.64


def test_single_driver_no_rear_false_positive():
    driver = TrackedFace(_face((100, 200, 350, 700)), 1, "DRIVER", True)
    occupancy = CabinOccupancyManager(DMSConfig()).update(_selection([driver]), 0)

    assert occupancy.cabin_occupant_count == 1
    assert occupancy.driver_present is True
    assert occupancy.front_passenger_present is False
    assert occupancy.rear_left_present == "unknown"


def test_rear_passenger_stable_partial_detection_becomes_partial_present():
    config = DMSConfig(rear_occupant_confirm_frames=2)
    manager = CabinOccupancyManager(config)
    rejected = [
        {
            "zone": "REAR_LEFT",
            "box_norm": [0.08, 0.15, 0.18, 0.34],
            "reason_codes": ["TEMPORAL_CONFIRMATION_PENDING"],
        }
    ]

    manager.update(_selection([], rejected), 0)
    occupancy = manager.update(_selection([], rejected), 100)

    assert occupancy.rear_left_present == "partial"
    assert occupancy.seats["rear_left"].detection_source == "PARTIAL_FACE"


def test_unstable_rear_detection_does_not_mark_present():
    config = DMSConfig(rear_occupant_confirm_frames=5)
    manager = CabinOccupancyManager(config)
    rejected = [
        {
            "zone": "REAR_RIGHT",
            "box_norm": [0.75, 0.15, 0.84, 0.30],
            "reason_codes": ["TEMPORAL_CONFIRMATION_PENDING"],
        }
    ]

    occupancy = manager.update(_selection([], rejected), 0)

    assert occupancy.rear_right_present == "possible"
    assert occupancy.cabin_occupant_count == 0


def test_headrest_suppression_blocks_rear_occupant():
    manager = CabinOccupancyManager(DMSConfig())
    rejected = [
        {
            "zone": "REAR_CENTER",
            "box_norm": [0.45, 0.20, 0.51, 0.34],
            "reason_codes": ["HEADREST_LIKE_STATIC_BOX"],
        }
    ]

    occupancy = manager.update(_selection([], rejected), 0)

    assert occupancy.rear_center_present == "unknown"
    assert "STATIC_OBJECT_SUPPRESSED" in occupancy.occupancy_reason_codes


def test_front_passenger_does_not_replace_driver():
    manager = CabinOccupantManager(DMSConfig(driver_image_side="LEFT"))
    driver = _face((100, 200, 330, 710), confidence=0.7)
    passenger = _face((650, 180, 930, 760), confidence=0.95)

    selection = manager.update([passenger, driver], (1000, 1000, 3), 0)

    assert selection.driver is not None
    assert selection.driver.zone == "DRIVER"


def test_occupancy_update_does_not_reset_pipeline_perclos():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.perclos_short = type("FakeTracker", (), {"value": 1})()
    manager = CabinOccupancyManager(DMSConfig())

    manager.update(_selection([TrackedFace(_face((100, 200, 350, 700)), 1, "DRIVER", True)]), 0)

    assert pipeline.perclos_short.value == 1


def test_occupancy_json_compatible_in_dms_state():
    from ind_vias_dms.core.types import DMSState, OccupancyState

    state = DMSState(occupancy=OccupancyState(cabin_occupant_count=1, driver_present=True))
    payload = state.to_dict()

    assert payload["occupancy"]["cabin_occupant_count"] == 1
    assert "driver_presence" in payload
