from ind_vias_perception.common.types import CameraCalibration
from ind_vias_perception.geometry.calibration.ground_distance import ground_contact_distance_m
from ind_vias_perception.geometry.scale_anchors.geometric_anchor import GeometricGroundContactAnchor
from ind_vias_perception.pipeline.metric_monocular_pipeline import bumper_relative_distance_m
from ind_vias_perception.perception.heads.ground_contact.dummy_ground_contact_head import DummyGroundContactHead
from ind_vias_perception.common.types import BBox2D, Detection, FramePacket, ObjectClass
import numpy as np


def test_ground_contact_distance_decreases_for_lower_pixels():
    cal = CameraCalibration(1100, 1100, 640, 360, 1.25, 2.0, 374)
    far = ground_contact_distance_m(450, cal)
    near = ground_contact_distance_m(650, cal)
    assert near < far


def test_ground_contact_fallback_uses_bbox_bottom_center():
    det = Detection(BBox2D(10, 20, 50, 80), ObjectClass.CAR, 0.9)
    packet = FramePacket(frame=np.zeros((100, 100, 3), dtype=np.uint8), timestamp_s=0.0)

    out = DummyGroundContactHead().forward([det], packet)

    assert out[0].ground_contact == (30.0, 80)
    assert out[0].metadata["u_gc"] == 30.0
    assert out[0].metadata["v_gc"] == 80.0


def test_ground_contact_distance_increases_closer_to_horizon():
    cal = CameraCalibration(1100, 1100, 624, 624, 1.25, 0.0, 640, min_distance_m=2.0, max_distance_m=120.0)

    farther = ground_contact_distance_m(660, cal)
    nearer = ground_contact_distance_m(1000, cal)

    assert farther > nearer


def test_invalid_ground_contact_above_horizon_is_safe():
    cal = CameraCalibration(1100, 1100, 624, 624, 1.25, 0.0, 640, min_distance_m=2.0, max_distance_m=120.0)
    det = Detection(BBox2D(10, 20, 50, 600), ObjectClass.CAR, 0.9)
    det.ground_contact = det.bbox.bottom_center

    anchor = GeometricGroundContactAnchor().estimate(det, cal)

    assert anchor.scale_or_distance_m == float("inf")
    assert det.metadata["u_gc"] == 30.0
    assert det.metadata["v_gc"] == 600.0
    assert det.metadata["raw_distance_m"] == float("inf")
    assert det.metadata["filtered_distance_m"] == float("inf")


def test_bumper_distance_subtracts_configured_offset():
    assert bumper_relative_distance_m(20.0, 1.45) == 18.55


def test_bumper_distance_never_becomes_negative():
    assert bumper_relative_distance_m(1.0, 1.45) == 0.0
