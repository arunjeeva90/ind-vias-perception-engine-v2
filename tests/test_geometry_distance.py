from ind_vias_perception.common.types import CameraCalibration
from ind_vias_perception.geometry.calibration.ground_distance import ground_contact_distance_m


def test_ground_contact_distance_decreases_for_lower_pixels():
    cal = CameraCalibration(1100, 1100, 640, 360, 1.25, 2.0, 374)
    far = ground_contact_distance_m(450, cal)
    near = ground_contact_distance_m(650, cal)
    assert near < far
