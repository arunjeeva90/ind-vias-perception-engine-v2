from ind_vias_perception.ttc.depth_ttc.depth_derivative import ttc_from_depth


def test_depth_ttc_for_closing_object():
    assert ttc_from_depth(20.0, -5.0) == 4.0
    assert ttc_from_depth(20.0, 1.0) is None
