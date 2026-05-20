from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "tune_ground_distance.py"
    spec = importlib.util.spec_from_file_location("tune_ground_distance", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["tune_ground_distance"] = module
    spec.loader.exec_module(module)
    return module


def test_tuning_math_estimates_current_distances_and_suggested_fy():
    module = _load_script_module()

    result = module.tune_ground_distance(
        bbox=(100.0, 200.0, 300.0, 1000.0),
        known_distance_m=10.0,
        camera_height_m=1.25,
        horizon_y=640.0,
        fy=1100.0,
        camera_to_front_bumper_offset_m=1.45,
    )

    assert result.u_gc == 200.0
    assert result.v_gc == 1000.0
    assert abs(result.distance_camera_m - 3.8194444444) < 1e-6
    assert abs(result.distance_bumper_m - 2.3694444444) < 1e-6
    assert result.suggested_fy == 2880.0


def test_tuning_math_suggests_horizon_for_fixed_fy_and_known_distance():
    module = _load_script_module()

    result = module.tune_ground_distance(
        bbox=(100.0, 200.0, 300.0, 1000.0),
        known_distance_m=10.0,
        camera_height_m=1.25,
        horizon_y=640.0,
        fy=1100.0,
    )

    assert result.suggested_horizon_y == 862.5


def test_tuning_math_handles_ground_contact_above_horizon():
    module = _load_script_module()

    result = module.tune_ground_distance(
        bbox=(100.0, 200.0, 300.0, 600.0),
        known_distance_m=10.0,
        camera_height_m=1.25,
        horizon_y=640.0,
        fy=1100.0,
    )

    assert result.distance_camera_m == float("inf")
