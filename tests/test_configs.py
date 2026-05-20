from __future__ import annotations

import yaml

from ind_vias_perception.config.settings import load_settings


def test_yolov8n_coco_demo_uses_coco_adas_subset():
    with open("configs/yolov8n_coco_demo.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["detection"]["backend"] == "onnx"
    assert cfg["detection"]["onnx_model_path"] == "models/weights/detector.onnx"
    assert cfg["detection"]["class_names"] == {
        0: "pedestrian",
        1: "cyclist",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }
    assert "auto_rickshaw" not in cfg["detection"]["class_names"].values()
    assert "animal" not in cfg["detection"]["class_names"].values()


def test_yolov8n_coco_demo_camera_aliases_load():
    settings = load_settings("configs/yolov8n_coco_demo.yaml")

    assert settings.camera.image_width == 1248
    assert settings.camera.image_height == 1248
    assert settings.camera.fx_px == 1100.0
    assert settings.camera.fy_px == 1100.0
    assert settings.camera.cx_px == 624.0
    assert settings.camera.cy_px == 624.0
    assert settings.camera.height_m == 1.25
    assert settings.camera.pitch_deg == 0.0
    assert settings.camera.horizon_v_px == 640.0
    assert settings.camera.min_distance_m == 2.0
    assert settings.camera.max_distance_m == 120.0
    assert settings.vehicle.camera_to_front_bumper_offset_m == 1.45


def test_missing_vehicle_config_defaults_to_zero_offset():
    settings = load_settings("configs/default.yaml")

    assert settings.vehicle.camera_to_front_bumper_offset_m == 0.0


def test_phone_demo_1440_profile_loads():
    settings = load_settings("configs/phone_demo_1440.yaml")

    assert settings.camera.image_width == 1440
    assert settings.camera.image_height == 1440
    assert settings.camera.fy_px == 1100.0
    assert settings.vehicle.camera_to_front_bumper_offset_m == 1.45

    with open("configs/phone_demo_1440.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert settings.camera.horizon_v_px == float(cfg["camera"]["horizon_y"])
    assert cfg["semantic_priors"]["car"]["width_m"] == 1.75
    assert cfg["semantic_priors"]["truck"]["height_m"] == 2.80
    assert cfg["ego_corridor"]["enabled"] is True
    assert cfg["ego_corridor"]["bottom_width_norm"] == 0.45
