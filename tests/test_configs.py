from __future__ import annotations

import yaml


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
