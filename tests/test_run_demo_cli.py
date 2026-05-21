from __future__ import annotations

import csv

import pytest

from ind_vias_perception.apps.run_demo import (
    DEBUG_CSV_COLUMNS,
    build_parser,
    validate_detector_config,
    write_debug_csv_row,
)
from ind_vias_perception.common.types import PerceptionOutput, SceneQuality


def test_cli_parser_accepts_image_video_output_show_and_max_frames():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--image",
            "frame.jpg",
            "--video",
            "clip.mp4",
            "--output",
            "annotated.mp4",
            "--show",
            "--debug-overlay",
            "--debug-csv",
            "debug.csv",
            "--max-frames",
            "7",
        ]
    )

    assert args.image == "frame.jpg"
    assert args.video == "clip.mp4"
    assert args.output == "annotated.mp4"
    assert args.show is True
    assert args.debug_overlay is True
    assert args.debug_csv == "debug.csv"
    assert args.max_frames == 7


def test_onnx_detector_config_missing_model_has_clear_error(tmp_path):
    missing_model = tmp_path / "detector.onnx"

    with pytest.raises(SystemExit) as exc:
        validate_detector_config(
            {
                "detection": {
                    "backend": "onnx",
                    "onnx_model_path": str(missing_model),
                }
            }
        )

    message = str(exc.value)
    assert "ONNX detector model is missing" in message
    assert "models/weights/detector.onnx" in message
    assert "configs/default.yaml" in message


def test_debug_csv_creation_and_expected_columns(tmp_path):
    output_path = tmp_path / "safety_debug.csv"
    output = PerceptionOutput(
        detections=[],
        scene_quality=SceneQuality(),
        mode="nominal",
        safety_payload={
            "target_track_id": 3,
            "selected_target_valid_for_safety": True,
            "selected_target_reason": "valid_safety_target",
            "debug_target_track_id": 4,
            "debug_target_distance_valid_for_safety": False,
            "target_distance_m": 12.5,
            "target_ttc_s": 2.4,
            "target_in_ego_corridor": True,
            "target_relevance": 0.9,
            "raw_warning_level": "visual",
            "confirmed_warning_level": "visual",
            "warning_candidate": "visual",
            "warning_suppressed_reason": None,
            "ego_motion_state": "straight",
            "yaw_confidence": 0.0,
            "cais_mode": "nominal",
            "cais_score": 0.0,
            "cais_reason_codes": "nominal",
            "cais_ttc_used_s": None,
            "cais_ttc_threshold_s": 3.0,
            "cais_ttc_source_track_id": None,
            "ttc_valid_for_safety": False,
            "ttc_reason_codes": "ttc_missing",
            "sentinel_state": "nominal",
        },
    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DEBUG_CSV_COLUMNS)
        writer.writeheader()
        write_debug_csv_row(writer, 5, 0.25, output)

    with open(output_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == DEBUG_CSV_COLUMNS
    assert len(rows) == 1
    assert rows[0]["frame_index"] == "5"
    assert rows[0]["timestamp_s"] == "0.25"
    assert rows[0]["selected_target_track_id"] == "3"
    assert rows[0]["selected_target_reason"] == "valid_safety_target"
    assert rows[0]["cais_score"] == "0.0"
    assert rows[0]["cais_ttc_threshold_s"] == "3.0"
    assert rows[0]["ttc_reason_codes"] == "ttc_missing"
