from __future__ import annotations

import csv

import pytest

from ind_vias_perception.apps.run_demo import (
    DEBUG_CSV_COLUMNS,
    build_parser,
    should_process_frame,
    validate_detector_config,
    video_run_summary,
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
            "side_state": "RIGHT",
            "cutin_state": "RIGHT_CUT_IN",
            "ttc_lateral_s": 1.2,
            "cutin_confidence": 0.8,
            "cutin_valid_for_safety": True,
            "cutin_reason_codes": "ok",
            "lateral_velocity_px_s": -300.0,
            "lateral_history_count": 5,
            "corridor_overlap_ratio": 0.25,
            "corridor_overlap_delta": 0.16,
            "corridor_entry_confirmed": True,
            "lateral_motion_stable": True,
            "lateral_center_history_count": 5,
            "lateral_velocity_px_s_smoothed": -250.0,
            "cutin_crossing_trend": True,
            "cutin_entry_side": "RIGHT",
            "cutin_warning_eligible": True,
            "cutin_warning_candidate": "cut_in_risk",
            "cutin_warning_confirmed": "cut_in_risk",
            "cutin_target_track_id": 3,
            "crossing_state": "none",
            "crossing_confidence": 0.0,
            "crossing_history_count": 5,
            "crossing_valid_for_safety": False,
            "crossing_reason_codes": "non_vru_class",
            "crossing_lateral_displacement_px": 0.0,
            "crossing_corridor_approach": False,
            "crossing_boundary_suppressed": False,
            "crossing_tiny_object_suppressed": False,
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
    assert rows[0]["cutin_state"] == "RIGHT_CUT_IN"
    assert rows[0]["cutin_valid_for_safety"] == "True"
    assert rows[0]["cutin_reason_codes"] == "ok"
    assert rows[0]["lateral_history_count"] == "5"
    assert rows[0]["cutin_crossing_trend"] == "True"
    assert rows[0]["corridor_entry_confirmed"] == "True"
    assert rows[0]["lateral_motion_stable"] == "True"
    assert rows[0]["crossing_state"] == "none"
    assert rows[0]["crossing_valid_for_safety"] == "False"
    assert rows[0]["crossing_reason_codes"] == "non_vru_class"
    assert rows[0]["cutin_warning_eligible"] == "True"
    assert rows[0]["cutin_target_track_id"] == "3"


def test_no_max_frames_means_no_artificial_frame_limit():
    assert should_process_frame(0, None) is True
    assert should_process_frame(10_000, None) is True
    summary = video_run_summary(30.0, 900, None)
    assert "max_frames=none" in summary
    assert "mode=full video" in summary


def test_max_frames_limits_processing_when_provided():
    assert should_process_frame(299, 300) is True
    assert should_process_frame(300, 300) is False
    summary = video_run_summary(30.0, 900, 300)
    assert "max_frames=300" in summary
    assert "mode=limited debug run" in summary
