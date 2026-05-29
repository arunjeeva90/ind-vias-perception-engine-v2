from __future__ import annotations

import csv
import subprocess
import sys


def test_offline_scenario_mining_script_loads_csv_and_prints_summary(tmp_path):
    csv_path = tmp_path / "debug.csv"
    detections_csv_path = tmp_path / "detections.csv"
    rows = [
        {
            "cutin_warning_candidate": "cut_in_risk",
            "cutin_warning_confirmed": "none",
            "target_distance_valid_for_safety": "False",
            "target_class": "pedestrian",
            "target_in_ego_corridor": "True",
            "distance_confidence": "0.2",
            "distance_reason_codes": "near_horizon,tiny_bbox",
            "target_relevance": "0.2",
            "ttc_lateral_s": "0.1",
            "cutin_reason_codes": "lateral_ttc_too_low",
            "crossing_state": "left_to_right",
            "crossing_valid_for_safety": "True",
            "crossing_reason_codes": "valid_crossing",
            "selected_target_track_id": "4",
        },
        {
            "cutin_warning_candidate": "none",
            "cutin_warning_confirmed": "none",
            "target_distance_valid_for_safety": "True",
            "target_class": "car",
            "target_in_ego_corridor": "False",
            "distance_confidence": "0.8",
            "distance_reason_codes": "ok",
            "target_relevance": "0.9",
            "ttc_lateral_s": "",
            "cutin_reason_codes": "invalid_distance_for_safety",
            "crossing_state": "none",
            "crossing_valid_for_safety": "False",
            "crossing_reason_codes": "too_far",
            "selected_target_track_id": "5",
        },
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    detection_rows = [
        {
            "frame_index": "1",
            "track_id": "10",
            "object_class": "car",
            "distance_valid_for_safety": "False",
            "distance_reason_codes": "near_horizon,low_distance_confidence",
            "distance_confidence": "0.2",
            "target_in_ego_corridor": "True",
            "side_state": "IN",
            "distance_bumper_m": "9.0",
        },
        {
            "frame_index": "1",
            "track_id": "11",
            "object_class": "pedestrian",
            "distance_valid_for_safety": "False",
            "distance_reason_codes": "tiny_bbox",
            "distance_confidence": "0.1",
            "target_in_ego_corridor": "False",
            "side_state": "LEFT",
            "distance_bumper_m": "18.0",
        },
        {
            "frame_index": "2",
            "track_id": "12",
            "object_class": "car",
            "distance_valid_for_safety": "True",
            "distance_reason_codes": "ok",
            "distance_confidence": "0.8",
            "target_in_ego_corridor": "True",
            "side_state": "IN",
            "distance_bumper_m": "14.0",
        },
    ]
    with detections_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detection_rows[0].keys())
        writer.writeheader()
        writer.writerows(detection_rows)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_scenarios_from_csv.py",
            str(csv_path),
            "--detections-csv",
            str(detections_csv_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Scenario mining summary" in result.stdout
    assert "total_frames" in result.stdout
    assert "cut_in_candidate_frames" in result.stdout
    assert "tiny_lateral_ttc_candidates" in result.stdout
    assert "crossing_valid_for_safety_count" in result.stdout
    assert "suppressed_crossing_counts_by_reason" in result.stdout
    assert "invalid_distance_count_by_reason_code" in result.stdout
    assert "average_distance_confidence_invalid_targets" in result.stdout
    assert "total_detection_rows: 3" in result.stdout
    assert "near_horizon" in result.stdout


def test_scenario_mining_still_works_without_detection_csv(tmp_path):
    csv_path = tmp_path / "debug.csv"
    rows = [
        {
            "cutin_warning_candidate": "none",
            "cutin_warning_confirmed": "none",
            "target_distance_valid_for_safety": "False",
            "target_class": "car",
            "target_in_ego_corridor": "False",
            "distance_confidence": "0.2",
            "distance_reason_codes": "side_object",
            "target_relevance": "0.2",
            "ttc_lateral_s": "",
            "cutin_reason_codes": "",
            "crossing_state": "none",
            "crossing_valid_for_safety": "False",
            "crossing_reason_codes": "",
            "selected_target_track_id": "5",
        },
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = subprocess.run(
        [sys.executable, "scripts/analyze_scenarios_from_csv.py", str(csv_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Scenario mining summary" in result.stdout
    assert "total_detection_rows: 0" in result.stdout
    assert "invalid_distance_suppressed_targets: 1" in result.stdout
