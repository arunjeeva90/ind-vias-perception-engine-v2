from __future__ import annotations

import csv
import subprocess
import sys


def test_offline_scenario_mining_script_loads_csv_and_prints_summary(tmp_path):
    csv_path = tmp_path / "debug.csv"
    rows = [
        {
            "cutin_warning_candidate": "cut_in_risk",
            "cutin_warning_confirmed": "none",
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

    result = subprocess.run(
        [sys.executable, "scripts/analyze_scenarios_from_csv.py", str(csv_path)],
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
