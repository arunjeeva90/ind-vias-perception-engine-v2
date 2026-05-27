from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize IND-VIAS debug CSV scenarios.")
    parser.add_argument("csv_path")
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_rows(Path(args.csv_path))
    summary = analyze_rows(rows)
    print_summary(summary)
    if args.output:
        write_report(Path(args.output), summary)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def analyze_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    cutin_candidates = [r for r in rows if _lower(r.get("cutin_warning_candidate")) == "cut_in_risk"]
    confirmed_cutins = [r for r in rows if _lower(r.get("cutin_warning_confirmed")) == "cut_in_risk"]
    low_relevance = [
        r for r in cutin_candidates if _float(r.get("target_relevance")) < 0.5
    ]
    tiny_ttc = [
        r for r in cutin_candidates if 0.0 <= _float(r.get("ttc_lateral_s"), 1e9) < 0.4
    ]
    near_boundary = [
        r for r in cutin_candidates if "near_image_boundary" in _lower(r.get("cutin_reason_codes"))
    ]
    night_glare = [
        r for r in rows if "scene_quality_cut_in_suppressed" in _lower(r.get("cutin_reason_codes"))
    ]
    invalid_distance = [
        r for r in rows if "invalid_distance_for_safety" in _lower(r.get("cutin_reason_codes"))
    ]
    false_patterns = Counter()
    for row in cutin_candidates:
        for reason in str(row.get("cutin_reason_codes", "")).split(","):
            reason = reason.strip()
            if reason and reason != "eligible_cut_in":
                false_patterns[reason] += 1
    crossing_by_track: dict[str, set[str]] = defaultdict(set)
    valid_crossing_rows = [r for r in rows if _truthy(r.get("crossing_valid_for_safety"))]
    non_safety_crossing_rows = [
        r
        for r in rows
        if _lower(r.get("crossing_state")) not in {"", "none", "nan"}
        and not _truthy(r.get("crossing_valid_for_safety"))
    ]
    crossing_reason_counts = Counter()
    for row in rows:
        if _truthy(row.get("crossing_valid_for_safety")):
            continue
        for reason in str(row.get("crossing_reason_codes", "")).split(","):
            reason = reason.strip()
            if reason:
                crossing_reason_counts[reason] += 1
    for row in rows:
        state = row.get("crossing_state", "none")
        if state and state != "none" and _truthy(row.get("crossing_valid_for_safety")):
            crossing_by_track[row.get("selected_target_track_id", "unknown")].add(state)

    return {
        "total_frames": len(rows),
        "cut_in_candidate_frames": len(cutin_candidates),
        "confirmed_cut_in_frames": len(confirmed_cutins),
        "pedestrian_crossing_like_tracks": sum(
            bool(states - {"parallel", "uncertain"}) for states in crossing_by_track.values()
        ),
        "repeated_false_cut_in_patterns": dict(false_patterns),
        "low_relevance_cut_in_candidates": len(low_relevance),
        "tiny_lateral_ttc_candidates": len(tiny_ttc),
        "near_boundary_candidates": len(near_boundary),
        "night_glare_conservative_suppressions": len(night_glare),
        "invalid_distance_suppressed_targets": len(invalid_distance),
        "crossing_valid_for_safety_count": len(valid_crossing_rows),
        "valid_crossing_states": dict(Counter(row.get("crossing_state", "none") for row in valid_crossing_rows)),
        "suppressed_crossing_counts_by_reason": dict(crossing_reason_counts),
        "vru_crossing_candidate_count": len(
            [r for r in rows if _lower(r.get("crossing_state")) in {"left_to_right", "right_to_left"}]
        ),
        "non_safety_crossing_state_count": len(non_safety_crossing_rows),
        "crossing_tracks_by_state": dict(
            Counter(row.get("crossing_state", "none") for row in valid_crossing_rows)
        ),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("Scenario mining summary")
    for key, value in summary.items():
        print(f"{key}: {value}")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, json.dumps(value) if isinstance(value, dict) else value])


def _lower(value: object) -> str:
    return str(value or "").lower()


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


if __name__ == "__main__":
    main()
