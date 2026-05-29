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
    parser.add_argument("--detections-csv", default=None)
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_rows(Path(args.csv_path))
    detection_rows = read_rows(Path(args.detections_csv)) if args.detections_csv else None
    summary = analyze_rows(rows, detection_rows)
    print_summary(summary)
    if args.output:
        write_report(Path(args.output), summary)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def analyze_rows(
    rows: list[dict[str, str]],
    detection_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
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
    distance_rows = detection_rows if detection_rows is not None else rows
    invalid_distance = [r for r in distance_rows if not _distance_valid(r)]
    valid_distance = [r for r in distance_rows if _distance_valid(r)]
    invalid_distance_reason_counts = Counter()
    for row in invalid_distance:
        for reason in str(row.get("distance_reason_codes", "")).split(","):
            reason = reason.strip()
            if reason:
                invalid_distance_reason_counts[reason] += 1
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
        "total_detection_rows": len(detection_rows or []),
        "invalid_distance_count_by_reason_code": dict(invalid_distance_reason_counts),
        "invalid_distance_count_by_object_class": dict(
            Counter(_object_class(row) for row in invalid_distance)
        ),
        "invalid_distance_count_for_ego_corridor_targets": len(
            [row for row in invalid_distance if _target_in_ego_corridor(row)]
        ),
        "invalid_distance_count_for_side_targets": len(
            [row for row in invalid_distance if not _target_in_ego_corridor(row)]
        ),
        "valid_distance_count_by_object_class": dict(
            Counter(_object_class(row) for row in valid_distance)
        ),
        "average_distance_confidence_valid_targets": _average(
            [_float(row.get("distance_confidence"), float("nan")) for row in valid_distance]
        ),
        "average_distance_confidence_invalid_targets": _average(
            [_float(row.get("distance_confidence"), float("nan")) for row in invalid_distance]
        ),
        "average_distance_confidence_valid_detections": _average(
            [_float(row.get("distance_confidence"), float("nan")) for row in valid_distance]
        ),
        "average_distance_confidence_invalid_detections": _average(
            [_float(row.get("distance_confidence"), float("nan")) for row in invalid_distance]
        ),
        "top_invalid_distance_examples": _top_invalid_examples(invalid_distance),
        "invalid_distance_rate_by_class": _invalid_rate_by(invalid_distance, valid_distance, _object_class),
        "invalid_distance_rate_by_side_state": _invalid_rate_by(
            invalid_distance,
            valid_distance,
            lambda row: row.get("side_state", "unknown") or "unknown",
        ),
        "invalid_distance_rate_by_day_night/glare": _invalid_rate_by_scene_quality(
            invalid_distance,
            valid_distance,
        ),
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


def _average(values: list[float]) -> float | None:
    finite = [value for value in values if isinstance(value, float) and value == value]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _distance_valid(row: dict[str, str]) -> bool:
    if "distance_valid_for_safety" in row:
        return _truthy(row.get("distance_valid_for_safety"))
    return _truthy(row.get("target_distance_valid_for_safety", "true"))


def _object_class(row: dict[str, str]) -> str:
    return row.get("object_class") or row.get("target_class") or "unknown"


def _target_in_ego_corridor(row: dict[str, str]) -> bool:
    if "target_in_ego_corridor" in row:
        return _truthy(row.get("target_in_ego_corridor"))
    return _truthy(row.get("in_ego_corridor"))


def _top_invalid_examples(rows: list[dict[str, str]], limit: int = 10) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for row in rows[:limit]:
        examples.append(
            {
                "frame_index": row.get("frame_index"),
                "track_id": row.get("track_id") or row.get("selected_target_track_id"),
                "object_class": _object_class(row),
                "distance_reason_codes": row.get("distance_reason_codes"),
                "distance_confidence": row.get("distance_confidence"),
                "distance_bumper_m": row.get("distance_bumper_m") or row.get("target_distance_m"),
            }
        )
    return examples


def _invalid_rate_by(
    invalid_rows: list[dict[str, str]],
    valid_rows: list[dict[str, str]],
    key_fn,
) -> dict[str, float]:
    invalid_counts = Counter(key_fn(row) for row in invalid_rows)
    valid_counts = Counter(key_fn(row) for row in valid_rows)
    rates: dict[str, float] = {}
    for key in sorted(set(invalid_counts) | set(valid_counts)):
        total = invalid_counts[key] + valid_counts[key]
        rates[str(key)] = invalid_counts[key] / total if total else 0.0
    return rates


def _invalid_rate_by_scene_quality(
    invalid_rows: list[dict[str, str]],
    valid_rows: list[dict[str, str]],
) -> dict[str, float]:
    def scene_bucket(row: dict[str, str]) -> str:
        if _truthy(row.get("night")) or _truthy(row.get("glare")):
            return "night_or_glare"
        if row.get("night") is None and row.get("glare") is None:
            return "unknown"
        return "day_or_clear"

    return _invalid_rate_by(invalid_rows, valid_rows, scene_bucket)


if __name__ == "__main__":
    main()
