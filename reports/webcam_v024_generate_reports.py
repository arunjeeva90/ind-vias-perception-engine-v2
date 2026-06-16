from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "webcam_v024"
REPORT_DIR = ROOT / "reports"
BASELINE_METRICS = REPORT_DIR / "webcam_v023_metrics.csv"
VIDEOS = [
    ("dmsDay101", Path(r"D:\Workspace\dmswebcam\dmsDay101.mp4"), "day"),
    ("dmsDay102", Path(r"D:\Workspace\dmswebcam\dmsDay102.mp4"), "day"),
    ("dmsNit201", Path(r"D:\Workspace\dmswebcam\dmsNit201.mp4"), "night"),
    ("dmsNit202", Path(r"D:\Workspace\dmswebcam\dmsNit202.mp4"), "night"),
]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get(row: dict, dotted: str, default=None):
    cur = row
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {
        "fps": fps,
        "frames": frames,
        "duration_s": frames / fps if fps else 0.0,
    }


def banner(row: dict) -> str:
    return get(row, "dms_v02.final_banner") or get(row, "driver_availability.state") or "UNKNOWN"


def frame_time(row: dict, fps: float) -> float:
    if row.get("timestamp_ms") is not None:
        return float(row["timestamp_ms"]) / 1000.0
    return int(row.get("frame_id", 0)) / fps if fps else 0.0


def reason_codes(row: dict) -> list[str]:
    codes: list[str] = []
    for path in [
        "driver_availability.reason_codes",
        "dms_v02.reason_codes",
        "dms_v02.classification_reason_codes",
        "attention.attention_reason_codes",
        "phone_use.reason_codes",
        "driver_observability.reason_codes",
        "drowsiness.perclos_validity_reason_codes",
    ]:
        values = get(row, path, [])
        if isinstance(values, list):
            codes.extend(str(value) for value in values)
    return list(dict.fromkeys(codes))


def transition_stats(rows: list[dict], fps: float) -> tuple[int, int]:
    transitions: list[float] = []
    prev = None
    for row in rows:
        current = banner(row)
        if prev is not None and current != prev:
            transitions.append(frame_time(row, fps))
        prev = current
    max_2s = 0
    for start in transitions:
        max_2s = max(max_2s, sum(1 for t in transitions if start <= t <= start + 2.0))
    return len(transitions), max_2s


def intervalize(rows: list[dict], mask, fps: float) -> list[tuple[int, int, float, float]]:
    intervals: list[tuple[int, int, float, float]] = []
    start_idx = None
    for idx, row in enumerate(rows):
        if mask(row):
            if start_idx is None:
                start_idx = idx
        elif start_idx is not None:
            intervals.append((start_idx, idx - 1, frame_time(rows[start_idx], fps), frame_time(rows[idx - 1], fps)))
            start_idx = None
    if start_idx is not None and rows:
        intervals.append((start_idx, len(rows) - 1, frame_time(rows[start_idx], fps), frame_time(rows[-1], fps)))
    return intervals


def driver_landmarks(row: dict) -> int:
    return int(get(row, "driver_identity.driver_landmark_count", 0) or 0)


def face_proposals(row: dict) -> int:
    return int(get(row, "dms_health.face_proposals", 0) or 0)


def is_unavailable(row: dict) -> bool:
    return banner(row) == "DRIVER UNAVAILABLE" or get(row, "driver_availability.state") == "UNAVAILABLE"


def is_degraded_attention(row: dict) -> bool:
    substate = get(row, "attention.attention_substate", "UNKNOWN")
    return get(row, "attention.attention_state", "UNKNOWN") == "DEGRADED" or substate in {
        "FACE_LOST",
        "HEAD_POSE_UNRELIABLE",
    }


def is_proposal_only_driver(row: dict) -> bool:
    state = get(row, "driver_identity.driver_face_state", "")
    return bool(get(row, "driver_identity.driver_proposal_visible", False)) or state in {
        "PROPOSAL_ONLY",
        "LANDMARK_FAILED",
    }


def add_anomalies(anomalies: list[dict], rows: list[dict], debug_rows: list[dict], video: str, fps: float, mode: str) -> None:
    debug_by_frame = {row.get("frame_id"): row for row in debug_rows}
    specs = [
        (
            "DRIVER_UNAVAILABLE_WHILE_PROPOSAL_EXISTS",
            lambda r: is_unavailable(r) and face_proposals(r) > 0,
            "DMS MONITOR or DMS DEGRADED",
            "Proposal exists while final state is unavailable.",
            "model issue",
            "Keep proposal-visible driver as degraded/monitor until proposal/body/recent-track evidence is gone.",
        ),
        (
            "NORMAL_WHILE_ATTENTION_DEGRADED_OR_FACE_LOST",
            lambda r: banner(r) == "NORMAL" and is_degraded_attention(r),
            "DMS MONITOR or DMS DEGRADED",
            "Final NORMAL contradicts degraded/facelost internal attention.",
            "model issue",
            "Block NORMAL unless a held-road/proposal-visible state explicitly allows it.",
        ),
        (
            "DMS_DEGRADED_VISIBLE_NEAR_ROAD",
            lambda r: banner(r) == "DMS DEGRADED" and abs(float(get(r, "gaze.relative_yaw_deg", 999) or 999)) <= 30,
            "NORMAL_HELD, DMS MONITOR, or DMS DEGRADED only after hold timeout",
            "Near-road yaw but degraded final state.",
            "threshold issue",
            "Use short webcam proposal-visible/track-hold before degraded.",
        ),
        (
            "FALSE_PASSENGER_OR_OBJECT_CANDIDATE",
            lambda r: int(get(r, "occupants.face_count", 0) or 0) > 1
            or int(get(r, "occupants.unconfirmed_proposal_count", 0) or 0) > 0,
            "Raw proposal hidden unless debug overlay is enabled",
            "Extra face/occupant candidate present.",
            "overlay issue",
            "Hide unvalidated raw proposals and require landmark/temporal confirmation for occupants.",
        ),
        (
            "FACE_DETECTED_LANDMARKS_UNAVAILABLE",
            lambda r: face_proposals(r) > 0 and driver_landmarks(r) == 0,
            "PROPOSAL_VISIBLE / LANDMARK_FAILED, not driver absence",
            "Face proposal exists but driver landmarks are unavailable.",
            "model issue",
            "Treat proposal-to-landmark failure as observability degraded rather than absence.",
        ),
        (
            "FACE_BOX_OR_DETECTION_BUT_DRIVER_NOT_VISIBLE",
            lambda r: face_proposals(r) > 0 and get(r, "driver_presence.state") in {"NOT_VISIBLE", "ABSENT", "LOST_LONG"},
            "PROPOSAL_VISIBLE or PRESENT",
            "Face detector sees a candidate while driver presence says not visible.",
            "model issue",
            "Emit proposal-visible driver state and suppress unavailable during proposal hold.",
        ),
        (
            "DROWSINESS_WARNING_PHONE_OR_HEAD_DOWN_LIKELY",
            lambda r: banner(r) == "DROWSINESS WARNING"
            and bool({"HEAD_DOWN", "POSSIBLE_PHONE_POSTURE", "GAZE_OFF_ROAD"} & set(reason_codes(r))),
            "DISTRACTION WARNING or DMS MONITOR unless valid eye evidence exists",
            "Drowsiness warning includes head-down/phone-like evidence.",
            "model issue",
            "Keep drowsiness gated on valid PERCLOS/eye-closure/microsleep evidence.",
        ),
    ]
    if mode == "night":
        specs.append(
            (
                "NIGHT_VIDEO_NO_FACE_FAILURE",
                lambda r: face_proposals(r) == 0 and get(r, "driver_presence.state") in {"ABSENT", "NOT_VISIBLE", "LOST_LONG"},
                "DMS DEGRADED with low-light diagnostics if the driver is visually present",
                "No proposal in night/low-light frame.",
                "camera/lighting issue",
                "Review webcam night exposure/noise and face proposal thresholds.",
            )
        )
    if mode == "day":
        specs.append(
            (
                "DAY_BACKLIGHT_OR_EXPOSURE_FAILURE",
                lambda r: face_proposals(r) == 0 and get(r, "driver_presence.state") in {"ABSENT", "NOT_VISIBLE", "LOST_LONG"},
                "DMS DEGRADED with backlight diagnostics if the driver is visually present",
                "No proposal in daylight/backlight frame.",
                "camera/lighting issue",
                "Review exposure/backlight diagnostics and proposal thresholds.",
            )
        )

    for anomaly_type, mask, expected, root, category, fix in specs:
        for start, end, start_s, end_s in intervalize(rows, mask, fps):
            sample = rows[start]
            dbg = debug_by_frame.get(sample.get("frame_id"), {})
            anomalies.append(
                {
                    "video_name": video,
                    "anomaly_type": anomaly_type,
                    "start_time_s": round(start_s, 3),
                    "end_time_s": round(end_s, 3),
                    "frame_start": int(sample.get("frame_id", start)),
                    "frame_end": int(rows[end].get("frame_id", end)),
                    "observed_banner": banner(sample),
                    "expected_banner": expected,
                    "available_reason_codes": reason_codes(sample),
                    "debug_flags": dbg.get("contradiction_flags", []),
                    "likely_root_cause": root,
                    "suggested_next_fix": fix,
                    "issue_category": category,
                }
            )

    transition_count, max_2s = transition_stats(rows, fps)
    if max_2s > 3 and rows:
        anomalies.append(
            {
                "video_name": video,
                "anomaly_type": "BANNER_FLICKER_GT_3_TRANSITIONS_2S",
                "start_time_s": 0.0,
                "end_time_s": round(frame_time(rows[-1], fps), 3),
                "frame_start": int(rows[0].get("frame_id", 0)),
                "frame_end": int(rows[-1].get("frame_id", len(rows) - 1)),
                "observed_banner": f"{transition_count} transitions, max_2s={max_2s}",
                "expected_banner": "No more than 3 transitions in any 2-second window",
                "available_reason_codes": [],
                "debug_flags": [],
                "likely_root_cause": "Banner state still transitions too quickly for this clip.",
                "suggested_next_fix": "Increase banner hold/recovery smoothing only if this is visually noisy.",
                "issue_category": "threshold issue",
            }
        )


def summarize_video(name: str, path: Path, mode: str) -> tuple[dict, list[dict]]:
    rows = read_jsonl(OUT_DIR / f"{name}_state_v024_stable.jsonl")
    debug_rows = read_jsonl(OUT_DIR / f"{name}_debug_trace_v024.jsonl")
    meta = video_meta(path)
    fps = meta["fps"] or 30.0
    banners = Counter(banner(row) for row in rows)
    transition_count, max_2s = transition_stats(rows, fps)
    anomalies: list[dict] = []
    add_anomalies(anomalies, rows, debug_rows, name, fps, mode)
    summary = {
        "video_name": name,
        "mode": mode,
        "duration_s": round(meta["duration_s"], 3),
        "fps": round(fps, 3),
        "video_total_frames": meta["frames"],
        "processed_frames": len(rows),
        "proposal_count": sum(face_proposals(row) for row in rows),
        "proposal_frames": sum(1 for row in rows if face_proposals(row) > 0),
        "validated_driver_count": sum(1 for row in rows if driver_landmarks(row) > 0 or get(row, "driver_presence.state") == "PRESENT"),
        "proposal_only_driver_frames": sum(1 for row in rows if is_proposal_only_driver(row)),
        "landmark_failed_driver_frames": sum(1 for row in rows if face_proposals(row) > 0 and driver_landmarks(row) == 0),
        "driver_unavailable_frames": sum(1 for row in rows if is_unavailable(row)),
        "driver_unavailable_with_proposal_count": sum(1 for row in rows if is_unavailable(row) and face_proposals(row) > 0),
        "normal_while_attention_degraded_count": sum(1 for row in rows if banner(row) == "NORMAL" and is_degraded_attention(row)),
        "degraded_visible_near_road_count": sum(
            1
            for row in rows
            if banner(row) == "DMS DEGRADED" and abs(float(get(row, "gaze.relative_yaw_deg", 999) or 999)) <= 30
        ),
        "false_occupant_candidate_count": sum(
            1
            for row in rows
            if int(get(row, "occupants.face_count", 0) or 0) > 1
            or int(get(row, "occupants.unconfirmed_proposal_count", 0) or 0) > 0
        ),
        "dms_degraded_frames": banners.get("DMS DEGRADED", 0),
        "normal_frames": banners.get("NORMAL", 0),
        "monitor_frames": banners.get("DMS MONITOR", 0),
        "distraction_warning_frames": banners.get("DISTRACTION WARNING", 0),
        "drowsiness_warning_frames": banners.get("DROWSINESS WARNING", 0),
        "danger_frames": banners.get("DANGER", 0),
        "phone_suspected_or_confirmed_frames": sum(
            1
            for row in rows
            if str(get(row, "phone_use.driver_state", "UNKNOWN")) not in {"NO_PHONE", "UNKNOWN", "PHONE_UNKNOWN"}
        ),
        "banner_transition_count": transition_count,
        "max_banner_transitions_2s": max_2s,
        "calibration_sources": ",".join(
            sorted({str(get(row, "gaze.road_axis_calibration_source", get(row, "gaze.calibration_source", "UNKNOWN"))) for row in rows})
        ),
        "nir_modes": ",".join(sorted({str(get(row, "dms_health.nir_mode", "UNKNOWN")) for row in rows})),
        "avg_face_detection_confidence": round(
            sum(float(get(row, "dms_health.face_detection_confidence", 0.0) or 0.0) for row in rows) / max(1, len(rows)),
            3,
        ),
        "anomaly_count": len(anomalies),
    }
    return summary, anomalies


def read_baseline() -> dict[str, dict]:
    if not BASELINE_METRICS.exists():
        return {}
    with BASELINE_METRICS.open("r", encoding="utf-8", newline="") as handle:
        return {row["video_name"]: row for row in csv.DictReader(handle)}


def write_reports() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    anomalies: list[dict] = []
    for name, path, mode in VIDEOS:
        summary, video_anomalies = summarize_video(name, path, mode)
        summaries.append(summary)
        anomalies.extend(video_anomalies)

    metrics_path = REPORT_DIR / "webcam_v024_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    anomalies_path = REPORT_DIR / "webcam_v024_anomalies.jsonl"
    with anomalies_path.open("w", encoding="utf-8") as handle:
        for item in anomalies:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    baseline = read_baseline()
    md = [
        "# IND-VIAS DualSight DMS v0.2.4 Webcam Stability Evaluation",
        "",
        "## Scope",
        "Targeted webcam-specific v0.2.4 patch evaluation on the Zebronics windshield/RVM-mounted webcam clips. This report compares v0.2.4 outputs against the v0.2.3 baseline metrics where available.",
        "",
        "## v0.2.3 vs v0.2.4 Summary",
        "",
        "| Video | v0.2.3 unavailable | v0.2.4 unavailable | v0.2.3 degraded | v0.2.4 degraded | v0.2.3 transitions | v0.2.4 transitions | unavailable+proposal | normal+degraded | false occupant candidates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        base = baseline.get(row["video_name"], {})
        md.append(
            "| {video} | {b_unavail} | {unavail} | {b_degraded} | {degraded} | {b_trans} | {trans} | {uwp} | {nwd} | {false_occ} |".format(
                video=row["video_name"],
                b_unavail=base.get("driver_unavailable_frames", "n/a"),
                unavail=row["driver_unavailable_frames"],
                b_degraded=base.get("dms_degraded_frames", "n/a"),
                degraded=row["dms_degraded_frames"],
                b_trans=base.get("banner_transition_count", "n/a"),
                trans=row["banner_transition_count"],
                uwp=row["driver_unavailable_with_proposal_count"],
                nwd=row["normal_while_attention_degraded_count"],
                false_occ=row["false_occupant_candidate_count"],
            )
        )

    md.extend(["", "## Per-Video Metrics", ""])
    for row in summaries:
        video_anoms = [item for item in anomalies if item["video_name"] == row["video_name"]]
        top_types = Counter(item["anomaly_type"] for item in video_anoms).most_common(6)
        md.extend(
            [
                f"### {row['video_name']}",
                f"- Duration/FPS/frames: {row['duration_s']} s, {row['fps']} fps, {row['processed_frames']} processed frames.",
                f"- Proposals: total={row['proposal_count']}, frames={row['proposal_frames']}; validated driver={row['validated_driver_count']}; proposal-only driver={row['proposal_only_driver_frames']}; landmark-failed driver={row['landmark_failed_driver_frames']}.",
                f"- Banners: NORMAL={row['normal_frames']}, MONITOR={row['monitor_frames']}, DEGRADED={row['dms_degraded_frames']}, DISTRACTION={row['distraction_warning_frames']}, DROWSINESS={row['drowsiness_warning_frames']}, DANGER={row['danger_frames']}, UNAVAILABLE={row['driver_unavailable_frames']}.",
                f"- Webcam patch counters: unavailable_with_proposal={row['driver_unavailable_with_proposal_count']}, normal_while_degraded={row['normal_while_attention_degraded_count']}, degraded_visible_near_road={row['degraded_visible_near_road_count']}, false_occupant_candidates={row['false_occupant_candidate_count']}.",
                f"- Stability: banner transitions={row['banner_transition_count']}, max transitions in 2s={row['max_banner_transitions_2s']}.",
                f"- Calibration/NIR: sources={row['calibration_sources']}, modes={row['nir_modes']}, avg face conf={row['avg_face_detection_confidence']}.",
                "- Top anomaly types: " + (", ".join(f"{name}={count}" for name, count in top_types) if top_types else "none"),
                "",
            ]
        )

    md.extend(["## Top 10 Anomalies", ""])
    for item in sorted(anomalies, key=lambda a: (a["video_name"], a["start_time_s"], a["anomaly_type"]))[:10]:
        md.append(
            f"- {item['video_name']} {item['start_time_s']:.2f}-{item['end_time_s']:.2f}s "
            f"{item['anomaly_type']}: observed `{item['observed_banner']}`, expected `{item['expected_banner']}`. "
            f"Root cause: {item['likely_root_cause']}"
        )

    md.extend(
        [
            "",
            "## Recommendation",
            "Keep this as a small webcam-specific v0.2.4 stabilization branch. The patch reduces the highest-risk semantic issue by treating proposal-visible landmark failures as degraded/monitor evidence instead of immediate absence. The remaining anomalies are best handled with more robust proposal-to-landmark retry diagnostics and camera profile tuning, not ADAS fusion.",
            "",
            "## Generated Artifacts",
            f"- Metrics CSV: `{metrics_path}`",
            f"- Anomalies JSONL: `{anomalies_path}`",
            "- Per-video outputs/logs: `outputs/webcam_v024/`",
        ]
    )
    summary_path = REPORT_DIR / "webcam_v024_summary.md"
    summary_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(metrics_path)
    print(anomalies_path)
    print(summary_path)


if __name__ == "__main__":
    write_reports()
