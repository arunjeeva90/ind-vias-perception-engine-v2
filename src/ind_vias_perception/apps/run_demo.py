from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ind_vias_perception.common.types import Detection, FramePacket, PerceptionOutput
from ind_vias_perception.config.settings import load_settings
from ind_vias_perception.pipeline.factory import build_pipeline
from ind_vias_perception.apps.visualization import draw_perception_output


DEBUG_CSV_COLUMNS = [
    "frame_index",
    "timestamp_s",
    "selected_target_track_id",
    "target_class",
    "selected_target_valid_for_safety",
    "selected_target_reason",
    "debug_target_track_id",
    "debug_target_distance_valid_for_safety",
    "target_distance_m",
    "target_ttc_s",
    "target_in_ego_corridor",
    "target_relevance",
    "distance_confidence",
    "distance_reason_codes",
    "distance_ground_m",
    "distance_semantic_m",
    "distance_fused_camera_m",
    "distance_bumper_m",
    "distance_source",
    "ground_semantic_ratio",
    "bbox_area_ratio",
    "ground_contact_row",
    "horizon_y",
    "near_horizon_margin_px",
    "is_side_object",
    "is_near_boundary",
    "is_tiny_bbox",
    "raw_warning_level",
    "confirmed_warning_level",
    "warning_candidate",
    "warning_suppressed_reason",
    "ego_motion_state",
    "yaw_confidence",
    "ego_motion_reason_codes",
    "ego_motion_feature_count",
    "ego_motion_roi_shape",
    "ego_motion_downscale_factor",
    "cais_mode",
    "cais_score",
    "cais_reason_codes",
    "cais_ttc_used_s",
    "cais_ttc_threshold_s",
    "cais_ttc_source_track_id",
    "ttc_valid_for_safety",
    "ttc_reason_codes",
    "side_state",
    "cutin_state",
    "ttc_lateral_s",
    "cutin_confidence",
    "cutin_valid_for_safety",
    "cutin_reason_codes",
    "lateral_velocity_px_s",
    "lateral_history_count",
    "corridor_overlap_ratio",
    "corridor_overlap_delta",
    "corridor_entry_confirmed",
    "lateral_motion_stable",
    "lateral_center_history_count",
    "lateral_velocity_px_s_smoothed",
    "cutin_crossing_trend",
    "cutin_entry_side",
    "cutin_warning_eligible",
    "cutin_warning_candidate",
    "cutin_warning_confirmed",
    "cutin_target_track_id",
    "crossing_state",
    "crossing_confidence",
    "crossing_history_count",
    "crossing_valid_for_safety",
    "crossing_reason_codes",
    "crossing_lateral_displacement_px",
    "crossing_corridor_approach",
    "crossing_boundary_suppressed",
    "crossing_tiny_object_suppressed",
    "sentinel_state",
]

DETECTION_DEBUG_CSV_COLUMNS = [
    "frame_index",
    "timestamp_s",
    "track_id",
    "object_class",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_clipped",
    "distance_ground_m",
    "distance_semantic_m",
    "distance_fused_camera_m",
    "distance_bumper_m",
    "distance_confidence",
    "distance_valid_for_safety",
    "distance_reason_codes",
    "ground_semantic_ratio",
    "bbox_area_ratio",
    "ground_contact_row",
    "horizon_y",
    "near_horizon_margin_px",
    "is_side_object",
    "is_near_boundary",
    "is_tiny_bbox",
    "target_relevance",
    "target_in_ego_corridor",
    "side_state",
    "cutin_state",
    "cutin_valid_for_safety",
    "cutin_reason_codes",
    "crossing_state",
    "crossing_valid_for_safety",
    "crossing_reason_codes",
    "ttc_s",
    "ttc_valid_for_safety",
    "ttc_reason_codes",
    "predicted_track",
    "missing_frames",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--debug-csv", default=None)
    parser.add_argument("--debug-detections-csv", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    settings = load_settings(args.config)
    detection_backend = settings.raw.get("detection", {}).get("backend", "dummy")
    ego_corridor = settings.raw.get("ego_corridor", {})
    validate_detector_config(settings.raw)
    pipeline = build_pipeline(settings)
    if hasattr(pipeline.detection_head, "debug"):
        pipeline.detection_head.debug = args.debug_overlay

    if args.image is not None:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read image: {args.image}")
        out = pipeline.process(FramePacket(frame=frame, timestamp_s=0.0, frame_id=0))
        annotated = draw_perception_output(
            frame,
            out,
            detection_backend=detection_backend,
            debug_overlay=args.debug_overlay,
            ego_corridor=ego_corridor,
        )
        print(out.safety_payload)
        if args.debug_detections_csv is not None:
            det_csv_file, det_writer = _open_debug_csv(
                args.debug_detections_csv,
                DETECTION_DEBUG_CSV_COLUMNS,
            )
            try:
                write_debug_detections_csv_rows(det_writer, 0, 0.0, out)
            finally:
                det_csv_file.close()
        if args.output is not None:
            if not cv2.imwrite(args.output, annotated):
                raise SystemExit(f"Could not write image: {args.output}")
        if args.show:
            cv2.imshow("IND-VIAS Perception", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    if args.video is None:
        import numpy as np
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(args.max_frames or 3):
            out = pipeline.process(FramePacket(frame=frame, timestamp_s=i / 30.0, frame_id=i))
            print(out.safety_payload)
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not read video: {args.video}")
    pipeline.ego_yaw_enabled = bool(
        settings.raw.get("ego_motion", {}).get("enable_yaw_detection", False)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_run_summary(fps, total_frames, args.max_frames))
    writer = None
    debug_csv_file = None
    debug_csv_writer = None
    debug_detections_csv_file = None
    debug_detections_csv_writer = None
    if args.debug_csv is not None:
        debug_csv_file, debug_csv_writer = _open_debug_csv(args.debug_csv, DEBUG_CSV_COLUMNS)
    if args.debug_detections_csv is not None:
        debug_detections_csv_file, debug_detections_csv_writer = _open_debug_csv(
            args.debug_detections_csv,
            DETECTION_DEBUG_CSV_COLUMNS,
        )
    i = 0
    try:
        while should_process_frame(i, args.max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_s = i / fps
            out = pipeline.process(FramePacket(frame=frame, timestamp_s=timestamp_s, frame_id=i))
            if debug_csv_writer is not None:
                write_debug_csv_row(debug_csv_writer, i, timestamp_s, out)
            if debug_detections_csv_writer is not None:
                write_debug_detections_csv_rows(
                    debug_detections_csv_writer,
                    i,
                    timestamp_s,
                    out,
                )
            annotated = draw_perception_output(
                frame,
                out,
                detection_backend=detection_backend,
                debug_overlay=args.debug_overlay,
                ego_corridor=ego_corridor,
            )
            print(out.safety_payload)
            if args.output is not None:
                if writer is None:
                    height, width = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
                    if not writer.isOpened():
                        raise SystemExit(f"Could not write video: {args.output}")
                writer.write(annotated)
            if args.show:
                cv2.imshow("IND-VIAS Perception", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            i += 1
    finally:
        if debug_csv_file is not None:
            debug_csv_file.close()
        if debug_detections_csv_file is not None:
            debug_detections_csv_file.close()
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


def write_debug_csv_row(
    writer: csv.DictWriter,
    frame_index: int,
    timestamp_s: float,
    output: PerceptionOutput,
) -> None:
    payload = output.safety_payload
    writer.writerow(
        {
            "frame_index": frame_index,
            "timestamp_s": timestamp_s,
            "selected_target_track_id": payload.get("target_track_id"),
            "target_class": payload.get("target_class"),
            "selected_target_valid_for_safety": payload.get("selected_target_valid_for_safety"),
            "selected_target_reason": payload.get("selected_target_reason"),
            "debug_target_track_id": payload.get("debug_target_track_id"),
            "debug_target_distance_valid_for_safety": payload.get(
                "debug_target_distance_valid_for_safety"
            ),
            "target_distance_m": payload.get("target_distance_m"),
            "target_ttc_s": payload.get("target_ttc_s"),
            "target_in_ego_corridor": payload.get("target_in_ego_corridor"),
            "target_relevance": payload.get("target_relevance"),
            "distance_confidence": payload.get("distance_confidence"),
            "distance_reason_codes": payload.get("distance_reason_codes"),
            "distance_ground_m": payload.get("distance_ground_m"),
            "distance_semantic_m": payload.get("distance_semantic_m"),
            "distance_fused_camera_m": payload.get("distance_fused_camera_m"),
            "distance_bumper_m": payload.get("distance_bumper_m"),
            "distance_source": payload.get("distance_source"),
            "ground_semantic_ratio": payload.get("ground_semantic_ratio"),
            "bbox_area_ratio": payload.get("bbox_area_ratio"),
            "ground_contact_row": payload.get("ground_contact_row"),
            "horizon_y": payload.get("horizon_y"),
            "near_horizon_margin_px": payload.get("near_horizon_margin_px"),
            "is_side_object": payload.get("is_side_object"),
            "is_near_boundary": payload.get("is_near_boundary"),
            "is_tiny_bbox": payload.get("is_tiny_bbox"),
            "raw_warning_level": payload.get("raw_warning_level"),
            "confirmed_warning_level": payload.get("confirmed_warning_level"),
            "warning_candidate": payload.get("warning_candidate"),
            "warning_suppressed_reason": payload.get("warning_suppressed_reason"),
            "ego_motion_state": payload.get("ego_motion_state"),
            "yaw_confidence": payload.get("yaw_confidence"),
            "ego_motion_reason_codes": payload.get("ego_motion_reason_codes"),
            "ego_motion_feature_count": payload.get("ego_motion_feature_count"),
            "ego_motion_roi_shape": payload.get("ego_motion_roi_shape"),
            "ego_motion_downscale_factor": payload.get("ego_motion_downscale_factor"),
            "cais_mode": payload.get("cais_mode"),
            "cais_score": payload.get("cais_score"),
            "cais_reason_codes": payload.get("cais_reason_codes"),
            "cais_ttc_used_s": payload.get("cais_ttc_used_s"),
            "cais_ttc_threshold_s": payload.get("cais_ttc_threshold_s"),
            "cais_ttc_source_track_id": payload.get("cais_ttc_source_track_id"),
            "ttc_valid_for_safety": payload.get("ttc_valid_for_safety"),
            "ttc_reason_codes": payload.get("ttc_reason_codes"),
            "side_state": payload.get("side_state"),
            "cutin_state": payload.get("cutin_state"),
            "ttc_lateral_s": payload.get("ttc_lateral_s"),
            "cutin_confidence": payload.get("cutin_confidence"),
            "cutin_valid_for_safety": payload.get("cutin_valid_for_safety"),
            "cutin_reason_codes": payload.get("cutin_reason_codes"),
            "lateral_velocity_px_s": payload.get("lateral_velocity_px_s"),
            "lateral_history_count": payload.get("lateral_history_count"),
            "corridor_overlap_ratio": payload.get("corridor_overlap_ratio"),
            "corridor_overlap_delta": payload.get("corridor_overlap_delta"),
            "corridor_entry_confirmed": payload.get("corridor_entry_confirmed"),
            "lateral_motion_stable": payload.get("lateral_motion_stable"),
            "lateral_center_history_count": payload.get("lateral_center_history_count"),
            "lateral_velocity_px_s_smoothed": payload.get("lateral_velocity_px_s_smoothed"),
            "cutin_crossing_trend": payload.get("cutin_crossing_trend"),
            "cutin_entry_side": payload.get("cutin_entry_side"),
            "cutin_warning_eligible": payload.get("cutin_warning_eligible"),
            "cutin_warning_candidate": payload.get("cutin_warning_candidate"),
            "cutin_warning_confirmed": payload.get("cutin_warning_confirmed"),
            "cutin_target_track_id": payload.get("cutin_target_track_id"),
            "crossing_state": payload.get("crossing_state"),
            "crossing_confidence": payload.get("crossing_confidence"),
            "crossing_history_count": payload.get("crossing_history_count"),
            "crossing_valid_for_safety": payload.get("crossing_valid_for_safety"),
            "crossing_reason_codes": payload.get("crossing_reason_codes"),
            "crossing_lateral_displacement_px": payload.get("crossing_lateral_displacement_px"),
            "crossing_corridor_approach": payload.get("crossing_corridor_approach"),
            "crossing_boundary_suppressed": payload.get("crossing_boundary_suppressed"),
            "crossing_tiny_object_suppressed": payload.get("crossing_tiny_object_suppressed"),
            "sentinel_state": payload.get("sentinel_state"),
        }
    )


def write_debug_detections_csv_rows(
    writer: csv.DictWriter,
    frame_index: int,
    timestamp_s: float,
    output: PerceptionOutput,
) -> None:
    for det in output.detections:
        writer.writerow(_detection_debug_row(frame_index, timestamp_s, det))


def _detection_debug_row(frame_index: int, timestamp_s: float, det: Detection) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "timestamp_s": timestamp_s,
        "track_id": det.track_id,
        "object_class": det.label.value,
        "confidence": det.confidence,
        "bbox_x1": det.bbox.x1,
        "bbox_y1": det.bbox.y1,
        "bbox_x2": det.bbox.x2,
        "bbox_y2": det.bbox.y2,
        "bbox_clipped": det.metadata.get("bbox_clipped"),
        "distance_ground_m": det.metadata.get("distance_ground_m"),
        "distance_semantic_m": det.metadata.get("distance_semantic_m"),
        "distance_fused_camera_m": det.metadata.get("distance_fused_camera_m"),
        "distance_bumper_m": det.metadata.get("distance_bumper_m"),
        "distance_confidence": det.metadata.get("distance_confidence"),
        "distance_valid_for_safety": det.metadata.get("distance_valid_for_safety"),
        "distance_reason_codes": det.metadata.get(
            "distance_reason_codes",
            det.metadata.get("reason_codes"),
        ),
        "ground_semantic_ratio": det.metadata.get("ground_semantic_ratio"),
        "bbox_area_ratio": det.metadata.get("bbox_area_ratio"),
        "ground_contact_row": det.metadata.get("ground_contact_row"),
        "horizon_y": det.metadata.get("horizon_y"),
        "near_horizon_margin_px": det.metadata.get("near_horizon_margin_px"),
        "is_side_object": det.metadata.get("is_side_object"),
        "is_near_boundary": det.metadata.get("is_near_boundary"),
        "is_tiny_bbox": det.metadata.get("is_tiny_bbox"),
        "target_relevance": det.metadata.get("target_relevance"),
        "target_in_ego_corridor": det.metadata.get("in_ego_corridor"),
        "side_state": det.metadata.get("side_state"),
        "cutin_state": det.metadata.get("cutin_state"),
        "cutin_valid_for_safety": det.metadata.get("cutin_valid_for_safety"),
        "cutin_reason_codes": det.metadata.get("cutin_reason_codes"),
        "crossing_state": det.metadata.get("crossing_state"),
        "crossing_valid_for_safety": det.metadata.get("crossing_valid_for_safety"),
        "crossing_reason_codes": det.metadata.get("crossing_reason_codes"),
        "ttc_s": det.ttc_s,
        "ttc_valid_for_safety": det.metadata.get("ttc_valid_for_safety"),
        "ttc_reason_codes": det.metadata.get("ttc_reason_codes"),
        "predicted_track": det.metadata.get("track_predicted"),
        "missing_frames": det.metadata.get("missing_frames"),
    }


def _open_debug_csv(path: str, columns: list[str]):
    csv_path = Path(path)
    if csv_path.parent != Path("."):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    writer.writeheader()
    return csv_file, writer


def should_process_frame(frame_index: int, max_frames: int | None) -> bool:
    return max_frames is None or frame_index < max_frames


def video_run_summary(fps: float, total_frames: int, max_frames: int | None) -> str:
    total = str(total_frames) if total_frames > 0 else "unknown"
    mode = "full video" if max_frames is None else "limited debug run"
    max_value = "none" if max_frames is None else str(max_frames)
    return (
        f"Video input: fps={fps:.2f}, total_frames={total}, "
        f"max_frames={max_value}, mode={mode}"
    )


def validate_detector_config(raw_settings: dict[str, object]) -> None:
    detection_cfg = raw_settings.get("detection", {})
    if not isinstance(detection_cfg, dict):
        return
    if detection_cfg.get("backend", "dummy") != "onnx":
        return

    model_path = Path(str(detection_cfg.get("onnx_model_path", "models/weights/detector.onnx")))
    if model_path.exists():
        return

    raise SystemExit(
        "ONNX detector model is missing. Place the ONNX model at "
        "models/weights/detector.onnx or switch back to configs/default.yaml."
    )


if __name__ == "__main__":
    main()
