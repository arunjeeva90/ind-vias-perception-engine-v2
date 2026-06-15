from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2  # noqa: E402

from ind_vias_dms.core.config import load_dms_config  # noqa: E402
from ind_vias_dms.core.pipeline import DMSPipeline  # noqa: E402
from ind_vias_dms.core.road_calibration import load_road_calibration, save_road_calibration  # noqa: E402
from ind_vias_dms.interface.dms_packet import serialize_dms_state  # noqa: E402
from ind_vias_dms.utils.debug_trace import DebugTraceRecorder  # noqa: E402
from ind_vias_dms.utils.jsonl_writer import JSONLWriter  # noqa: E402
from ind_vias_dms.utils.learning_memory import LearningMemoryWriter  # noqa: E402
from ind_vias_dms.utils.video_io import make_video_writer, open_video_source, resize_to_width  # noqa: E402
from ind_vias_dms.visualization.overlay import OverlayRenderer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone IND-VIAS DualSight DMS v0.1.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", default=None)
    source.add_argument("--camera", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--config", default="configs/dms/dualsight_dms_v0_1.yaml")
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--status-window", action="store_true")
    parser.add_argument("--show-track-id", action="store_true")
    parser.add_argument("--show-debug-proposal-boxes", action="store_true")
    parser.add_argument("--debug-trace", default=None)
    parser.add_argument("--event-log", default=None)
    parser.add_argument("--event-json", default=None)
    parser.add_argument("--review-bundle", default=None)
    parser.add_argument("--save-event-keyframes", action="store_true")
    parser.add_argument("--save-event-crops", action="store_true")
    parser.add_argument("--learning-memory", default=None)
    parser.add_argument("--save-learning-keyframes", action="store_true")
    parser.add_argument("--save-learning-crops", action="store_true")
    parser.add_argument("--keyframe-before-ms", type=int, default=500)
    parser.add_argument("--keyframe-after-ms", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_dms_config(args.config)
    cap = open_video_source(args.video, args.camera)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = config.output_fps

    try:
        pipeline = DMSPipeline(config)
    except RuntimeError as exc:
        cap.release()
        raise SystemExit(str(exc)) from exc
    road_calibration_source = "DEFAULT"
    if config.auto_load_road_calibration:
        calibration = load_road_calibration(config.road_calibration_file)
        if calibration.calibrated:
            pipeline.calibrate_road_gaze(
                calibration.road_axis_yaw_ref_deg,
                calibration.road_axis_pitch_ref_deg,
                calibration.road_axis_roll_ref_deg,
                source=calibration.source,
                confidence=calibration.road_axis_confidence,
            )
            road_calibration_source = calibration.source
            pipeline.road_calibration_source = calibration.source
    overlay = OverlayRenderer(
        banner_min_hold_ms=config.min_banner_hold_ms,
        normal_min_hold_ms=config.normal_min_hold_ms,
        state_clear_confirm_ms=config.state_clear_confirm_ms,
    )
    jsonl = JSONLWriter(args.jsonl)
    debug_trace = DebugTraceRecorder(
        trace_path=args.debug_trace,
        event_log_path=args.event_log,
        event_json_path=args.event_json,
        review_bundle_dir=args.review_bundle,
        save_event_keyframes=args.save_event_keyframes,
        save_event_crops=args.save_event_crops,
        keyframe_before_ms=args.keyframe_before_ms,
        keyframe_after_ms=args.keyframe_after_ms,
    )
    learning_memory = LearningMemoryWriter(
        args.learning_memory,
        config,
        save_keyframes=args.save_learning_keyframes,
        save_crops=args.save_learning_crops,
    )
    writer = None
    frame_id = 0

    try:
        while args.max_frames is None or frame_id < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_to_width(frame, config.frame_resize_width)
            timestamp_ms = int((frame_id / fps) * 1000)
            state, context = pipeline.process(frame, timestamp_ms, frame_id)
            state.gaze.calibration_source = road_calibration_source
            jsonl.write(serialize_dms_state(state))
            debug_trace.write_frame(state, context, frame)
            learning_memory.write_frame(state, context, frame)

            annotated = frame
            if args.debug_overlay or config.overlay_enabled:
                use_status_window = args.status_window or config.status_window_enabled
                draw_panel = (
                    config.telemetry_panel_enabled
                    and config.overlay_panel_embedded
                    and not use_status_window
                )
                annotated = overlay.draw_video_overlay(
                    frame,
                    state,
                    context["face"],
                    context["head_pose"],
                    float(context["fps"]),
                    draw_panel=draw_panel,
                    max_axis_length_px=config.max_axis_length_px,
                    max_gaze_vector_length_px=config.max_gaze_vector_length_px,
                    draw_pose_axes=config.draw_pose_axes,
                    draw_gaze_vector=config.draw_gaze_vector,
                    faces=context["faces"],
                    draw_all_faces=config.draw_all_faces,
                    show_track_id=args.show_track_id,
                    face_proposals=context["face_proposals"],
                    driver_proposal_candidate=context.get("driver_proposal_candidate"),
                    driver_roi_norm=context["driver_roi_norm"],
                    show_debug_proposal_boxes=(
                        args.show_debug_proposal_boxes
                        or config.show_debug_proposal_boxes
                        or config.show_raw_face_proposals
                    ),
                )
            if args.output is not None:
                if writer is None:
                    writer = make_video_writer(args.output, fps, annotated)
                writer.write(annotated)
            if args.display:
                if args.status_window or config.status_window_enabled:
                    cv2.imshow("IND-VIAS DualSight DMS - Video", annotated)
                    status = overlay.render_status_dashboard(
                        state,
                        float(context["fps"]),
                        road_yaw_offset_deg=pipeline.gaze_estimator.yaw_offset_deg,
                        road_pitch_offset_deg=pipeline.gaze_estimator.pitch_offset_deg,
                        road_calibrated=pipeline.gaze_estimator.road_gaze_calibrated,
                        vehicle_layout=config.vehicle_layout,
                        driver_image_side=config.driver_image_side,
                        camera_mount_position=config.camera_mount_position,
                        camera_view_direction=config.camera_view_direction,
                        driver_roi_state=pipeline.occupants.driver_roi_state(),
                    )
                    cv2.imshow("IND-VIAS DualSight DMS - Status", status)
                else:
                    cv2.imshow("IND-VIAS DualSight DMS", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    yaw, pitch = pipeline.calibrate_road_gaze(
                        state.gaze.head_yaw_deg,
                        state.gaze.head_pitch_deg,
                        state.gaze.head_roll_deg,
                        timestamp_ms,
                        source="RUNTIME",
                        confidence=state.gaze.confidence,
                    )
                    if pipeline.last_road_calibration_status == "REJECTED":
                        print(pipeline.last_road_calibration_reason)
                        continue
                    road_calibration_source = "RUNTIME"
                    if config.auto_save_road_calibration:
                        save_road_calibration(
                            config.road_calibration_file,
                            yaw,
                            pitch,
                            state.gaze.head_roll_deg,
                            state.gaze.confidence,
                        )
                    print(
                        "Road gaze calibrated: "
                        f"yaw_offset={yaw:.2f}, pitch_offset={pitch:.2f}, roll_offset={state.gaze.head_roll_deg:.2f}"
                    )
                elif key == ord("r"):
                    yaw, pitch = pipeline.reset_road_gaze_calibration()
                    road_calibration_source = "DEFAULT"
                    print(f"Road gaze calibrated: yaw_offset={yaw:.2f}, pitch_offset={pitch:.2f}")
            frame_id += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        jsonl.close()
        debug_trace.close()
        learning_memory.close()
        pipeline.close()
        if args.display:
            cv2.destroyAllWindows()

    if args.output is not None:
        print(f"Wrote overlay video: {Path(args.output)}")
    if args.jsonl is not None:
        print(f"Wrote DMS JSONL: {Path(args.jsonl)}")
    if args.debug_trace is not None:
        print(f"Wrote debug trace: {Path(args.debug_trace)}")
    if args.event_log is not None:
        print(f"Wrote event log: {Path(args.event_log)}")
    if args.event_json is not None:
        print(f"Wrote event JSON: {Path(args.event_json)}")
    if args.review_bundle is not None:
        print(f"Wrote review bundle: {Path(args.review_bundle)}")
    if args.learning_memory is not None:
        print(f"Wrote learning memory: {Path(args.learning_memory)}")


if __name__ == "__main__":
    main()
