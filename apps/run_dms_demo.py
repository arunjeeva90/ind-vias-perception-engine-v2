from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2  # noqa: E402

from ind_vias_dms.core.config import load_dms_config  # noqa: E402
from ind_vias_dms.core.pipeline import DMSPipeline  # noqa: E402
from ind_vias_dms.core.road_calibration import load_road_calibration, save_road_calibration  # noqa: E402
from ind_vias_dms.core.vehicle_state import VehicleStateManager  # noqa: E402
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
    parser.add_argument("--start-ms", type=int, default=None)
    parser.add_argument("--end-ms", type=int, default=None)
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
    parser.add_argument("--live-output-fps", choices=("measured", "camera", "fixed"), default="measured")
    parser.add_argument("--output-fps", type=float, default=None)
    parser.add_argument("--keyframe-before-ms", type=int, default=500)
    parser.add_argument("--keyframe-after-ms", type=int, default=500)
    parser.add_argument("--cabin-evidence-backend", choices=("dummy", "synthetic", "manual", "onnx"), default=None)
    parser.add_argument("--cabin-evidence-timeline", default=None)
    parser.add_argument("--cabin-evidence-model", default=None)
    parser.add_argument("--cabin-evidence-class-map", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_dms_config(args.config)
    config = _apply_cabin_evidence_overrides(config, args)
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
    vehicle_manager = VehicleStateManager(config, output_fps_mode=args.live_output_fps)
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
    processed_frames = 0
    if args.start_ms is not None:
        if args.video is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0, args.start_ms))
            frame_id = int((max(0, args.start_ms) / 1000.0) * fps)
        else:
            raise SystemExit("--start-ms/--end-ms are supported for --video sources only")

    try:
        while args.max_frames is None or processed_frames < args.max_frames:
            capture_start = time.perf_counter()
            ok, frame = cap.read()
            capture_elapsed_ms = (time.perf_counter() - capture_start) * 1000.0
            if not ok:
                break
            frame = resize_to_width(frame, config.frame_resize_width)
            timestamp_ms = int((frame_id / fps) * 1000)
            if args.end_ms is not None and timestamp_ms > args.end_ms:
                break
            process_start = time.perf_counter()
            state, context = pipeline.process(frame, timestamp_ms, frame_id)
            process_elapsed_ms = (time.perf_counter() - process_start) * 1000.0
            state.gaze.calibration_source = road_calibration_source
            live_output_fps = _select_output_fps(
                args.live_output_fps,
                args.output_fps,
                args.camera is not None,
                fps,
                float(context["fps"]),
                config.output_fps,
            )
            state = vehicle_manager.update(
                state,
                timestamp_ms=timestamp_ms,
                live_output_fps=live_output_fps,
                frame_capture_time_ms=capture_elapsed_ms,
                processing_time_ms=process_elapsed_ms,
            )

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
                    driver_roi_norm=context["driver_roi_norm"],
                    show_debug_proposal_boxes=(
                        args.show_debug_proposal_boxes
                        or config.show_debug_proposal_boxes
                        or config.show_raw_face_proposals
                    ),
                )
            if args.output is not None:
                if writer is None:
                    writer = make_video_writer(args.output, live_output_fps, annotated)
                write_start = time.perf_counter()
                writer.write(annotated)
                state.vehicle.frame_write_time_ms = (time.perf_counter() - write_start) * 1000.0
            jsonl.write(serialize_dms_state(state))
            debug_trace.write_frame(state, context, frame)
            learning_memory.write_frame(state, context, frame)
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
                    if config.vehicle_monitor_window_enabled:
                        cv2.imshow("IND-VIAS Vehicle Monitor", overlay.render_vehicle_monitor(state))
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
                elif key in {ord("="), ord("+")}:
                    vehicle_manager.increase_speed(fast=key == ord("+"))
                    print(f"Sim speed: {vehicle_manager.speed_kph:.1f} km/h")
                elif key == ord("-"):
                    vehicle_manager.decrease_speed()
                    print(f"Sim speed: {vehicle_manager.speed_kph:.1f} km/h")
                elif key == ord("9"):
                    vehicle_manager.toggle_left_indicator()
                    print(f"Left indicator: {'ON' if vehicle_manager.left_indicator_on else 'OFF'}")
                elif key == ord("0"):
                    vehicle_manager.toggle_right_indicator()
                    print(f"Right indicator: {'ON' if vehicle_manager.right_indicator_on else 'OFF'}")
            frame_id += 1
            processed_frames += 1
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


def _select_output_fps(
    mode: str,
    requested_fps: float | None,
    is_camera: bool,
    source_fps: float,
    measured_fps: float,
    config_fps: float,
) -> float:
    if not is_camera:
        return max(1.0, float(source_fps or requested_fps or config_fps))
    if mode == "fixed":
        return max(1.0, float(requested_fps or config_fps))
    if mode == "camera":
        return max(1.0, float(source_fps or requested_fps or config_fps))
    return max(1.0, float(measured_fps or source_fps or requested_fps or config_fps))


def _apply_cabin_evidence_overrides(config, args):
    if (
        args.cabin_evidence_backend is None
        and args.cabin_evidence_timeline is None
        and args.cabin_evidence_model is None
        and args.cabin_evidence_class_map is None
    ):
        return config
    cabin_evidence = dict(config.cabin_evidence or {})
    if args.cabin_evidence_backend is not None:
        cabin_evidence["detector_backend"] = args.cabin_evidence_backend
    if args.cabin_evidence_timeline is not None:
        cabin_evidence["synthetic_timeline_path"] = args.cabin_evidence_timeline
    if args.cabin_evidence_model is not None:
        cabin_evidence["model_path"] = args.cabin_evidence_model
    if args.cabin_evidence_class_map is not None:
        cabin_evidence["class_map_path"] = args.cabin_evidence_class_map
    return replace(config, cabin_evidence=cabin_evidence)


if __name__ == "__main__":
    main()


