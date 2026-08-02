from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
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
from ind_vias_dms.utils.latest_frame_capture import LatestFrameCapture  # noqa: E402
from ind_vias_dms.utils.axon_face_box_tracker import AxonFaceBoxTracker  # noqa: E402
from ind_vias_dms.utils.learning_memory import LearningMemoryWriter  # noqa: E402
from ind_vias_dms.utils.perf_monitor import RuntimePerfMonitor  # noqa: E402
from ind_vias_dms.utils.video_io import make_video_writer, open_video_source, resize_to_width  # noqa: E402
from ind_vias_dms.visualization.overlay import OverlayRenderer  # noqa: E402


VIDEO_WINDOW = "IND-VIAS DualSight DMS - Video"
STATUS_WINDOW = "IND-VIAS DualSight DMS - Status"
VEHICLE_WINDOW = "IND-VIAS Vehicle Monitor"


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
    parser.add_argument(
        "--window-layout",
        choices=("opencv", "vehicle-test"),
        default="opencv",
        help=(
            "GUI layout. 'vehicle-test' restores the three-window AXON layout "
            "used for in-vehicle validation."
        ),
    )
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
    parser.add_argument("--camera-fps", type=float, default=None)
    parser.add_argument("--inference-fps", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fourcc", default=None)
    parser.add_argument("--latest-frame-buffer", dest="latest_frame_buffer", action="store_true", default=None)
    parser.add_argument("--disable-latest-frame-buffer", dest="latest_frame_buffer", action="store_false")
    parser.add_argument("--enable-opencl", dest="opencl", action="store_true", default=None)
    parser.add_argument("--disable-opencl", dest="opencl", action="store_false")
    parser.add_argument("--model-gops-per-frame", type=float, default=None)
    parser.add_argument("--show-perf", action="store_true", default=None)
    parser.add_argument("--axon-face-box-tracker", action="store_true")
    parser.add_argument("--perf-jsonl", default=None)
    parser.add_argument(
        "--session-json",
        default=None,
        help=(
            "Write a compact valid-JSON feedback manifest with run metadata, "
            "output paths, final performance, and runtime error status."
        ),
    )
    parser.add_argument(
        "--landmark-106-backend",
        choices=("disabled", "onnx", "rknn"),
        default=None,
        help=(
            "Optional driver-only 106-point confidence backend. RKNN remains "
            "internal-PoC-only and requires a working RKNNLite/NPU runtime."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_dms_config(args.config)
    config = _apply_cabin_evidence_overrides(config, args)
    config = _apply_landmark_106_override(config, args.landmark_106_backend)
    runtime_cfg = config.runtime or {}
    perf_cfg = config.performance or {}
    camera_fps = _cfg_float(args.camera_fps, runtime_cfg, "camera_fps", 20.0)
    inference_fps = _cfg_float(args.inference_fps, runtime_cfg, "inference_fps", 15.0)
    camera_width = _cfg_int(args.width, runtime_cfg, "width", 640)
    camera_height = _cfg_int(args.height, runtime_cfg, "height", 480)
    fourcc = str(args.fourcc or runtime_cfg.get("fourcc") or "MJPG")
    latest_frame_enabled = _cfg_bool(args.latest_frame_buffer, runtime_cfg, "latest_frame_buffer", True)
    show_perf = _cfg_bool(args.show_perf, perf_cfg, "show_perf", False)
    model_gops_per_frame = _cfg_float(args.model_gops_per_frame, perf_cfg, "model_gops_per_frame", 0.0)
    compute_backend = "CPU / MediaPipe XNNPACK"
    npu_active = False

    _configure_opencl(args.opencl, runtime_cfg)

    cap = open_video_source(args.video, args.camera)
    if args.camera is not None:
        _configure_camera_capture(cap, args.camera, camera_width, camera_height, camera_fps, fourcc)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = camera_fps if args.camera is not None else config.output_fps

    try:
        pipeline = DMSPipeline(config)
    except RuntimeError as exc:
        cap.release()
        raise SystemExit(str(exc)) from exc
    landmark_106_backend = pipeline.eye_state_estimator.landmark_106
    if landmark_106_backend.npu_active:
        compute_backend = "CPU MediaPipe + RKNN NPU (106)"
        npu_active = True
    print(f"Compute backend: {compute_backend}")
    if npu_active:
        print("NPU active: YES (driver-only 106-point confidence evidence)")
        print("NPU TOPS counter: unavailable from RKNNLite")
    else:
        print(
            "NPU active: NO "
            f"(106 backend status: {landmark_106_backend.backend_status})"
        )
    if show_perf:
        print("Performance telemetry enabled in status consoles/JSONL; camera overlay is disabled")
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
    perf_monitor = RuntimePerfMonitor(
        model_gops_per_frame=model_gops_per_frame,
        jsonl_path=args.perf_jsonl,
        compute_backend=compute_backend,
        npu_active=npu_active,
    )
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
    last_inference_time_s = 0.0
    last_inference_elapsed_ms = 0.0
    last_inference_state = None
    last_inference_context = None
    latest_capture = None
    pending_latest_frame = None
    axon_face_tracker = AxonFaceBoxTracker() if args.axon_face_box_tracker else None
    start_wall_s = time.time()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    session_status = "COMPLETED"
    session_error = None
    perf_payload: dict[str, object] = {}
    last_latest_frame_id = -1
    if args.start_ms is not None:
        if args.video is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0, args.start_ms))
            frame_id = int((max(0, args.start_ms) / 1000.0) * fps)
        else:
            raise SystemExit("--start-ms/--end-ms are supported for --video sources only")
    if args.camera is not None and latest_frame_enabled:
        latest_capture = LatestFrameCapture(cap, debug=True).start()
        print("Latest-frame buffer enabled")
        pending_latest_frame = latest_capture.read(timeout_s=3.0)
        if not pending_latest_frame[0]:
            if latest_capture.last_error:
                print(f"Latest-frame buffer error: {latest_capture.last_error}")
            print(f"Latest-frame captured_frames at shutdown: {latest_capture.captured_frames}")
            print(f"Latest-frame overwritten/dropped frames at shutdown: {latest_capture.overwritten_frames}")
            latest_capture.stop()
            latest_capture = None
            pending_latest_frame = None
            print("Latest-frame buffer failed; falling back to direct camera capture")
    elif args.camera is not None:
        print("Latest-frame buffer disabled")

    use_status_window = args.status_window or config.status_window_enabled
    if args.display:
        _configure_display_windows(
            use_status_window=use_status_window,
            show_vehicle_monitor=(
                use_status_window and config.vehicle_monitor_window_enabled
            ),
            layout=args.window_layout,
        )
        _print_display_controls(config.dms_activation_speed_kph)

    try:
        while args.max_frames is None or processed_frames < args.max_frames:
            loop_start = time.perf_counter()
            capture_start = time.perf_counter()
            capture_timestamp_s = time.time()
            captured_frames = frame_id + 1
            dropped_frames = 0
            if latest_capture is not None:
                if pending_latest_frame is not None:
                    ok, frame, latest_timestamp_s, latest_frame_id = pending_latest_frame
                    pending_latest_frame = None
                else:
                    ok, frame, latest_timestamp_s, latest_frame_id = latest_capture.read(
                        timeout_s=0.05,
                        after_frame_id=last_latest_frame_id,
                    )
                if not ok:
                    if latest_capture.last_error:
                        print(f"Latest-frame buffer error: {latest_capture.last_error}")
                        print(f"Latest-frame captured_frames at shutdown: {latest_capture.captured_frames}")
                        print(f"Latest-frame overwritten/dropped frames at shutdown: {latest_capture.overwritten_frames}")
                        latest_capture.stop()
                        latest_capture = None
                        print("Latest-frame buffer failed; falling back to direct camera capture")
                    else:
                        time.sleep(0.002)
                    continue
                last_latest_frame_id = latest_frame_id
                frame_id = latest_frame_id
                capture_timestamp_s = latest_timestamp_s
                captured_frames = latest_capture.captured_frames
                dropped_frames = latest_capture.overwritten_frames
            else:
                ok, frame = cap.read()
            capture_elapsed_ms = (time.perf_counter() - capture_start) * 1000.0
            if not ok or frame is None:
                break
            frame = resize_to_width(frame, config.frame_resize_width)
            if args.camera is not None:
                timestamp_ms = int(max(0.0, capture_timestamp_s - start_wall_s) * 1000.0)
            else:
                timestamp_ms = int((frame_id / fps) * 1000)
            if args.end_ms is not None and timestamp_ms > args.end_ms:
                break
            now_s = time.perf_counter()
            inference_interval_s = 1.0 / inference_fps if inference_fps > 0 else 0.0
            inference_ran = (
                last_inference_state is None
                or args.camera is None
                or inference_interval_s <= 0.0
                or now_s - last_inference_time_s >= inference_interval_s
            )
            inference_elapsed_ms = 0.0
            if inference_ran:
                process_start = time.perf_counter()
                state, context = pipeline.process(frame, timestamp_ms, frame_id)
                inference_elapsed_ms = (time.perf_counter() - process_start) * 1000.0
                last_inference_elapsed_ms = inference_elapsed_ms
                process_elapsed_ms = inference_elapsed_ms
                last_inference_time_s = time.perf_counter()
                last_inference_state = copy.deepcopy(state)
                last_inference_context = context
            else:
                state = copy.deepcopy(last_inference_state)
                state.timestamp_ms = timestamp_ms
                state.frame_id = frame_id
                context = last_inference_context
                process_elapsed_ms = 0.0
            if axon_face_tracker is not None:
                if inference_ran:
                    axon_face_tracker.update_from_detection(
                        frame,
                        context["face"].bbox,
                        timestamp_ms,
                    )
                else:
                    tracked_box = axon_face_tracker.track(frame, timestamp_ms)
                    if tracked_box is not None:
                        context["face"].bbox = tracked_box.bbox
                        # AXON fast-live mode: the bbox is updated by lightweight
                        # optical-flow tracking, but FaceMesh landmarks are from
                        # the previous full inference frame. Hide stale landmarks
                        # so dots do not trail behind the moving face.
                        context["face"].landmarks_px = {}
            feature_latency_ms = max(
                0.0,
                (time.time() - capture_timestamp_s) * 1000.0,
            )

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
            overlay_start = time.perf_counter()
            if args.debug_overlay or config.overlay_enabled:
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
            overlay_elapsed_ms = (time.perf_counter() - overlay_start) * 1000.0
            if args.output is not None:
                if writer is None:
                    writer = make_video_writer(args.output, live_output_fps, annotated)
                write_start = time.perf_counter()
                writer.write(annotated)
                state.vehicle.frame_write_time_ms = (time.perf_counter() - write_start) * 1000.0
            jsonl.write(serialize_dms_state(state))
            debug_trace.write_frame(state, context, frame)
            learning_memory.write_frame(state, context, frame)
            displayed = False
            if args.display:
                if use_status_window:
                    cv2.imshow(VIDEO_WINDOW, annotated)
                    eye_runtime = context["eye_state"]
                    status_width, status_height = _status_dashboard_size(args.window_layout)
                    runtime_metrics = perf_monitor.snapshot()
                    runtime_metrics.update(
                        {
                            "processing_fps": float(context["fps"]),
                            "feature_latency_ms": feature_latency_ms,
                            "inference_time_ms": last_inference_elapsed_ms,
                            "landmark_106_inference_ms": (
                                eye_runtime.landmark_106_inference_ms
                            ),
                        }
                    )
                    status = overlay.render_status_dashboard(
                        state,
                        float(context["fps"]),
                        width=status_width,
                        height=status_height,
                        road_yaw_offset_deg=pipeline.gaze_estimator.yaw_offset_deg,
                        road_pitch_offset_deg=pipeline.gaze_estimator.pitch_offset_deg,
                        road_calibrated=pipeline.gaze_estimator.road_gaze_calibrated,
                        vehicle_layout=config.vehicle_layout,
                        driver_image_side=config.driver_image_side,
                        camera_mount_position=config.camera_mount_position,
                        camera_view_direction=config.camera_view_direction,
                        driver_roi_state=pipeline.occupants.driver_roi_state(),
                        eye_runtime_source=eye_runtime.classification_source,
                        eye_model_status=eye_runtime.model_backend_status,
                        landmark_106_status=eye_runtime.landmark_106_status,
                        runtime_metrics=runtime_metrics,
                        compute_backend=compute_backend,
                        npu_active=npu_active,
                    )
                    cv2.imshow(STATUS_WINDOW, status)
                    if config.vehicle_monitor_window_enabled:
                        cv2.imshow(
                            VEHICLE_WINDOW,
                            overlay.render_vehicle_monitor(
                                state,
                                # Modest supersampling keeps text clear without
                                # imposing the substantial cost of a 2x canvas.
                                width=975,
                                height=488,
                                runtime_metrics=runtime_metrics,
                                compute_backend=compute_backend,
                                npu_active=npu_active,
                            ),
                        )
                else:
                    cv2.imshow("IND-VIAS DualSight DMS", annotated)
                key = cv2.waitKey(1) & 0xFF
                displayed = True
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
            processed_frames += 1
            loop_elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
            perf_payload = perf_monitor.update(
                timestamp_s=time.time(),
                frame_id=frame_id,
                frame_latency_ms=feature_latency_ms,
                inference_time_ms=last_inference_elapsed_ms,
                overlay_time_ms=overlay_elapsed_ms,
                loop_time_ms=loop_elapsed_ms,
                inference_ran=inference_ran,
                captured_frames=captured_frames,
                processed_frames=processed_frames,
                dropped_frames=dropped_frames,
                displayed=displayed,
            )
            perf_monitor.write_jsonl(perf_payload)
            if latest_capture is None:
                frame_id += 1
    except BaseException as exc:
        session_status = "ERROR"
        session_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if latest_capture is not None:
            print(f"Latest-frame captured_frames at shutdown: {latest_capture.captured_frames}")
            print(f"Latest-frame overwritten/dropped frames at shutdown: {latest_capture.overwritten_frames}")
            latest_capture.release()
        else:
            cap.release()
        if writer is not None:
            writer.release()
        jsonl.close()
        perf_monitor.close()
        debug_trace.close()
        learning_memory.close()
        pipeline.close()
        _write_feedback_session_json(
            args.session_json,
            started_at_utc=started_at_utc,
            status=session_status,
            error=session_error,
            args=args,
            processed_frames=processed_frames,
            last_frame_id=frame_id,
            compute_backend=compute_backend,
            npu_active=npu_active,
            final_performance=perf_payload,
        )
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
    if args.perf_jsonl is not None:
        print(f"Wrote performance JSONL: {Path(args.perf_jsonl)}")
    if args.session_json is not None:
        print(f"Wrote feedback session JSON: {Path(args.session_json)}")


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


def _write_feedback_session_json(
    path: str | None,
    *,
    started_at_utc: str,
    status: str,
    error: str | None,
    args: argparse.Namespace,
    processed_frames: int,
    last_frame_id: int,
    compute_backend: str,
    npu_active: bool,
    final_performance: dict[str, object],
) -> None:
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "axon_dms_feedback_session_v1",
        "started_at_utc": started_at_utc,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error": error,
        "source": {
            "camera": args.camera,
            "video": args.video,
        },
        "config_path": args.config,
        "git_commit": _git_commit_or_unknown(),
        "compute_backend": compute_backend,
        "npu_active": npu_active,
        "processed_frames": processed_frames,
        "last_frame_id": last_frame_id,
        "final_performance": final_performance,
        "outputs": {
            "overlay_video": args.output,
            "state_jsonl": args.jsonl,
            "debug_trace_jsonl": args.debug_trace,
            "event_csv": args.event_log,
            "event_json": args.event_json,
            "learning_memory_jsonl": args.learning_memory,
            "performance_jsonl": args.perf_jsonl,
        },
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_commit_or_unknown() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"
    return result.stdout.strip() or "UNKNOWN"


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


def _apply_landmark_106_override(config, backend: str | None):
    if backend is None:
        return config
    eye_config = dict(config.eye_state_classifier or {})
    eye_config["landmark_106_backend"] = "onnx" if backend == "disabled" else backend
    eye_config["landmark_106_enabled"] = backend != "disabled"
    return replace(config, eye_state_classifier=eye_config)


def _cfg_float(cli_value: float | None, section: dict, key: str, default: float) -> float:
    if cli_value is not None:
        return float(cli_value)
    return float(section.get(key, default))


def _cfg_int(cli_value: int | None, section: dict, key: str, default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    return int(section.get(key, default))


def _cfg_bool(cli_value: bool | None, section: dict, key: str, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    return bool(section.get(key, default))


def _configure_display_windows(
    *,
    use_status_window: bool,
    show_vehicle_monitor: bool,
    layout: str,
) -> None:
    if not use_status_window:
        cv2.namedWindow("IND-VIAS DualSight DMS", cv2.WINDOW_NORMAL)
        return

    for title in (VIDEO_WINDOW, STATUS_WINDOW):
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    if show_vehicle_monitor:
        cv2.namedWindow(VEHICLE_WINDOW, cv2.WINDOW_NORMAL)

    if layout != "vehicle-test":
        return

    # Matches the operator layout used by the accepted DualSight reference:
    # video at left, live status at right, and the compact vehicle simulator
    # monitor overlapping the lower video edge. OpenCV/window-manager chrome
    # can add a few pixels around these requested client dimensions.
    cv2.resizeWindow(VIDEO_WINDOW, 960, 720)
    cv2.moveWindow(VIDEO_WINDOW, 10, 20)
    cv2.resizeWindow(STATUS_WINDOW, 720, 1000)
    cv2.moveWindow(STATUS_WINDOW, 1180, 20)
    if show_vehicle_monitor:
        cv2.resizeWindow(VEHICLE_WINDOW, 780, 390)
        cv2.moveWindow(VEHICLE_WINDOW, 415, 650)


def _status_dashboard_size(layout: str) -> tuple[int, int]:
    if layout == "vehicle-test":
        # 1.25x supersampled canvas: the window remains 720x1000, while
        # OpenCV/Qt downsamples for clearer text at a bounded rendering cost.
        return (900, 1250)
    return (480, 720)


def _print_display_controls(activation_speed_kph: float) -> None:
    print("DualSight vehicle-test controls:")
    print("  q quit | c calibrate road axis | r reset road calibration")
    print("  = +1 km/h | + +5 km/h | - -1 km/h | 9 left indicator | 0 right indicator")
    print(
        "  DMS alerts remain suppressed until simulated speed is above "
        f"{activation_speed_kph:.0f} km/h"
    )


def _configure_opencl(cli_value: bool | None, runtime_cfg: dict) -> None:
    probe = _probe_opencl_subprocess()
    if probe is None:
        print("OpenCV OpenCL available: UNKNOWN (probe failed; continuing CPU-only)")
        print("OpenCV OpenCL initial use: UNKNOWN")
        print("OpenCV OpenCL final use: False")
        return
    print(f"OpenCV OpenCL available: {probe['have_opencl']}")
    print(f"OpenCV OpenCL initial use: {probe['use_opencl']}")
    requested = cli_value
    if requested is None and "opencl" in runtime_cfg:
        requested = bool(runtime_cfg["opencl"])
    if requested is not None:
        cv2.ocl.setUseOpenCL(bool(requested))
    print(f"OpenCV OpenCL final use: {bool(cv2.ocl.useOpenCL())}")


def _probe_opencl_subprocess() -> dict[str, bool] | None:
    code = (
        "import json, cv2; "
        "print(json.dumps({'have_opencl': bool(cv2.ocl.haveOpenCL()), "
        "'use_opencl': bool(cv2.ocl.useOpenCL())}))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"OpenCV OpenCL probe error: {exc}")
        return None
    if result.returncode != 0:
        stderr = result.stderr.strip().splitlines()
        detail = stderr[-1] if stderr else f"return code {result.returncode}"
        print(f"OpenCV OpenCL probe failed: {detail}")
        return None
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        print(f"OpenCV OpenCL probe parse error: {exc}")
        return None


def _configure_camera_capture(
    cap,
    camera_index: int,
    width: int,
    height: int,
    camera_fps: float,
    fourcc: str,
) -> None:
    fourcc = (fourcc or "MJPG")[:4].ljust(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(camera_fps))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    buffersize_set = cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    actual_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    print(f"Requested camera: index={camera_index}, {width}x{height} @ {camera_fps:g} FPS, FOURCC={fourcc.strip()}")
    print(f"Actual camera: {actual_width}x{actual_height} @ {actual_fps:g} FPS, FOURCC={actual_fourcc}")
    print(f"CAP_PROP_BUFFERSIZE {'set to 1' if buffersize_set else 'not supported'}")


def _decode_fourcc(value: float) -> str:
    code = int(value)
    chars = [chr((code >> 8 * i) & 0xFF) for i in range(4)]
    text = "".join(chars).strip()
    return text or "UNKNOWN"


if __name__ == "__main__":
    main()
