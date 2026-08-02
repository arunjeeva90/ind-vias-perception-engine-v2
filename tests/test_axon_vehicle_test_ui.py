from __future__ import annotations

import json

from apps.run_dms_demo import (
    STATUS_WINDOW,
    VEHICLE_WINDOW,
    VIDEO_WINDOW,
    _apply_landmark_106_override,
    _configure_display_windows,
    _status_dashboard_size,
    _write_feedback_session_json,
    build_parser,
)
from ind_vias_dms.core.config import load_dms_config
from ind_vias_dms.core.types import DMSState
from ind_vias_dms.visualization.overlay import (
    _prioritize_status_dashboard_lines,
    status_dashboard_lines,
)


AXON_CONFIG = "configs/dms/dualsight_dms_axon.yaml"


def test_axon_vehicle_test_profile_restores_ui_and_keeps_unaccepted_models_off():
    config = load_dms_config(AXON_CONFIG)

    assert config.status_window_enabled is True
    assert config.vehicle_monitor_window_enabled is True
    assert config.draw_pose_axes is True
    assert config.draw_gaze_vector is True
    assert config.dms_activation_speed_kph == 30.0
    assert config.face_mesh_on_crops is False
    assert config.max_num_faces == 4
    assert config.draw_all_faces is True
    assert config.driver_largest_face_in_roi_priority is True
    assert config.retain_non_driver_faces_in_driver_roi is True
    assert config.retain_non_driver_landmarks is False
    assert config.non_driver_reject_static_headrest_like_boxes is False
    assert config.performance["show_perf"] is False
    assert config.mobile_distraction_enabled is False
    assert config.eye_state_classifier["enabled"] is False
    assert config.seatbelt_detection["enabled"] is False
    assert config.phone_object_detection["enabled"] is False
    assert config.cabin_evidence["detector_backend"] == "dummy"


def test_vehicle_test_window_layout_is_explicit_and_screenshot_sized(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        "apps.run_dms_demo.cv2.namedWindow",
        lambda *args: calls.append(("named",) + args),
    )
    monkeypatch.setattr(
        "apps.run_dms_demo.cv2.resizeWindow",
        lambda *args: calls.append(("resize",) + args),
    )
    monkeypatch.setattr(
        "apps.run_dms_demo.cv2.moveWindow",
        lambda *args: calls.append(("move",) + args),
    )

    _configure_display_windows(
        use_status_window=True,
        show_vehicle_monitor=True,
        layout="vehicle-test",
    )

    assert ("resize", VIDEO_WINDOW, 960, 720) in calls
    assert ("resize", STATUS_WINDOW, 720, 1000) in calls
    assert ("resize", VEHICLE_WINDOW, 780, 390) in calls
    assert _status_dashboard_size("vehicle-test") == (900, 1250)


def test_vehicle_test_cli_and_status_put_live_eye_and_vehicle_signals_first():
    args = build_parser().parse_args(
        ["--camera", "0", "--display", "--window-layout", "vehicle-test"]
    )
    assert args.window_layout == "vehicle-test"

    rows = status_dashboard_lines(
        DMSState(),
        fps=15.0,
        eye_runtime_source="LANDMARK_EAR",
        eye_model_status="DISABLED",
        landmark_106_status="DISABLED",
    )
    prioritized = _prioritize_status_dashboard_lines(rows)
    first_page = dict(prioritized[:50])

    assert first_page["Eye runtime"] == "LANDMARK_EAR"
    assert first_page["Eye CNN"] == "DISABLED"
    assert first_page["106 geometry"] == "DISABLED"
    assert "Vehicle gate" in first_page
    assert "Head angle" in first_page
    assert "PERCLOS 5s/60s" in first_page
    assert "HMI banner" in first_page


def test_106_rknn_requires_explicit_runtime_opt_in():
    config = load_dms_config(AXON_CONFIG)

    assert config.eye_state_classifier["landmark_106_enabled"] is False
    enabled = _apply_landmark_106_override(config, "rknn")

    assert enabled.eye_state_classifier["landmark_106_enabled"] is True
    assert enabled.eye_state_classifier["landmark_106_backend"] == "rknn"


def test_feedback_session_json_is_valid_and_self_describing(tmp_path):
    args = build_parser().parse_args(
        [
            "--camera",
            "0",
            "--output",
            "run/webcam_output.mp4",
            "--jsonl",
            "run/webcam_state.jsonl",
            "--perf-jsonl",
            "run/webcam_performance.jsonl",
        ]
    )
    output = tmp_path / "webcam_session.json"

    _write_feedback_session_json(
        str(output),
        started_at_utc="2026-08-02T12:00:00+00:00",
        status="COMPLETED",
        error=None,
        args=args,
        processed_frames=42,
        last_frame_id=41,
        compute_backend="CPU / MediaPipe XNNPACK",
        npu_active=False,
        final_performance={"feature_latency_ms": 55.5},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "axon_dms_feedback_session_v1"
    assert payload["processed_frames"] == 42
    assert payload["outputs"]["overlay_video"] == "run/webcam_output.mp4"
    assert payload["outputs"]["state_jsonl"] == "run/webcam_state.jsonl"
    assert payload["final_performance"]["feature_latency_ms"] == 55.5
