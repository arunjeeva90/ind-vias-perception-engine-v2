from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DMSConfig:
    face_backend: str = "mediapipe"
    frame_resize_width: int = 960
    output_fps: float = 30.0
    perclos_short_window_s: float = 5.0
    perclos_long_window_s: float = 60.0
    eye_closed_threshold: float = 0.21
    blink_min_duration_ms: int = 80
    microsleep_duration_ms: int = 1500
    eyes_off_road_warning_ms: int = 2000
    head_yaw_left_threshold_deg: float = -25.0
    head_yaw_right_threshold_deg: float = 25.0
    head_pitch_down_threshold_deg: float = 18.0
    head_pitch_up_threshold_deg: float = -18.0
    drowsiness_perclos_medium: float = 0.25
    drowsiness_perclos_high: float = 0.40
    no_face_timeout_ms: int = 1000
    overlay_enabled: bool = True
    telemetry_panel_enabled: bool = True


def load_dms_config(path: str | Path | None = None) -> DMSConfig:
    if path is None:
        path = "configs/dms/dualsight_dms_v0_1.yaml"
    config_path = Path(path)
    raw: dict[str, Any] = {}
    if config_path.exists():
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"DMS config must be a YAML mapping: {config_path}")
        raw = loaded
    valid_keys = set(DMSConfig.__dataclass_fields__)
    return DMSConfig(**{key: value for key, value in raw.items() if key in valid_keys})
