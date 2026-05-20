from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from ind_vias_perception.common.types import CameraCalibration, VehicleConfig


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    camera: CameraCalibration
    vehicle: VehicleConfig


def load_settings(path: str | Path = "configs/default.yaml") -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cam = raw["camera"]
    camera = CameraCalibration(
        fx_px=float(_camera_value(cam, "fx_px", "fx")),
        fy_px=float(_camera_value(cam, "fy_px", "fy")),
        cx_px=float(_camera_value(cam, "cx_px", "cx")),
        cy_px=float(_camera_value(cam, "cy_px", "cy")),
        height_m=float(_camera_value(cam, "height_m", "camera_height_m")),
        pitch_deg=float(cam["pitch_deg"]),
        horizon_v_px=float(_camera_value(cam, "horizon_v_px", "horizon_y")),
        image_width=int(cam.get("image_width", 0)),
        image_height=int(cam.get("image_height", 0)),
        min_distance_m=float(cam.get("min_distance_m", 2.0)),
        max_distance_m=float(cam.get("max_distance_m", 120.0)),
    )
    vehicle_raw = raw.get("vehicle", {})
    vehicle = VehicleConfig(
        camera_to_front_bumper_offset_m=float(
            vehicle_raw.get("camera_to_front_bumper_offset_m", 0.0)
        )
    )
    return Settings(raw=raw, camera=camera, vehicle=vehicle)


def _camera_value(cam: dict[str, Any], primary: str, alias: str) -> Any:
    if primary in cam:
        return cam[primary]
    return cam[alias]
