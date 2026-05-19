from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from ind_vias_perception.common.types import CameraCalibration


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    camera: CameraCalibration


def load_settings(path: str | Path = "configs/default.yaml") -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cam = raw["camera"]
    camera = CameraCalibration(
        fx_px=float(cam["fx_px"]), fy_px=float(cam["fy_px"]),
        cx_px=float(cam["cx_px"]), cy_px=float(cam["cy_px"]),
        height_m=float(cam["height_m"]), pitch_deg=float(cam["pitch_deg"]),
        horizon_v_px=float(cam["horizon_v_px"]),
    )
    return Settings(raw=raw, camera=camera)
