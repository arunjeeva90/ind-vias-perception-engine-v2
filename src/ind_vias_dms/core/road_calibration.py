from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class RoadCalibration:
    yaw_offset_deg: float = 0.0
    pitch_offset_deg: float = 0.0
    roll_offset_deg: float = 0.0
    road_axis_yaw_ref_deg: float = 0.0
    road_axis_pitch_ref_deg: float = 0.0
    road_axis_roll_ref_deg: float = 0.0
    road_axis_confidence: float = 0.0
    calibrated: bool = False
    source: str = "DEFAULT"


def load_road_calibration(path: str | Path) -> RoadCalibration:
    calibration_path = Path(path)
    if not calibration_path.exists():
        return RoadCalibration()
    with open(calibration_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return RoadCalibration(
        yaw_offset_deg=float(raw.get("yaw_offset_deg", 0.0)),
        pitch_offset_deg=float(raw.get("pitch_offset_deg", 0.0)),
        roll_offset_deg=float(raw.get("roll_offset_deg", raw.get("road_axis_roll_ref_deg", 0.0))),
        road_axis_yaw_ref_deg=float(raw.get("road_axis_yaw_ref_deg", raw.get("yaw_offset_deg", 0.0))),
        road_axis_pitch_ref_deg=float(raw.get("road_axis_pitch_ref_deg", raw.get("pitch_offset_deg", 0.0))),
        road_axis_roll_ref_deg=float(raw.get("road_axis_roll_ref_deg", raw.get("roll_offset_deg", 0.0))),
        road_axis_confidence=float(raw.get("road_axis_confidence", 1.0 if raw.get("calibrated", False) else 0.0)),
        calibrated=bool(raw.get("calibrated", False)),
        source="FILE",
    )


def save_road_calibration(
    path: str | Path,
    yaw_offset_deg: float,
    pitch_offset_deg: float,
    roll_offset_deg: float = 0.0,
    road_axis_confidence: float = 1.0,
) -> None:
    calibration_path = Path(path)
    if calibration_path.parent != Path("."):
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "yaw_offset_deg": float(yaw_offset_deg),
        "pitch_offset_deg": float(pitch_offset_deg),
        "roll_offset_deg": float(roll_offset_deg),
        "road_axis_yaw_ref_deg": float(yaw_offset_deg),
        "road_axis_pitch_ref_deg": float(pitch_offset_deg),
        "road_axis_roll_ref_deg": float(roll_offset_deg),
        "road_axis_confidence": float(road_axis_confidence),
        "calibrated": True,
    }
    with open(calibration_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True)
