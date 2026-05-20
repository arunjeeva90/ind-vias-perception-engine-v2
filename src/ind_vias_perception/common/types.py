from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class ObjectClass(str, Enum):
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    TWO_WHEELER_AGENT = "two_wheeler_agent"
    AUTO_RICKSHAW = "auto_rickshaw"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    ANIMAL = "animal"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BBox2D:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, self.y2)


@dataclass
class Detection:
    bbox: BBox2D
    label: ObjectClass
    confidence: float
    ground_contact: Optional[Tuple[float, float]] = None
    distance_m: Optional[float] = None
    sigma_depth: float = 1.0
    track_id: Optional[int] = None
    ttc_s: Optional[float] = None
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraCalibration:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    height_m: float
    pitch_deg: float
    horizon_v_px: float
    image_width: int = 0
    image_height: int = 0
    min_distance_m: float = 2.0
    max_distance_m: float = 120.0


@dataclass(frozen=True)
class VehicleConfig:
    camera_to_front_bumper_offset_m: float = 0.0


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp_s: float
    ego_speed_mps: float = 0.0
    frame_id: int = 0


@dataclass
class SceneQuality:
    glare: float = 0.0
    rain: float = 0.0
    fog: float = 0.0
    night: float = 0.0
    occlusion: float = 0.0
    complexity: float = 0.0
    ego_motion_state: str = "straight"
    yaw_score: float = 0.0
    yaw_confidence: float = 0.0
    turning_confirmation_count: int = 0
    median_dx: float = 0.0
    flow_points: int = 0

    @property
    def degraded_score(self) -> float:
        return max(self.glare, self.rain, self.fog, self.night, self.occlusion)


@dataclass
class PerceptionOutput:
    detections: List[Detection]
    scene_quality: SceneQuality
    mode: str
    safety_payload: Dict[str, object]
