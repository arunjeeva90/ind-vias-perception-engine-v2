from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class StableStrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class CameraStatus(StableStrEnum):
    OK = "OK"
    NO_FACE = "NO_FACE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    ERROR = "ERROR"


class PresenceState(StableStrEnum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNKNOWN = "UNKNOWN"


class AvailabilityState(StableStrEnum):
    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class GazeZone(StableStrEnum):
    ROAD = "ROAD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    DOWN = "DOWN"
    UP = "UP"
    PHONE_DOWN = "PHONE_DOWN"
    UNKNOWN = "UNKNOWN"


class DrowsinessLevel(StableStrEnum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    MICROSLEEP = "MICROSLEEP"
    UNKNOWN = "UNKNOWN"


class DistractionLevel(StableStrEnum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class DistractionType(StableStrEnum):
    NONE = "NONE"
    VISUAL = "VISUAL"
    PHONE_SUSPECTED = "PHONE_SUSPECTED"
    UNKNOWN = "UNKNOWN"


class RiskLevel(StableStrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class DMSHealth:
    camera_status: CameraStatus = CameraStatus.ERROR
    face_visibility_score: float = 0.0
    eye_visibility_score: float = 0.0
    confidence: float = 0.0


@dataclass
class DriverPresence:
    state: PresenceState = PresenceState.UNKNOWN
    confidence: float = 0.0


@dataclass
class DriverAvailability:
    state: AvailabilityState = AvailabilityState.UNKNOWN
    score: float = 0.0
    reason_codes: list[str] = field(default_factory=list)


@dataclass
class GazeState:
    zone: GazeZone = GazeZone.UNKNOWN
    eyes_off_road_duration_ms: int = 0
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    head_roll_deg: float = 0.0
    confidence: float = 0.0


@dataclass
class DrowsinessState:
    level: DrowsinessLevel = DrowsinessLevel.UNKNOWN
    perclos_5s: float = 0.0
    perclos_60s: float = 0.0
    eye_closure_duration_ms: int = 0
    blink_rate_per_min: float = 0.0
    confidence: float = 0.0


@dataclass
class DistractionState:
    level: DistractionLevel = DistractionLevel.UNKNOWN
    type: DistractionType = DistractionType.UNKNOWN
    duration_ms: int = 0
    confidence: float = 0.0


@dataclass
class PlaceholderState:
    state: str = "UNKNOWN"
    confidence: float = 0.0


@dataclass
class SeatbeltAuthenticity:
    buckle_switch: str = "UNKNOWN"
    visual_belt_path: str = "UNKNOWN"
    final_state: str = "UNKNOWN"
    confidence: float = 0.0


@dataclass
class DriverReadinessScore:
    score_0_to_1: float = 0.0
    risk_level: RiskLevel = RiskLevel.UNKNOWN


@dataclass
class DMSState:
    timestamp_ms: int = 0
    frame_id: int = 0
    dms_health: DMSHealth = field(default_factory=DMSHealth)
    driver_presence: DriverPresence = field(default_factory=DriverPresence)
    driver_availability: DriverAvailability = field(default_factory=DriverAvailability)
    gaze: GazeState = field(default_factory=GazeState)
    drowsiness: DrowsinessState = field(default_factory=DrowsinessState)
    distraction: DistractionState = field(default_factory=DistractionState)
    phone_use: PlaceholderState = field(default_factory=PlaceholderState)
    seatbelt_authenticity: SeatbeltAuthenticity = field(default_factory=SeatbeltAuthenticity)
    driver_readiness_score: DriverReadinessScore = field(default_factory=DriverReadinessScore)

    def to_dict(self) -> dict[str, Any]:
        return _enum_to_value(asdict(self))


def _enum_to_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _enum_to_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_enum_to_value(item) for item in value]
    return value
