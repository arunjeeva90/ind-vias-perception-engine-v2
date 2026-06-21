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
    NOT_VISIBLE = "NOT_VISIBLE"
    PROPOSAL_VISIBLE = "PROPOSAL_VISIBLE"
    LOST_TEMP = "LOST_TEMP"
    LOST_LONG = "LOST_LONG"
    LOST = "LOST"
    UNKNOWN = "UNKNOWN"


class AvailabilityState(StableStrEnum):
    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class DriverObservabilityState(StableStrEnum):
    OBSERVABLE = "OBSERVABLE"
    PARTIALLY_OBSERVABLE = "PARTIALLY_OBSERVABLE"
    UNOBSERVABLE_TEMP = "UNOBSERVABLE_TEMP"
    UNOBSERVABLE_LONG = "UNOBSERVABLE_LONG"
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
    MANUAL = "MANUAL"
    PHONE_TO_EAR = "PHONE_TO_EAR"
    PHONE_SUSPECTED = "PHONE_SUSPECTED"
    PHONE_CONFIRMED = "PHONE_CONFIRMED"
    UNKNOWN = "UNKNOWN"


class AttentionState(StableStrEnum):
    NORMAL = "NORMAL"
    ATTENTION_LOST = "ATTENTION_LOST"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"


class DMSV02Level(StableStrEnum):
    NORMAL = "NORMAL"
    MONITOR = "MONITOR"
    WARNING = "WARNING"
    DANGER = "DANGER"
    CRITICAL = "CRITICAL"
    DEGRADED = "DEGRADED"


class DMSConfidenceState(StableStrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNAVAILABLE = "UNAVAILABLE"


class CabinEvidenceObjectType(StableStrEnum):
    PHONE = "PHONE"
    CIGARETTE = "CIGARETTE"
    SEATBELT = "SEATBELT"
    HAND = "HAND"
    UNKNOWN_OBJECT = "UNKNOWN_OBJECT"


class CabinEvidenceRegion(StableStrEnum):
    DRIVER = "DRIVER"
    PASSENGER = "PASSENGER"
    REAR = "REAR"
    UNKNOWN = "UNKNOWN"


class CabinEvidenceRelation(StableStrEnum):
    NEAR_EAR = "NEAR_EAR"
    NEAR_LAP = "NEAR_LAP"
    NEAR_HAND = "NEAR_HAND"
    NEAR_MOUTH = "NEAR_MOUTH"
    ACROSS_TORSO = "ACROSS_TORSO"
    UNKNOWN = "UNKNOWN"


class CabinEvidenceLifecycleState(StableStrEnum):
    RAW = "RAW"
    CANDIDATE = "CANDIDATE"
    SUSPECTED = "SUSPECTED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


class CabinPhoneState(StableStrEnum):
    NO_PHONE = "NO_PHONE"
    PHONE_OBJECT_CANDIDATE = "PHONE_OBJECT_CANDIDATE"
    PHONE_IN_HAND_SUSPECTED = "PHONE_IN_HAND_SUSPECTED"
    PHONE_TO_EAR_SUSPECTED = "PHONE_TO_EAR_SUSPECTED"
    PHONE_DOWN_TEXTING_SUSPECTED = "PHONE_DOWN_TEXTING_SUSPECTED"
    PHONE_CONFIRMED = "PHONE_CONFIRMED"
    PHONE_UNKNOWN = "PHONE_UNKNOWN"


class CabinSeatbeltState(StableStrEnum):
    SEATBELT_UNKNOWN = "SEATBELT_UNKNOWN"
    SEATBELT_WORN_CONFIRMED = "SEATBELT_WORN_CONFIRMED"
    SEATBELT_NOT_VISIBLE = "SEATBELT_NOT_VISIBLE"
    SEATBELT_NOT_WORN_SUSPECTED = "SEATBELT_NOT_WORN_SUSPECTED"
    SEATBELT_MISUSE_SUSPECTED = "SEATBELT_MISUSE_SUSPECTED"
    SEATBELT_CONFIDENCE_LOW = "SEATBELT_CONFIDENCE_LOW"


class CabinSmokingState(StableStrEnum):
    NO_SMOKING = "NO_SMOKING"
    HAND_TO_MOUTH_CANDIDATE = "HAND_TO_MOUTH_CANDIDATE"
    SMOKING_SUSPECTED = "SMOKING_SUSPECTED"
    SMOKING_CONFIRMED = "SMOKING_CONFIRMED"
    SMOKING_UNKNOWN = "SMOKING_UNKNOWN"


class AttentionSubstate(StableStrEnum):
    ROAD = "ROAD"
    ROAD_AXIS_NORMAL = "ROAD_AXIS_NORMAL"
    HEAD_DOWN_CANDIDATE = "HEAD_DOWN_CANDIDATE"
    PHONE_DOWN_CANDIDATE = "PHONE_DOWN_CANDIDATE"
    HEAD_DOWN = "HEAD_DOWN"
    HEAD_DOWN_DISTRACTION = "HEAD_DOWN_DISTRACTION"
    HEAD_DOWN_UNCERTAIN = "HEAD_DOWN_UNCERTAIN"
    VISUAL_DISTRACTION = "VISUAL_DISTRACTION"
    VISUAL_OBSERVATION_LIMITED = "VISUAL_OBSERVATION_LIMITED"
    HEAD_POSE_UNRELIABLE = "HEAD_POSE_UNRELIABLE"
    FACE_PARTIAL_SIDE_PROFILE = "FACE_PARTIAL_SIDE_PROFILE"
    PHONE_SUSPECTED = "PHONE_SUSPECTED"
    PHONE_DOWN_SUSPECTED = "PHONE_DOWN_SUSPECTED"
    PHONE_TEXTING_SCROLLING_SUSPECTED = "PHONE_TEXTING_SCROLLING_SUSPECTED"
    PHONE_TEXTING_SCROLLING_CONFIRMED = "PHONE_TEXTING_SCROLLING_CONFIRMED"
    PHONE_TO_EAR_SUSPECTED = "PHONE_TO_EAR_SUSPECTED"
    TEXTING_SUSPECTED = "TEXTING_SUSPECTED"
    PHONE_CONFIRMED = "PHONE_CONFIRMED"
    SIDE_GLANCE_LEFT = "SIDE_GLANCE_LEFT"
    SIDE_GLANCE_RIGHT = "SIDE_GLANCE_RIGHT"
    SIDE_PROFILE_TRACKED = "SIDE_PROFILE_TRACKED"
    SIDE_PROFILE_ATTENTION_LOSS = "SIDE_PROFILE_ATTENTION_LOSS"
    SIDE_PROFILE_RECOVERY = "SIDE_PROFILE_RECOVERY"
    VISUAL_AWAY = "VISUAL_AWAY"
    DROWSY = "DROWSY"
    MICROSLEEP = "MICROSLEEP"
    AMBIGUOUS = "AMBIGUOUS"
    FACE_LOST = "FACE_LOST"
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
    face_detection_status: CameraStatus = CameraStatus.ERROR
    face_visibility_score: float = 0.0
    eye_visibility_score: float = 0.0
    confidence: float = 0.0
    face_backend: str = "UNKNOWN"
    nir_mode: str = "UNKNOWN"
    nir_mode_detected: str = "UNKNOWN"
    input_color_mode: str = "UNKNOWN"
    active_eye_threshold_profile: str = "UNKNOWN"
    active_perclos_profile: str = "UNKNOWN"
    nir_preprocessing_active: bool = False
    nir_reason_codes: list[str] = field(default_factory=list)
    face_proposals: int = 0
    face_detection_confidence: float = 0.0


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
class DriverObservability:
    state: DriverObservabilityState = DriverObservabilityState.UNKNOWN
    confidence: float = 0.0
    reason_codes: list[str] = field(default_factory=list)


@dataclass
class GazeState:
    zone: GazeZone = GazeZone.UNKNOWN
    eyes_off_road_duration_ms: int = 0
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    head_roll_deg: float = 0.0
    confidence: float = 0.0
    calibration_source: str = "DEFAULT"
    head_pose_raw_yaw_deg: float = 0.0
    head_pose_raw_pitch_deg: float = 0.0
    head_pose_raw_roll_deg: float = 0.0
    road_axis_yaw_ref_deg: float = 0.0
    road_axis_pitch_ref_deg: float = 0.0
    road_axis_roll_ref_deg: float = 0.0
    relative_yaw_deg: float = 0.0
    relative_pitch_deg: float = 0.0
    relative_roll_deg: float = 0.0
    head_angle_from_road_deg: float = 0.0
    head_pose_vector_quality: float = 0.0
    head_yaw_relative_label: str = "0"
    head_pitch_relative_label: str = "0"
    head_roll_relative_label: str = "0"
    status_head_angle_line: str = ""
    status_head_raw_rel_line: str = ""
    head_angle_display_visible: bool = False
    road_axis_calibration_source: str = "DEFAULT"
    road_axis_calibration_confidence: float = 0.0


@dataclass
class DrowsinessState:
    level: DrowsinessLevel = DrowsinessLevel.UNKNOWN
    eye_state: str = "UNKNOWN"
    raw_eye_state: str = "UNKNOWN"
    effective_eye_state: str = "UNKNOWN"
    eye_openness_raw: float = 0.0
    eye_openness_normalized: float = 0.0
    eye_calibration_state: str = "FALLBACK"
    eye_visibility_score: float = 0.0
    perclos_valid: bool = False
    perclos_validity_reason_codes: list[str] = field(default_factory=list)
    perclos_valid_time_5s_ms: int = 0
    perclos_valid_time_60s_ms: int = 0
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
    reason_codes: list[str] = field(default_factory=list)


@dataclass
class PlaceholderState:
    state: str = "UNKNOWN"
    confidence: float = 0.0


@dataclass
class PhoneUseState:
    state: str = "UNKNOWN"
    confidence: float = 0.0
    driver_state: str = "UNKNOWN"
    cabin_events: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    phone_object_detected: bool = False
    phone_object_bbox: list[float] = field(default_factory=list)
    phone_object_confidence: float = 0.0
    phone_object_region: str = "UNKNOWN"
    phone_object_backend_status: str = "NOT_CONFIGURED"
    phone_evidence_score: float = 0.0
    phone_texting_candidate_ms: int = 0
    phone_down_candidate_ms: int = 0
    phone_to_ear_candidate_ms: int = 0
    phone_final_state: str = "UNKNOWN"


@dataclass
class CabinEvidenceObject:
    object_type: CabinEvidenceObjectType = CabinEvidenceObjectType.UNKNOWN_OBJECT
    bbox: list[float] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "dummy"
    region: CabinEvidenceRegion = CabinEvidenceRegion.UNKNOWN
    relation_to_driver: CabinEvidenceRelation = CabinEvidenceRelation.UNKNOWN
    first_seen_ms: int = 0
    last_seen_ms: int = 0
    duration_ms: int = 0
    stable_count: int = 0
    state: CabinEvidenceLifecycleState = CabinEvidenceLifecycleState.RAW


@dataclass
class CabinEvidenceState:
    enabled: bool = True
    detector_backend: str = "dummy"
    backend_status: str = "DUMMY"
    model_path: str = ""
    class_map_path: str = ""
    synthetic_active: bool = False
    affect_final_dms_state: bool = False
    phone_state: CabinPhoneState = CabinPhoneState.NO_PHONE
    phone_relation: str = "NONE"
    phone_source: str = "NONE"
    phone_confidence: float = 0.0
    seatbelt_state: CabinSeatbeltState = CabinSeatbeltState.SEATBELT_UNKNOWN
    smoking_state: CabinSmokingState = CabinSmokingState.NO_SMOKING
    cabin_evidence_count: int = 0
    evidence_objects: list[CabinEvidenceObject] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    phone_reason_codes: list[str] = field(default_factory=list)
    seatbelt_reason_codes: list[str] = field(default_factory=list)
    smoking_reason_codes: list[str] = field(default_factory=list)


@dataclass
class DMSV02DecisionState:
    drowsiness_state: str = "UNKNOWN"
    distraction_state: str = "UNKNOWN"
    driver_availability_state: str = "UNCONFIRMED"
    dms_confidence_state: DMSConfidenceState = DMSConfidenceState.LOW
    final_level: DMSV02Level = DMSV02Level.DEGRADED
    final_banner: str = "DMS DEGRADED"
    final_decision_path: str = ""
    reason_codes: list[str] = field(default_factory=list)
    raw_observation_codes: list[str] = field(default_factory=list)
    classification_reason_codes: list[str] = field(default_factory=list)
    alert_subtype: str = ""
    alert_explanation: str = ""
    hmi_banner_text: str = ""
    hmi_primary_reason: str = ""
    hmi_secondary_reason: str = ""


@dataclass
class VehicleRuntimeState:
    ego_vehicle_speed_kph: float = 0.0
    ego_vehicle_speed_source: str = "SIMULATED"
    vehicle_speed_sim_enabled: bool = True
    dms_speed_gate_state: str = "STARTUP_INITIALIZING"
    dms_operational_mode: str = "STARTUP_INITIALIZING"
    dms_alerts_enabled: bool = False
    dms_alert_suppression_reason: str = "NONE"
    dms_activation_threshold_kph: float = 30.0
    dms_deactivation_threshold_kph: float = 28.0
    vehicle_speed_reason_codes: list[str] = field(default_factory=list)
    left_indicator_on: bool = False
    right_indicator_on: bool = False
    indicator_source: str = "SIMULATED"
    indicator_reason_codes: list[str] = field(default_factory=list)
    sanctioned_task_state: str = "NONE"
    sanctioned_task_reason_codes: list[str] = field(default_factory=list)
    hmi_banner_text: str = ""
    hmi_alert_subtype: str = ""
    hmi_primary_reason: str = ""
    hmi_secondary_reason: str = ""
    vehicle_monitor_line: str = ""
    hmi_operational_reason: str = ""
    time_to_activation_s: float = 0.0
    critical_driver_unavailable_requested: bool = False
    critical_unavailable_requires_no_face: bool = True
    critical_unavailable_requires_no_body: bool = True
    critical_unavailable_reason_codes: list[str] = field(default_factory=list)
    live_output_fps_mode: str = "measured"
    live_output_fps: float = 0.0
    frame_capture_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    frame_write_time_ms: float = 0.0
    timing_reason_codes: list[str] = field(default_factory=list)


@dataclass
class AttentionOutput:
    attention_state: AttentionState = AttentionState.UNKNOWN
    attention_substate: AttentionSubstate = AttentionSubstate.UNKNOWN
    attention_confidence: float = 0.0
    head_down_duration_ms: int = 0
    pose_based_head_down_duration_ms: int = 0
    appearance_based_head_down_duration_ms: int = 0
    head_down_uncertain_duration_ms: int = 0
    gaze_offroad_duration_ms: int = 0
    phone_down_candidate_duration_ms: int = 0
    phone_texting_candidate_duration_ms: int = 0
    visual_distraction_duration_ms: int = 0
    observation_degraded_duration_ms: int = 0
    eye_closed_duration_ms: int = 0
    attention_lost_duration_ms: int = 0
    side_profile_lost_duration_ms: int = 0
    side_glance_state: str = "ROAD_AXIS_NORMAL"
    side_glance_duration_ms: int = 0
    side_glance_recovery_ms: int = 0
    relative_yaw_deg: float = 0.0
    relative_pitch_deg: float = 0.0
    relative_roll_deg: float = 0.0
    yaw_classifiable: bool = False
    side_profile_context_active: bool = False
    microsleep_candidate: bool = False
    phone_suspicion_candidate: bool = False
    ambiguous_attention_loss: bool = False
    low_head_motion: bool = False
    pose_reliable: bool = True
    effective_attention_source: str = "UNKNOWN"
    attention_reason_codes: list[str] = field(default_factory=list)
    driver_availability_reason: str = ""
    final_decision_path: str = ""


@dataclass
class OccupantFace:
    track_id: int
    zone: str
    box_norm: list[float]
    selected_as_driver: bool = False


@dataclass
class OccupantsState:
    count: int = 0
    face_count: int = 0
    proposal_count: int = 0
    confirmed_face_count: int = 0
    unconfirmed_proposal_count: int = 0
    rejected_proposals: list[dict[str, Any]] = field(default_factory=list)
    driver_track_id: int | None = None
    driver_zone: str = "DRIVER"
    driver_body_present: bool = False
    faces: list[OccupantFace] = field(default_factory=list)


@dataclass
class OccupancySeatState:
    occupied: str = "unknown"
    occupant_type: str = "UNKNOWN"
    detection_source: str = "UNKNOWN"
    confidence: float = 0.0
    track_id: int | None = None
    stable_frames: int = 0
    occlusion_state: str = "UNKNOWN"
    face_visible: bool = False
    body_visible: bool = False
    bbox: list[float] = field(default_factory=list)
    depth_layer: str = "UNKNOWN"
    slot_reason: str = ""


@dataclass
class OccupancyState:
    cabin_occupant_count: int = 0
    face_count: int = 0
    body_count: int = 0
    driver_present: bool = False
    front_passenger_present: bool = False
    rear_left_present: str = "unknown"
    rear_center_present: str = "unknown"
    rear_right_present: str = "unknown"
    unknown_occupant_count: int = 0
    occupancy_confidence: float = 0.0
    occupancy_reason_codes: list[str] = field(default_factory=list)
    seats: dict[str, OccupancySeatState] = field(default_factory=dict)


@dataclass
class DriverIdentityState:
    driver_session_id: str | None = None
    driver_track_id: int | None = None
    session_state: str = "UNKNOWN"
    reassociated: bool = False
    time_since_seen_ms: int = 0
    driver_body_state: str = "UNKNOWN"
    driver_candidate_score: float = 0.0
    driver_front_layer_score: float = 0.0
    driver_rear_layer_penalty: float = 0.0
    driver_slot_assignment: str = "UNKNOWN"
    driver_slot_reason: str = ""
    candidate_depth_layer: str = "UNKNOWN"
    candidate_seat_slot: str = "UNKNOWN"
    rear_overlap_rejected_as_driver: bool = False
    driver_validation_state: str = "UNKNOWN"
    driver_validation_reasons: list[str] = field(default_factory=list)
    driver_proposal_confidence: float = 0.0
    driver_face_completeness_score: float = 0.0
    driver_landmark_coverage_score: float = 0.0
    driver_landmark_count: int = 0
    driver_partial_face: bool = False
    face_proposal_state: str = "NO_PROPOSAL"
    driver_face_state: str = "NOT_VISIBLE"
    driver_proposal_visible: bool = False
    driver_proposal_bbox_norm: list[float] = field(default_factory=list)
    driver_track_hold_state: str = "NONE"


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
    driver_observability: DriverObservability = field(default_factory=DriverObservability)
    driver_availability: DriverAvailability = field(default_factory=DriverAvailability)
    occupants: OccupantsState = field(default_factory=OccupantsState)
    driver_identity: DriverIdentityState = field(default_factory=DriverIdentityState)
    gaze: GazeState = field(default_factory=GazeState)
    drowsiness: DrowsinessState = field(default_factory=DrowsinessState)
    distraction: DistractionState = field(default_factory=DistractionState)
    phone_use: PhoneUseState = field(default_factory=PhoneUseState)
    attention: AttentionOutput = field(default_factory=AttentionOutput)
    dms_v02: DMSV02DecisionState = field(default_factory=DMSV02DecisionState)
    vehicle: VehicleRuntimeState = field(default_factory=VehicleRuntimeState)
    occupancy: OccupancyState = field(default_factory=OccupancyState)
    seatbelt_authenticity: SeatbeltAuthenticity = field(default_factory=SeatbeltAuthenticity)
    cabin_evidence: CabinEvidenceState = field(default_factory=CabinEvidenceState)
    driver_readiness_score: DriverReadinessScore = field(default_factory=DriverReadinessScore)

    def to_dict(self) -> dict[str, Any]:
        payload = _enum_to_value(asdict(self))
        attention = payload.get("attention", {})
        for key in (
            "attention_state",
            "attention_substate",
            "attention_confidence",
            "head_down_duration_ms",
            "pose_based_head_down_duration_ms",
            "appearance_based_head_down_duration_ms",
            "head_down_uncertain_duration_ms",
            "gaze_offroad_duration_ms",
            "phone_down_candidate_duration_ms",
            "phone_texting_candidate_duration_ms",
            "visual_distraction_duration_ms",
            "observation_degraded_duration_ms",
            "eye_closed_duration_ms",
            "attention_lost_duration_ms",
            "side_profile_lost_duration_ms",
            "side_glance_state",
            "side_glance_duration_ms",
            "side_glance_recovery_ms",
            "relative_yaw_deg",
            "relative_pitch_deg",
            "relative_roll_deg",
            "yaw_classifiable",
            "side_profile_context_active",
            "microsleep_candidate",
            "phone_suspicion_candidate",
            "ambiguous_attention_loss",
            "low_head_motion",
            "pose_reliable",
            "effective_attention_source",
            "attention_reason_codes",
            "driver_availability_reason",
            "final_decision_path",
        ):
            payload[key] = attention.get(key)
        dms_v02 = payload.get("dms_v02", {})
        payload["raw_observation_codes"] = dms_v02.get("raw_observation_codes", [])
        payload["classification_reason_codes"] = dms_v02.get("classification_reason_codes", [])
        return payload


def _enum_to_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _enum_to_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_enum_to_value(item) for item in value]
    return value


