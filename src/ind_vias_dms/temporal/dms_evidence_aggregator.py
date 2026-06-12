from __future__ import annotations

from dataclasses import dataclass, field

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import GazeZone


OFF_ROAD_ZONES = {GazeZone.LEFT, GazeZone.RIGHT, GazeZone.DOWN, GazeZone.UP, GazeZone.PHONE_DOWN}
DOWNWARD_ZONES = {GazeZone.DOWN, GazeZone.PHONE_DOWN}


@dataclass(frozen=True)
class DMSEvidenceInput:
    timestamp_ms: int
    face_present: bool
    face_confidence: float
    head_yaw_deg: float
    head_pitch_deg: float
    head_roll_deg: float
    pose_reliable: bool
    gaze_zone: GazeZone
    gaze_confidence: float
    raw_eye_state: str
    effective_eye_state: str
    eye_visibility: float
    phone_state: str
    phone_reason_codes: list[str] = field(default_factory=list)
    driver_body_present: bool = False
    previous_driver_state: str = "UNKNOWN"


@dataclass(frozen=True)
class DMSEvidence:
    head_down_candidate: bool
    gaze_offroad_candidate: bool
    phone_down_candidate: bool
    visual_distraction_candidate: bool
    drowsiness_candidate: bool
    observation_degraded_candidate: bool
    head_down_duration_ms: int
    pose_based_head_down_duration_ms: int
    appearance_based_head_down_duration_ms: int
    head_down_uncertain_duration_ms: int
    gaze_offroad_duration_ms: int
    phone_down_candidate_duration_ms: int
    phone_texting_candidate_duration_ms: int
    visual_distraction_duration_ms: int
    observation_degraded_duration_ms: int
    effective_attention_source: str
    reason_codes: list[str]


class DMSEvidenceAggregator:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self._last_timestamp_ms: int | None = None
        self._head_down_ms = 0
        self._pose_head_down_ms = 0
        self._appearance_head_down_ms = 0
        self._uncertain_head_down_ms = 0
        self._gaze_offroad_ms = 0
        self._phone_down_ms = 0
        self._phone_texting_ms = 0
        self._visual_distraction_ms = 0
        self._observation_degraded_ms = 0
        self._road_clear_ms = 0

    def reset(self) -> None:
        self._last_timestamp_ms = None
        self._head_down_ms = 0
        self._pose_head_down_ms = 0
        self._appearance_head_down_ms = 0
        self._uncertain_head_down_ms = 0
        self._gaze_offroad_ms = 0
        self._phone_down_ms = 0
        self._phone_texting_ms = 0
        self._visual_distraction_ms = 0
        self._observation_degraded_ms = 0
        self._road_clear_ms = 0

    def update(self, signals: DMSEvidenceInput) -> DMSEvidence:
        dt_ms = self._delta(signals.timestamp_ms)
        phone_reasons = set(signals.phone_reason_codes)
        phone_state_candidate = signals.phone_state in {
            "PHONE_SUSPECTED",
            "PHONE_CONFIRMED",
            "PHONE_TO_EAR_SUSPECTED",
            "PHONE_TO_EAR_CONFIRMED",
            "PHONE_DOWN_SUSPECTED",
            "PHONE_DOWN_CONFIRMED",
            "PHONE_TEXTING_SCROLLING_SUSPECTED",
            "PHONE_TEXTING_SCROLLING_CONFIRMED",
            "TEXTING_SUSPECTED",
            "HAND_NEAR_FACE",
        }
        pose_head_down = signals.pose_reliable and signals.head_pitch_deg >= self.config.head_pitch_down_threshold_deg
        gaze_offroad = signals.gaze_zone in OFF_ROAD_ZONES
        gaze_down_or_unknown = signals.gaze_zone in DOWNWARD_ZONES or signals.gaze_zone == GazeZone.UNKNOWN
        posture_reason = bool(
            phone_reasons
            & {
                "POSSIBLE_PHONE_POSTURE",
                "PHONE_DOWN_SUSPECTED",
                "HEAD_DOWN",
                "GAZE_OFF_ROAD",
                "GAZE_UNKNOWN",
            }
        )
        appearance_head_down = (
            self.config.enable_appearance_based_head_down
            and signals.face_present
            and (signals.gaze_zone in DOWNWARD_ZONES or posture_reason or phone_state_candidate)
        )
        uncertain_head_down = (
            signals.face_present
            and not signals.pose_reliable
            and (
                posture_reason
                or signals.head_pitch_deg >= self.config.head_pitch_down_threshold_deg
                or gaze_offroad
                or signals.eye_visibility < self.config.eye_visibility_min_confidence
            )
        )
        head_down = pose_head_down or appearance_head_down or uncertain_head_down
        road_clear = (
            signals.pose_reliable
            and signals.gaze_zone == GazeZone.ROAD
            and signals.eye_visibility >= self.config.eye_visibility_min_confidence
            and not posture_reason
            and not phone_state_candidate
        )

        self._road_clear_ms = self._step(self._road_clear_ms, road_clear, dt_ms)
        clear_confirmed = self._road_clear_ms >= self.config.head_down_clear_ms
        self._pose_head_down_ms = self._accumulate_or_decay(self._pose_head_down_ms, pose_head_down, dt_ms, clear_confirmed)
        self._appearance_head_down_ms = self._accumulate_or_decay(
            self._appearance_head_down_ms,
            appearance_head_down,
            dt_ms,
            clear_confirmed,
        )
        self._uncertain_head_down_ms = self._accumulate_or_decay(
            self._uncertain_head_down_ms,
            uncertain_head_down,
            dt_ms,
            clear_confirmed,
        )
        self._head_down_ms = max(self._pose_head_down_ms, self._appearance_head_down_ms, self._uncertain_head_down_ms)
        self._gaze_offroad_ms = self._accumulate_or_decay(self._gaze_offroad_ms, gaze_offroad, dt_ms, clear_confirmed)

        phone_down = (
            signals.face_present
            and gaze_down_or_unknown
            and self._head_down_ms >= self.config.phone_down_candidate_ms
            and (
                signals.eye_visibility >= self.config.eye_visibility_min_confidence
                or posture_reason
                or phone_state_candidate
            )
            and signals.effective_eye_state != "CLOSED"
        ) or signals.phone_state == "PHONE_DOWN_SUSPECTED"
        phone_texting = (
            phone_down
            and (posture_reason or phone_state_candidate)
            and self._head_down_ms >= self.config.phone_down_candidate_ms
        ) or signals.phone_state in {"TEXTING_SUSPECTED", "PHONE_TEXTING_SCROLLING_SUSPECTED"}
        visual_distraction = self._head_down_ms >= self.config.head_down_candidate_ms or self._gaze_offroad_ms >= self.config.gaze_away_low_ms
        observation_degraded = (
            not signals.pose_reliable
            or signals.eye_visibility < self.config.eye_visibility_min_confidence
            or signals.gaze_confidence < 0.35
        )
        drowsiness_candidate = signals.effective_eye_state == "CLOSED"

        self._phone_down_ms = self._accumulate_or_decay(self._phone_down_ms, phone_down, dt_ms, clear_confirmed)
        self._phone_texting_ms = self._accumulate_or_decay(
            self._phone_texting_ms,
            phone_texting,
            dt_ms,
            clear_confirmed,
        )
        self._visual_distraction_ms = self._accumulate_or_decay(
            self._visual_distraction_ms,
            visual_distraction,
            dt_ms,
            clear_confirmed,
        )
        self._observation_degraded_ms = self._step(self._observation_degraded_ms, observation_degraded, dt_ms)

        reasons: list[str] = []
        if self._head_down_ms >= self.config.head_down_candidate_ms:
            reasons.append("HEAD_DOWN")
        if self._gaze_offroad_ms >= self.config.gaze_away_low_ms:
            reasons.append("GAZE_OFF_ROAD")
        if self._phone_down_ms >= self.config.phone_down_candidate_ms or phone_state_candidate:
            reasons.append("PHONE_DOWN_SUSPECTED" if phone_down else "POSSIBLE_PHONE_POSTURE")
        if self._phone_texting_ms >= self.config.phone_texting_warning_ms:
            reasons.append("PHONE_TEXTING_SCROLLING_SUSPECTED")
        if observation_degraded:
            if not signals.pose_reliable:
                reasons.append("HEAD_POSE_UNRELIABLE")
            if signals.eye_visibility < self.config.eye_visibility_min_confidence:
                reasons.append("LOW_EYE_VISIBILITY")

        source = "UNKNOWN"
        if self._phone_down_ms >= self.config.phone_down_candidate_ms:
            source = "PHONE"
        elif self._pose_head_down_ms >= self.config.head_down_candidate_ms:
            source = "POSE"
        elif self._gaze_offroad_ms >= self.config.gaze_away_low_ms:
            source = "GAZE"
        elif self._appearance_head_down_ms >= self.config.head_down_candidate_ms:
            source = "APPEARANCE"

        return DMSEvidence(
            head_down_candidate=self._head_down_ms >= self.config.head_down_candidate_ms,
            gaze_offroad_candidate=self._gaze_offroad_ms >= self.config.gaze_away_low_ms,
            phone_down_candidate=self._phone_down_ms >= self.config.phone_down_candidate_ms,
            visual_distraction_candidate=self._visual_distraction_ms >= self.config.head_down_candidate_ms,
            drowsiness_candidate=drowsiness_candidate,
            observation_degraded_candidate=observation_degraded,
            head_down_duration_ms=self._head_down_ms,
            pose_based_head_down_duration_ms=self._pose_head_down_ms,
            appearance_based_head_down_duration_ms=self._appearance_head_down_ms,
            head_down_uncertain_duration_ms=self._uncertain_head_down_ms,
            gaze_offroad_duration_ms=self._gaze_offroad_ms,
            phone_down_candidate_duration_ms=self._phone_down_ms,
            phone_texting_candidate_duration_ms=self._phone_texting_ms,
            visual_distraction_duration_ms=self._visual_distraction_ms,
            observation_degraded_duration_ms=self._observation_degraded_ms,
            effective_attention_source=source,
            reason_codes=list(dict.fromkeys(reasons)),
        )

    def _delta(self, timestamp_ms: int) -> int:
        if self._last_timestamp_ms is None:
            self._last_timestamp_ms = timestamp_ms
            return 0
        dt_ms = max(0, timestamp_ms - self._last_timestamp_ms)
        self._last_timestamp_ms = timestamp_ms
        return dt_ms

    @staticmethod
    def _step(current_ms: int, active: bool, dt_ms: int) -> int:
        return current_ms + dt_ms if active else 0

    def _accumulate_or_decay(
        self,
        current_ms: int,
        active: bool,
        dt_ms: int,
        clear_confirmed: bool,
    ) -> int:
        if active:
            return current_ms + dt_ms
        if clear_confirmed:
            return 0
        return max(0, current_ms - max(self.config.head_down_decay_ms_per_frame, dt_ms))
