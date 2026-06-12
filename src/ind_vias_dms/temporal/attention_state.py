from __future__ import annotations

from dataclasses import dataclass

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import (
    AttentionOutput,
    AttentionState,
    AttentionSubstate,
    DistractionLevel,
    DrowsinessLevel,
    GazeZone,
)
from ind_vias_dms.temporal.dms_evidence_aggregator import (
    DMSEvidence,
    DMSEvidenceAggregator,
    DMSEvidenceInput,
)


OFF_ROAD_ZONES = {GazeZone.LEFT, GazeZone.RIGHT, GazeZone.DOWN, GazeZone.UP, GazeZone.PHONE_DOWN}
DOWNWARD_ZONES = {GazeZone.DOWN, GazeZone.PHONE_DOWN}


@dataclass(frozen=True)
class AttentionSignals:
    timestamp_ms: int
    driver_face_present: bool
    driver_body_present: bool
    session_state: str
    gaze_zone: GazeZone
    gaze_confidence: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    eye_state: str
    eye_visibility: float
    eye_closure_duration_ms: int
    perclos_5s: float
    perclos_60s: float
    phone_state: str
    distraction_level: DistractionLevel
    drowsiness_level: DrowsinessLevel
    head_pose_unreliable: bool = False
    phone_reason_codes: list[str] | None = None
    relative_yaw_deg: float = 0.0
    relative_pitch_deg: float = 0.0
    relative_roll_deg: float = 0.0
    side_glance_state: str = "ROAD_AXIS_NORMAL"
    side_glance_duration_ms: int = 0
    side_glance_recovery_ms: int = 0
    yaw_classifiable: bool = False
    side_profile_context_active: bool = False


class AttentionStateClassifier:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        attention = config.attention_state or {}
        self.enabled = bool(attention.get("enabled", True))
        self.head_down_pitch_deg = float(attention.get("head_down_pitch_deg", 18.0))
        self.gaze_down_pitch_deg = float(attention.get("gaze_down_pitch_deg", 15.0))
        self.eyes_offroad_min_ms = int(attention.get("eyes_offroad_min_ms", 1500))
        self.phone_suspect_min_ms = int(attention.get("phone_suspect_min_ms", 2000))
        self.phone_attention_lost_ms = int(attention.get("phone_attention_lost_ms", config.phone_attention_lost_ms))
        self.microsleep_eye_closed_ms = int(attention.get("microsleep_eye_closed_ms", 1500))
        self.drowsy_eye_closed_ms = int(attention.get("drowsy_eye_closed_ms", 800))
        self.perclos_high_threshold = float(attention.get("perclos_high_threshold", 0.40))
        self.perclos_medium_threshold = float(attention.get("perclos_medium_threshold", 0.25))
        self.low_head_motion_deg_per_s = float(attention.get("low_head_motion_deg_per_s", 3.0))
        self.ambiguous_timeout_ms = int(attention.get("ambiguous_timeout_ms", 1000))
        self.side_profile_hold_ms = int(attention.get("side_profile_hold_ms", 1200))
        self.head_down_uncertain_sustain_ms = int(
            attention.get("head_down_uncertain_sustain_ms", config.head_down_uncertain_sustain_ms)
        )
        self.head_down_warning_ms = int(attention.get("head_down_warning_ms", config.head_down_warning_ms))
        self.head_down_attention_lost_ms = int(
            attention.get("head_down_attention_lost_ms", config.head_down_attention_lost_ms)
        )
        self.require_eye_visibility_for_drowsiness = bool(
            attention.get("require_eye_visibility_for_drowsiness", True)
        )
        self.prefer_drowsiness_when_eyes_closed = bool(attention.get("prefer_drowsiness_when_eyes_closed", True))
        self.prefer_distraction_when_eyes_open = bool(attention.get("prefer_distraction_when_eyes_open", True))
        self.suppress_drowsiness_when_phone_likely = bool(
            attention.get("suppress_drowsiness_when_phone_likely", True)
        )
        self.degrade_on_sustained_ambiguous_attention_loss = bool(
            attention.get("degrade_on_sustained_ambiguous_attention_loss", True)
        )
        self._head_down_since_ms: int | None = None
        self._pose_head_down_since_ms: int | None = None
        self._appearance_head_down_since_ms: int | None = None
        self._uncertain_head_down_since_ms: int | None = None
        self._gaze_offroad_since_ms: int | None = None
        self._attention_lost_since_ms: int | None = None
        self._side_profile_lost_since_ms: int | None = None
        self._last_timestamp_ms: int | None = None
        self._last_yaw_deg: float | None = None
        self._last_pitch_deg: float | None = None
        self.evidence_aggregator = DMSEvidenceAggregator(config)

    def reset(self) -> None:
        self._head_down_since_ms = None
        self._pose_head_down_since_ms = None
        self._appearance_head_down_since_ms = None
        self._uncertain_head_down_since_ms = None
        self._gaze_offroad_since_ms = None
        self._attention_lost_since_ms = None
        self._side_profile_lost_since_ms = None
        self._last_timestamp_ms = None
        self._last_yaw_deg = None
        self._last_pitch_deg = None
        self.evidence_aggregator.reset()

    def update(self, signals: AttentionSignals) -> AttentionOutput:
        if not self.enabled:
            return AttentionOutput(
                attention_state=AttentionState.UNKNOWN,
                attention_substate=AttentionSubstate.UNKNOWN,
                attention_reason_codes=["ATTENTION_STATE_DISABLED"],
            )

        pose_reliable = not signals.head_pose_unreliable
        evidence = self.evidence_aggregator.update(
            DMSEvidenceInput(
                timestamp_ms=signals.timestamp_ms,
                face_present=signals.driver_face_present,
                face_confidence=1.0 if signals.driver_face_present else 0.0,
                head_yaw_deg=signals.yaw_deg,
                head_pitch_deg=signals.pitch_deg,
                head_roll_deg=signals.roll_deg,
                pose_reliable=pose_reliable,
                gaze_zone=signals.gaze_zone,
                gaze_confidence=signals.gaze_confidence,
                raw_eye_state=signals.eye_state,
                effective_eye_state=signals.eye_state,
                eye_visibility=signals.eye_visibility,
                phone_state=signals.phone_state,
                phone_reason_codes=signals.phone_reason_codes or [],
                driver_body_present=signals.driver_body_present,
                previous_driver_state=signals.session_state,
            )
        )
        pose_head_down = pose_reliable and signals.pitch_deg >= self.head_down_pitch_deg
        pose_gaze_down = pose_reliable and signals.pitch_deg >= self.gaze_down_pitch_deg
        gaze_down = pose_gaze_down or signals.gaze_zone in DOWNWARD_ZONES
        gaze_offroad = signals.gaze_zone in OFF_ROAD_ZONES
        eye_valid = signals.eye_visibility >= self.config.eye_visibility_min_confidence and signals.eye_state != "UNKNOWN"
        eyes_closed = signals.eye_state in {"CLOSED", "PARTIALLY_CLOSED"}
        eyes_openish = signals.eye_state == "OPEN" or (
            signals.eye_state == "PARTIALLY_CLOSED"
            and signals.eye_closure_duration_ms < self.config.blink_max_duration_ms
        )
        phone_evidence = signals.phone_state in {
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
        appearance_head_down = (
            self.config.enable_appearance_based_head_down
            and signals.driver_face_present
            and (signals.gaze_zone in DOWNWARD_ZONES or phone_evidence)
        )
        uncertain_head_down = (
            not pose_reliable
            and signals.driver_face_present
            and (
                signals.pitch_deg >= self.gaze_down_pitch_deg
                or signals.gaze_zone in OFF_ROAD_ZONES
                or phone_evidence
                or signals.eye_visibility < self.config.eye_visibility_min_confidence
            )
        )
        head_down = pose_head_down or appearance_head_down or uncertain_head_down or evidence.head_down_candidate
        side_profile_lost = (
            not signals.driver_face_present
            and signals.driver_body_present
            and signals.session_state == "LOST_TEMP"
        )
        side_profile_attention_available = (
            self.config.side_profile_attention_first_enabled
            and signals.side_profile_context_active
            and signals.driver_body_present
        )
        side_glance_monitor = (
            signals.yaw_classifiable
            and signals.side_glance_duration_ms >= self.config.side_glance_monitor_ms
            and signals.side_glance_state in {"SIDE_GLANCE_LEFT", "SIDE_GLANCE_RIGHT", "SIDE_PROFILE_ATTENTION_LOSS"}
        )
        side_glance_warning = (
            signals.yaw_classifiable
            and abs(signals.relative_yaw_deg) >= self.config.side_glance_warning_deg
            and signals.side_glance_duration_ms >= self.config.side_glance_warning_ms
        )

        pose_head_down_ms = evidence.pose_based_head_down_duration_ms
        appearance_head_down_ms = evidence.appearance_based_head_down_duration_ms
        uncertain_head_down_ms = evidence.head_down_uncertain_duration_ms
        head_down_ms = evidence.head_down_duration_ms
        self._duration(signals.timestamp_ms, "_head_down_since_ms", head_down)
        gaze_offroad_ms = evidence.gaze_offroad_duration_ms
        side_profile_ms = self._duration(signals.timestamp_ms, "_side_profile_lost_since_ms", side_profile_lost)
        low_head_motion = self._low_head_motion(signals)
        reasons: list[str] = []

        reasons.extend(evidence.reason_codes)
        if head_down and "HEAD_DOWN" not in reasons:
            reasons.append("HEAD_DOWN")
        if gaze_offroad and "GAZE_OFF_ROAD" not in reasons:
            reasons.append("GAZE_OFF_ROAD")
        if phone_evidence:
            reasons.append("POSSIBLE_PHONE_POSTURE")
            reasons.append("PHONE_OR_LAP_SUSPECTED")
        if signals.head_pose_unreliable:
            reasons.append("HEAD_POSE_UNRELIABLE")
        if uncertain_head_down:
            reasons.append("HEAD_DOWN_UNCERTAIN")
        if signals.yaw_classifiable:
            reasons.append("ROAD_AXIS_HEADPOSE_REFERENCE_ACTIVE")
        if side_glance_monitor:
            reasons.extend(["RELATIVE_YAW_SIDE_GLANCE", "SIDE_GLANCE_MONITOR"])
        if side_glance_warning:
            reasons.extend(["RELATIVE_YAW_SIDE_GLANCE_SUSTAINED", "SIDE_GLANCE_DISTRACTION_WARNING"])

        microsleep_candidate = (
            eye_valid
            and signals.eye_state == "CLOSED"
            and signals.eye_closure_duration_ms >= self.microsleep_eye_closed_ms
        )
        elevated_perclos = (
            signals.perclos_5s >= self.perclos_medium_threshold
            or signals.perclos_60s >= self.perclos_medium_threshold
        )
        high_perclos = (
            signals.perclos_5s >= self.perclos_high_threshold
            or signals.perclos_60s >= self.perclos_high_threshold
        )
        phone_posture_eye_supported = (
            signals.eye_state in {"OPEN", "PARTIALLY_CLOSED", "UNKNOWN"}
            and signals.eye_visibility >= self.config.eye_visibility_min_confidence
        )
        phone_suspicion_candidate = evidence.phone_down_candidate or phone_evidence or (
            phone_posture_eye_supported
            and gaze_down
            and gaze_offroad_ms >= self.phone_suspect_min_ms
            and not microsleep_candidate
        ) or (
            (
                phone_posture_eye_supported
                or (signals.eye_state == "UNKNOWN" and phone_evidence)
            )
            and head_down_ms >= self.config.phone_down_suspect_ms
            and not microsleep_candidate
        )
        drowsy_candidate = (
            eye_valid
            and eyes_closed
            and signals.eye_closure_duration_ms >= self.drowsy_eye_closed_ms
            and elevated_perclos
            and not (self.suppress_drowsiness_when_phone_likely and phone_suspicion_candidate)
        )
        ambiguous_candidate = (
            (head_down or gaze_offroad or side_profile_lost)
            and not phone_suspicion_candidate
            and not drowsy_candidate
            and not microsleep_candidate
            and (not eye_valid or signals.head_pose_unreliable or side_profile_lost)
        )
        attention_lost_candidate = (
            microsleep_candidate
            or drowsy_candidate
            or phone_suspicion_candidate
            or ambiguous_candidate
            or evidence.visual_distraction_duration_ms >= self.eyes_offroad_min_ms
            or head_down_ms >= self.head_down_warning_ms
        )
        attention_lost_ms = self._duration(
            signals.timestamp_ms,
            "_attention_lost_since_ms",
            attention_lost_candidate,
        )

        state = AttentionState.NORMAL
        substate = AttentionSubstate.ROAD
        confidence = max(0.2, min(1.0, signals.gaze_confidence))
        availability_reason = ""
        effective_source = "UNKNOWN"
        final_path = "NORMAL > ROAD"

        if side_profile_lost and side_profile_attention_available:
            state = AttentionState.ATTENTION_LOST if signals.side_glance_duration_ms >= self.config.side_glance_warning_ms else AttentionState.DEGRADED
            substate = (
                AttentionSubstate.SIDE_PROFILE_ATTENTION_LOSS
                if signals.side_glance_duration_ms >= self.config.side_glance_warning_ms
                else AttentionSubstate.SIDE_PROFILE_TRACKED
            )
            confidence = 0.6
            availability_reason = "SIDE_PROFILE_ATTENTION_LOSS" if state == AttentionState.ATTENTION_LOST else "SIDE_PROFILE_TRACKED"
            effective_source = "APPEARANCE"
            reasons.extend([
                "SIDE_PROFILE_ATTENTION_FIRST",
                "SIDE_PROFILE_CLASSIFIED_AS_ATTENTION_NOT_FACE_LOSS",
                "FACE_LOSS_SUPPRESSED_BY_SIDE_PROFILE_TRACK",
                "DRIVER_TRACK_PRESERVED_SIDE_PROFILE",
                "POSSIBLE_GAZE_AWAY_DURING_LOST_TEMP",
            ])
            final_path = (
                f"{'DISTRACTION_WARNING' if state == AttentionState.ATTENTION_LOST else 'DMS_MONITOR'} "
                f"> {substate.value} > side_yaw_ms={signals.side_glance_duration_ms}"
            )
        elif side_profile_lost:
            state = AttentionState.DEGRADED
            substate = AttentionSubstate.FACE_LOST
            confidence = 0.45
            effective_source = "APPEARANCE"
            reasons.extend([
                "SIDE_PROFILE_FACE_LOST",
                "DRIVER_BODY_PRESENT_FACE_LOST",
                "POSSIBLE_GAZE_AWAY_DURING_LOST_TEMP",
            ])
            final_path = "DMS_DEGRADED > FACE_LOST"
        elif side_glance_warning:
            state = AttentionState.ATTENTION_LOST
            substate = AttentionSubstate.SIDE_PROFILE_ATTENTION_LOSS
            confidence = max(confidence, 0.7)
            availability_reason = "SIDE_PROFILE_ATTENTION_LOSS"
            effective_source = "POSE"
            reasons.extend([
                "SIDE_PROFILE_ATTENTION_FIRST",
                "SIDE_PROFILE_POSE_VALID",
                "SIDE_PROFILE_NOT_FACE_LOST",
                "SIDE_GLANCE_DISTRACTION_WARNING",
            ])
            final_path = f"DISTRACTION_WARNING > SIDE_PROFILE_ATTENTION_LOSS > relative_yaw={signals.relative_yaw_deg:.1f}"
        elif side_glance_monitor:
            state = AttentionState.DEGRADED
            substate = (
                AttentionSubstate.SIDE_GLANCE_RIGHT
                if signals.relative_yaw_deg > 0.0
                else AttentionSubstate.SIDE_GLANCE_LEFT
            )
            confidence = max(confidence, 0.6)
            availability_reason = "SIDE_GLANCE_MONITOR"
            effective_source = "POSE"
            reasons.extend([
                "SIDE_PROFILE_ATTENTION_FIRST",
                "SIDE_PROFILE_TRACKED",
                "SIDE_PROFILE_NOT_FACE_LOST",
            ])
            final_path = f"DMS_MONITOR > {substate.value} > relative_yaw={signals.relative_yaw_deg:.1f}"
        elif microsleep_candidate:
            state = AttentionState.ATTENTION_LOST
            substate = AttentionSubstate.MICROSLEEP
            confidence = max(confidence, signals.eye_visibility)
            availability_reason = "MICROSLEEP_CANDIDATE"
            effective_source = "POSE" if pose_reliable else "APPEARANCE"
            reasons.extend(["SUSTAINED_EYE_CLOSURE", "MICROSLEEP_CANDIDATE"])
            if high_perclos:
                reasons.append("PERCLOS_HIGH")
            if head_down:
                reasons.append("HEAD_DROP")
            if low_head_motion:
                reasons.append("LOW_HEAD_MOTION")
            final_path = "DRIVER_UNAVAILABLE > MICROSLEEP_CANDIDATE"
        elif drowsy_candidate:
            state = AttentionState.ATTENTION_LOST
            substate = AttentionSubstate.DROWSY
            confidence = max(confidence, min(1.0, signals.eye_visibility + 0.1))
            availability_reason = "DROWSY_ATTENTION_LOSS"
            effective_source = "POSE" if pose_reliable else "APPEARANCE"
            reasons.extend(["SUSTAINED_EYE_CLOSURE", "PERCLOS_HIGH" if high_perclos else "PERCLOS_MEDIUM"])
            final_path = "DROWSINESS_WARNING > DROWSY"
        elif phone_suspicion_candidate:
            state = (
                AttentionState.ATTENTION_LOST
                if gaze_offroad_ms >= min(self.eyes_offroad_min_ms, self.phone_attention_lost_ms)
                or head_down_ms >= self.phone_attention_lost_ms
                else AttentionState.DEGRADED
            )
            if signals.phone_state == "PHONE_CONFIRMED":
                substate = AttentionSubstate.PHONE_CONFIRMED
            elif signals.phone_state in {"PHONE_TEXTING_SCROLLING_CONFIRMED", "PHONE_DOWN_CONFIRMED"}:
                substate = AttentionSubstate.PHONE_TEXTING_SCROLLING_CONFIRMED
            elif signals.phone_state == "PHONE_DOWN_SUSPECTED":
                substate = AttentionSubstate.PHONE_DOWN_SUSPECTED
            elif (
                signals.phone_state in {"PHONE_TEXTING_SCROLLING_SUSPECTED", "TEXTING_SUSPECTED"}
                or evidence.phone_texting_candidate_duration_ms >= self.config.phone_texting_warning_ms
            ):
                substate = AttentionSubstate.PHONE_TEXTING_SCROLLING_SUSPECTED
            elif signals.phone_state == "PHONE_TO_EAR_SUSPECTED":
                substate = AttentionSubstate.PHONE_TO_EAR_SUSPECTED
            elif signals.phone_state == "TEXTING_SUSPECTED":
                substate = AttentionSubstate.TEXTING_SUSPECTED
            elif state == AttentionState.DEGRADED:
                substate = AttentionSubstate.PHONE_DOWN_CANDIDATE
            else:
                substate = AttentionSubstate.PHONE_SUSPECTED
            confidence = 0.75 if phone_evidence else 0.6
            availability_reason = "PHONE_ATTENTION_LOSS"
            effective_source = "PHONE"
            if eyes_openish:
                reasons.append("EYES_OPEN_OR_INTERMITTENT")
            if evidence.phone_down_candidate_duration_ms > 0:
                reasons.append("POSSIBLE_PHONE_POSTURE_ACCUMULATING")
            if evidence.phone_down_candidate_duration_ms >= self.config.phone_down_warning_ms:
                reasons.extend(["PHONE_DOWN_POSTURE_SUSTAINED", "PHONE_WARNING_FROM_POSTURE"])
            if evidence.phone_texting_candidate_duration_ms >= self.config.phone_texting_warning_ms:
                reasons.extend(["PHONE_TEXTING_SCROLLING_SUSPECTED", "PHONE_OBJECT_NOT_REQUIRED_POSTURE_BASED"])
            final_path = (
                f"DISTRACTION_WARNING > {substate.value} > "
                f"phone_down_ms={evidence.phone_down_candidate_duration_ms}"
            )
        elif ambiguous_candidate:
            state = (
                AttentionState.DEGRADED
                if attention_lost_ms < self.ambiguous_timeout_ms
                else AttentionState.ATTENTION_LOST
            )
            substate = (
                AttentionSubstate.HEAD_DOWN_UNCERTAIN
                if uncertain_head_down and attention_lost_ms < self.ambiguous_timeout_ms
                else AttentionSubstate.AMBIGUOUS
            )
            confidence = 0.45
            availability_reason = "AMBIGUOUS_ATTENTION_LOSS"
            effective_source = "APPEARANCE" if uncertain_head_down or side_profile_lost else "UNKNOWN"
            reasons.extend(["INSUFFICIENT_EVIDENCE", "AMBIGUOUS_ATTENTION_LOSS"])
            if not eye_valid:
                reasons.append("LOW_EYE_VISIBILITY")
            final_path = f"DMS_DEGRADED > {substate.value}"
        elif gaze_offroad_ms >= self.eyes_offroad_min_ms:
            state = AttentionState.ATTENTION_LOST
            substate = AttentionSubstate.VISUAL_DISTRACTION
            availability_reason = "VISUAL_ATTENTION_LOSS"
            effective_source = "GAZE"
            if eyes_openish:
                reasons.append("EYES_OPEN_OR_INTERMITTENT")
            final_path = f"DISTRACTION_WARNING > VISUAL_DISTRACTION > gaze_offroad_ms={gaze_offroad_ms}"
        elif uncertain_head_down and uncertain_head_down_ms >= self.head_down_uncertain_sustain_ms:
            state = AttentionState.DEGRADED
            substate = AttentionSubstate.HEAD_DOWN_UNCERTAIN
            confidence = 0.45
            availability_reason = "HEAD_DOWN_UNCERTAIN"
            effective_source = "APPEARANCE"
            final_path = f"DMS_DEGRADED > HEAD_DOWN_UNCERTAIN > head_down_ms={head_down_ms}"
        elif signals.head_pose_unreliable:
            state = AttentionState.DEGRADED
            substate = AttentionSubstate.HEAD_POSE_UNRELIABLE
            confidence = 0.35
            availability_reason = "HEAD_POSE_UNRELIABLE"
            effective_source = "UNKNOWN"
            final_path = "DMS_DEGRADED > HEAD_POSE_UNRELIABLE"
        elif self.config.head_down_candidate_ms <= head_down_ms < self.head_down_warning_ms:
            state = AttentionState.DEGRADED
            substate = AttentionSubstate.HEAD_DOWN_CANDIDATE
            confidence = 0.5
            availability_reason = "HEAD_DOWN_CANDIDATE"
            effective_source = evidence.effective_attention_source
            final_path = f"DMS_DEGRADED > HEAD_DOWN_CANDIDATE > head_down_ms={head_down_ms}"
        elif head_down_ms >= self.head_down_attention_lost_ms:
            state = AttentionState.ATTENTION_LOST
            substate = AttentionSubstate.HEAD_DOWN_DISTRACTION
            confidence = 0.65
            availability_reason = "HEAD_DOWN_ATTENTION_LOST"
            effective_source = "POSE" if pose_head_down_ms >= self.head_down_attention_lost_ms else "APPEARANCE"
            final_path = f"DISTRACTION_WARNING > HEAD_DOWN_DISTRACTION > head_down_ms={head_down_ms}"
        elif head_down_ms >= self.head_down_warning_ms:
            state = AttentionState.DEGRADED
            substate = AttentionSubstate.HEAD_DOWN_CANDIDATE
            confidence = 0.55
            availability_reason = "HEAD_DOWN"
            effective_source = "POSE" if pose_head_down_ms >= self.head_down_warning_ms else "APPEARANCE"
            final_path = f"DMS_DEGRADED > HEAD_DOWN_CANDIDATE > head_down_ms={head_down_ms}"

        if self.require_eye_visibility_for_drowsiness and not eye_valid and eyes_closed:
            reasons.append("LOW_EYE_VISIBILITY")
        if low_head_motion:
            reasons.append("LOW_HEAD_MOTION")

        return AttentionOutput(
            attention_state=state,
            attention_substate=substate,
            attention_confidence=confidence if state != AttentionState.NORMAL else 0.95,
            head_down_duration_ms=head_down_ms,
            pose_based_head_down_duration_ms=pose_head_down_ms,
            appearance_based_head_down_duration_ms=appearance_head_down_ms,
            head_down_uncertain_duration_ms=uncertain_head_down_ms,
            gaze_offroad_duration_ms=gaze_offroad_ms,
            phone_down_candidate_duration_ms=evidence.phone_down_candidate_duration_ms,
            phone_texting_candidate_duration_ms=evidence.phone_texting_candidate_duration_ms,
            visual_distraction_duration_ms=evidence.visual_distraction_duration_ms,
            observation_degraded_duration_ms=evidence.observation_degraded_duration_ms,
            eye_closed_duration_ms=signals.eye_closure_duration_ms,
            attention_lost_duration_ms=attention_lost_ms,
            side_profile_lost_duration_ms=side_profile_ms,
            side_glance_state=signals.side_glance_state,
            side_glance_duration_ms=signals.side_glance_duration_ms,
            side_glance_recovery_ms=signals.side_glance_recovery_ms,
            relative_yaw_deg=signals.relative_yaw_deg,
            relative_pitch_deg=signals.relative_pitch_deg,
            relative_roll_deg=signals.relative_roll_deg,
            yaw_classifiable=signals.yaw_classifiable,
            side_profile_context_active=signals.side_profile_context_active,
            microsleep_candidate=microsleep_candidate,
            phone_suspicion_candidate=phone_suspicion_candidate,
            ambiguous_attention_loss=ambiguous_candidate,
            low_head_motion=low_head_motion,
            pose_reliable=pose_reliable,
            effective_attention_source=effective_source if effective_source != "UNKNOWN" else evidence.effective_attention_source,
            attention_reason_codes=list(dict.fromkeys(reasons)),
            driver_availability_reason=availability_reason,
            final_decision_path=final_path,
        )

    def _duration(self, timestamp_ms: int, attr: str, active: bool) -> int:
        since = getattr(self, attr)
        if active:
            if since is None:
                setattr(self, attr, timestamp_ms)
                return 0
            return max(0, timestamp_ms - since)
        setattr(self, attr, None)
        return 0

    def _low_head_motion(self, signals: AttentionSignals) -> bool:
        if self._last_timestamp_ms is None or self._last_yaw_deg is None or self._last_pitch_deg is None:
            self._last_timestamp_ms = signals.timestamp_ms
            self._last_yaw_deg = signals.yaw_deg
            self._last_pitch_deg = signals.pitch_deg
            return False
        dt_ms = max(1, signals.timestamp_ms - self._last_timestamp_ms)
        motion = abs(signals.yaw_deg - self._last_yaw_deg) + abs(signals.pitch_deg - self._last_pitch_deg)
        deg_per_s = motion * 1000.0 / dt_ms
        self._last_timestamp_ms = signals.timestamp_ms
        self._last_yaw_deg = signals.yaw_deg
        self._last_pitch_deg = signals.pitch_deg
        return deg_per_s <= self.low_head_motion_deg_per_s
