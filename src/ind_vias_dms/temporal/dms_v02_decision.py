from __future__ import annotations

from dataclasses import dataclass

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import (
    AttentionOutput,
    AttentionSubstate,
    AvailabilityState,
    DMSConfidenceState,
    DMSHealth,
    DMSV02DecisionState,
    DMSV02Level,
    DistractionLevel,
    DistractionType,
    DriverAvailability,
    DrowsinessLevel,
    DrowsinessState,
)


@dataclass(frozen=True)
class DMSV02Inputs:
    timestamp_ms: int
    health: DMSHealth
    availability: DriverAvailability
    drowsiness: DrowsinessState
    distraction_level: DistractionLevel
    distraction_type: DistractionType
    attention: AttentionOutput
    phone_state: str
    driver_present: bool
    driver_body_present: bool
    no_face_duration_ms: int
    driver_observability: str = "UNKNOWN"
    driver_proposal_visible: bool = False
    driver_track_held: bool = False


class DMSV02DecisionMatrix:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self._last_banner: str | None = None
        self._last_level: DMSV02Level | None = None
        self._last_path: str = ""
        self._last_change_ms: int | None = None
        self._degraded_candidate_since_ms: int | None = None
        self._degraded_recovery_since_ms: int | None = None

    def evaluate(self, inputs: DMSV02Inputs) -> DMSV02DecisionState:
        confidence_state = self._confidence(inputs)
        drowsiness_state = self._drowsiness_state(inputs)
        distraction_state = self._distraction_state(inputs)
        availability_state = self._availability_state(inputs, drowsiness_state)
        raw_observation_codes = self._raw_observation_codes(inputs)
        reasons = list(
            dict.fromkeys(
                inputs.availability.reason_codes
                + inputs.attention.attention_reason_codes
                + self._drowsiness_gate_reasons(inputs, drowsiness_state)
                + inputs.drowsiness.perclos_validity_reason_codes
            )
        )

        level = DMSV02Level.NORMAL
        banner = "NORMAL"
        path = "NORMAL > road/available"

        if availability_state == "UNAVAILABLE":
            level = DMSV02Level.CRITICAL
            banner = "DRIVER UNAVAILABLE"
            path = "CRITICAL > DRIVER_UNAVAILABLE"
        elif drowsiness_state == "MICROSLEEP" or (
            inputs.drowsiness.eye_closure_duration_ms >= self.config.eye_closure_microsleep_ms
        ):
            level = DMSV02Level.DANGER
            banner = "DANGER"
            path = "DANGER > MICROSLEEP_OR_LONG_EYE_CLOSURE"
        elif self._danger_distraction(inputs):
            level = DMSV02Level.DANGER
            banner = "DANGER"
            path = f"DANGER > {distraction_state}"
        elif self._warning_distraction(inputs):
            level = DMSV02Level.WARNING
            banner = "DISTRACTION WARNING"
            path = f"WARNING > {distraction_state}"
        elif drowsiness_state == "DROWSY":
            level = DMSV02Level.WARNING
            banner = "DROWSINESS WARNING"
            path = "WARNING > DROWSINESS"
        elif self._monitor_distraction(inputs):
            level = DMSV02Level.MONITOR
            banner = "DMS MONITOR"
            path = f"MONITOR > {distraction_state}"
        elif availability_state == "DEGRADED" and (inputs.driver_proposal_visible or inputs.driver_track_held):
            level = DMSV02Level.MONITOR
            banner = "DMS MONITOR"
            path = "MONITOR > proposal_visible_or_track_held"
        elif availability_state == "DEGRADED":
            level = DMSV02Level.DEGRADED
            banner = "DMS DEGRADED"
            path = "DEGRADED > availability_degraded"
        elif confidence_state in {DMSConfidenceState.LOW, DMSConfidenceState.UNAVAILABLE}:
            level = DMSV02Level.DEGRADED
            banner = "DMS DEGRADED"
            path = "DEGRADED > observation_quality"
        elif self._normal_blocked(inputs):
            level = DMSV02Level.MONITOR
            banner = "DMS MONITOR"
            path = "MONITOR > active_attention_evidence"

        level, banner, path = self._apply_banner_hysteresis(inputs.timestamp_ms, level, banner, path, inputs)
        reasons = self._sanitize_reasons(reasons, banner, inputs, path)
        classification_reason_codes = self._classification_reason_codes(reasons, banner)

        return DMSV02DecisionState(
            drowsiness_state=drowsiness_state,
            distraction_state=distraction_state,
            driver_availability_state=availability_state,
            dms_confidence_state=confidence_state,
            final_level=level,
            final_banner=banner,
            final_decision_path=path,
            reason_codes=reasons,
            raw_observation_codes=raw_observation_codes,
            classification_reason_codes=classification_reason_codes,
        )

    def _confidence(self, inputs: DMSV02Inputs) -> DMSConfidenceState:
        if inputs.health.camera_status.value == "ERROR":
            return DMSConfidenceState.UNAVAILABLE
        if inputs.driver_proposal_visible or inputs.driver_track_held:
            return DMSConfidenceState.MEDIUM
        if not inputs.driver_present and inputs.no_face_duration_ms >= self.config.no_face_degraded_ms:
            return DMSConfidenceState.LOW
        if inputs.health.eye_visibility_score < self.config.eye_visibility_min_confidence:
            return DMSConfidenceState.LOW
        if inputs.attention.pose_reliable is False or inputs.attention.attention_confidence < 0.35:
            return DMSConfidenceState.MEDIUM
        return DMSConfidenceState.HIGH

    def _drowsiness_state(self, inputs: DMSV02Inputs) -> str:
        if inputs.drowsiness.level == DrowsinessLevel.MICROSLEEP and self._valid_microsleep_evidence(inputs):
            return "MICROSLEEP"
        if (
            inputs.drowsiness.eye_closure_duration_ms >= self.config.eye_closure_microsleep_ms
            and self._valid_microsleep_evidence(inputs)
        ):
            return "MICROSLEEP"
        if not self._drowsiness_warning_eligible(inputs):
            if inputs.drowsiness.level == DrowsinessLevel.UNKNOWN:
                if (
                    self.config.drowsiness_resolve_to_none_when_open
                    and inputs.drowsiness.effective_eye_state in {"OPEN", "PARTIALLY_CLOSED"}
                    and inputs.drowsiness.eye_calibration_state in {"CALIBRATED", "FALLBACK"}
                    and inputs.drowsiness.eye_visibility_score >= self.config.eye_visibility_min_confidence
                    and inputs.drowsiness.perclos_valid_time_5s_ms >= self.config.drowsiness_min_valid_eye_ms
                ):
                    return "NONE"
                return "UNKNOWN"
            if inputs.drowsiness.level == DrowsinessLevel.LOW:
                return "EARLY_DROWSY"
            return "NONE"
        if (
            inputs.drowsiness.eye_closure_duration_ms >= self.config.eye_closure_warning_ms
            or inputs.drowsiness.perclos_5s >= self.config.perclos_5s_warning_threshold
            or inputs.drowsiness.level in {DrowsinessLevel.MEDIUM, DrowsinessLevel.HIGH}
        ):
            return "DROWSY"
        if inputs.drowsiness.level == DrowsinessLevel.LOW:
            return "EARLY_DROWSY"
        if inputs.drowsiness.level == DrowsinessLevel.UNKNOWN:
            if (
                self.config.drowsiness_resolve_to_none_when_open
                and inputs.drowsiness.effective_eye_state in {"OPEN", "PARTIALLY_CLOSED"}
                and inputs.drowsiness.eye_calibration_state in {"CALIBRATED", "FALLBACK"}
                and inputs.drowsiness.eye_visibility_score >= self.config.eye_visibility_min_confidence
                and inputs.drowsiness.perclos_valid_time_5s_ms >= self.config.drowsiness_min_valid_eye_ms
            ):
                return "NONE"
            return "UNKNOWN"
        return "NONE"

    def _distraction_state(self, inputs: DMSV02Inputs) -> str:
        if inputs.phone_state == "PHONE_CONFIRMED" or inputs.distraction_type == DistractionType.PHONE_CONFIRMED:
            return "PHONE_CONFIRMED"
        if inputs.phone_state in {
            "PHONE_DOWN_SUSPECTED",
            "PHONE_DOWN_CONFIRMED",
            "PHONE_SUSPECTED",
            "PHONE_TO_EAR_SUSPECTED",
            "PHONE_TO_EAR_CONFIRMED",
            "PHONE_TEXTING_SCROLLING_SUSPECTED",
            "PHONE_TEXTING_SCROLLING_CONFIRMED",
        }:
            return "PHONE_SUSPECTED"
        if inputs.distraction_type == DistractionType.MANUAL:
            return "MANUAL"
        if inputs.attention.attention_substate in {
            AttentionSubstate.HEAD_DOWN_DISTRACTION,
            AttentionSubstate.VISUAL_DISTRACTION,
            AttentionSubstate.SIDE_PROFILE_ATTENTION_LOSS,
        }:
            return "VISUAL"
        if inputs.attention.attention_substate in {
            AttentionSubstate.HEAD_DOWN_CANDIDATE,
            AttentionSubstate.PHONE_DOWN_CANDIDATE,
            AttentionSubstate.AMBIGUOUS,
            AttentionSubstate.HEAD_POSE_UNRELIABLE,
            AttentionSubstate.FACE_PARTIAL_SIDE_PROFILE,
            AttentionSubstate.SIDE_GLANCE_LEFT,
            AttentionSubstate.SIDE_GLANCE_RIGHT,
            AttentionSubstate.SIDE_PROFILE_TRACKED,
            AttentionSubstate.SIDE_PROFILE_RECOVERY,
        }:
            return "MONITOR"
        if inputs.distraction_level == DistractionLevel.UNKNOWN:
            return "UNKNOWN"
        return "NONE"

    def _availability_state(self, inputs: DMSV02Inputs, drowsiness_state: str) -> str:
        temp_unobservable = inputs.driver_observability in {
            "UNOBSERVABLE_TEMP",
            "PARTIALLY_OBSERVABLE",
        }
        if inputs.availability.state == AvailabilityState.UNAVAILABLE:
            if temp_unobservable:
                return "DEGRADED"
            return "UNAVAILABLE"
        if inputs.no_face_duration_ms >= self.config.driver_absent_unavailable_ms:
            if inputs.driver_proposal_visible or inputs.driver_track_held:
                return "DEGRADED"
            if temp_unobservable:
                return "DEGRADED"
            return "UNAVAILABLE"
        if drowsiness_state == "MICROSLEEP":
            return "UNAVAILABLE"
        if inputs.availability.state == AvailabilityState.DEGRADED:
            return "DEGRADED"
        if inputs.attention.attention_state.value in {"DEGRADED", "ATTENTION_LOST"}:
            return "PARTIALLY_AVAILABLE"
        return "AVAILABLE" if inputs.driver_present else "UNCONFIRMED"

    def _apply_banner_hysteresis(
        self,
        timestamp_ms: int,
        level: DMSV02Level,
        banner: str,
        path: str,
        inputs: DMSV02Inputs,
    ) -> tuple[DMSV02Level, str, str]:
        if self._last_banner is None:
            self._set_banner(timestamp_ms, level, banner, path)
            return level, banner, path
        if banner == self._last_banner:
            if banner != "DMS DEGRADED":
                self._degraded_candidate_since_ms = None
            return level, banner, path

        elapsed_ms = timestamp_ms - (self._last_change_ms or timestamp_ms)
        normal_blocked_now = self._last_banner == "NORMAL" and self._normal_blocked(inputs)
        if banner == "DMS DEGRADED" and self._last_banner in {"NORMAL", "DMS MONITOR"}:
            if self._degraded_candidate_since_ms is None:
                self._degraded_candidate_since_ms = timestamp_ms
            if timestamp_ms - self._degraded_candidate_since_ms < self.config.degraded_entry_sustain_ms:
                if normal_blocked_now:
                    path = "MONITOR > degraded_candidate_normal_blocked"
                    self._set_banner(timestamp_ms, DMSV02Level.MONITOR, "DMS MONITOR", path)
                    return DMSV02Level.MONITOR, "DMS MONITOR", path
                return self._last_level or level, self._last_banner, self._last_path
            critical_escalation = True
        else:
            self._degraded_candidate_since_ms = None
            critical_escalation = banner in {"DRIVER UNAVAILABLE", "DANGER", "DISTRACTION WARNING", "DROWSINESS WARNING"}
            critical_escalation = critical_escalation or normal_blocked_now

        if self._last_banner == "DMS DEGRADED" and banner in {"NORMAL", "DMS MONITOR"}:
            if self._observation_strongly_recovered(inputs, banner):
                self._degraded_recovery_since_ms = None
                self._set_banner(timestamp_ms, level, banner, path)
                return level, banner, path
            if self._degraded_recovery_since_ms is None:
                self._degraded_recovery_since_ms = timestamp_ms
            if (
                timestamp_ms - self._degraded_recovery_since_ms < self.config.degraded_recovery_stable_ms
                or elapsed_ms < self.config.degraded_exit_hold_ms
            ):
                return (
                    self._last_level or DMSV02Level.DEGRADED,
                    self._last_banner,
                    "DEGRADED > RECOVERY_HOLD",
                )
        else:
            self._degraded_recovery_since_ms = None

        if not critical_escalation and elapsed_ms < self.config.min_banner_hold_ms:
            return self._last_level or level, self._last_banner, self._last_path
        self._set_banner(timestamp_ms, level, banner, path)
        return level, banner, path

    def _observation_strongly_recovered(self, inputs: DMSV02Inputs, next_banner: str) -> bool:
        if next_banner == "DMS MONITOR" and self._monitor_distraction(inputs):
            return True
        return (
            next_banner == "NORMAL"
            and inputs.driver_present
            and inputs.availability.state == AvailabilityState.AVAILABLE
            and inputs.driver_observability == "OBSERVABLE"
            and inputs.health.face_detection_status.value == "OK"
            and inputs.health.face_visibility_score >= 0.75
            and inputs.health.eye_visibility_score >= self.config.eye_visibility_min_confidence
            and inputs.attention.attention_state.value == "NORMAL"
            and inputs.attention.attention_substate == AttentionSubstate.ROAD
            and inputs.attention.pose_reliable
            and inputs.drowsiness.effective_eye_state in {"OPEN", "PARTIALLY_CLOSED"}
        )

    def _set_banner(self, timestamp_ms: int, level: DMSV02Level, banner: str, path: str) -> None:
        self._last_banner = banner
        self._last_level = level
        self._last_path = path
        self._last_change_ms = timestamp_ms

    def _sanitize_reasons(self, reasons: list[str], banner: str, inputs: DMSV02Inputs, path: str) -> list[str]:
        classification_attention_loss = {
            "GAZE_OFF_ROAD",
            "VISUAL_ATTENTION_LOSS",
            "GAZE_OFF_ROAD_SUSTAINED",
        }
        stale_normal_reasons = classification_attention_loss | {
            "PHONE_TO_EAR_SUSPECTED",
            "PHONE_TO_EAR_CONFIRMED",
            "PHONE_TO_EAR_CANDIDATE",
            "PHONE_TO_EAR",
            "PHONE_SUSPECTED",
            "PHONE_CONFIRMED",
            "HEAD_DOWN",
            "FACE_LOST",
            "DRIVER_FACE_NOT_VISIBLE",
            "DRIVER_FACE_LOST_TEMP",
            "SIDE_PROFILE_FACE_LOST",
            "LOW_EYE_VISIBILITY",
            "HEAD_POSE_UNRELIABLE",
        }
        cleaned = [reason for reason in reasons if reason not in {"VALID"}]
        low_head_motion_present = "LOW_HEAD_MOTION" in cleaned
        if banner == "DMS DEGRADED":
            cleaned = [reason for reason in cleaned if reason != "LOW_HEAD_MOTION"]
            if low_head_motion_present:
                cleaned.append("LOW_HEAD_MOTION_IGNORED_FOR_DEGRADED")
        elif low_head_motion_present:
            cleaned.append("LOW_HEAD_MOTION_DROWSINESS_SUPPORT_ONLY")
        if "RECOVERY_HOLD" in path:
            cleaned.extend(["DEGRADED_RECOVERY_HOLD", "OBSERVATION_RECOVERY_WAIT"])
        if banner == "NORMAL":
            cleaned = [reason for reason in cleaned if reason not in stale_normal_reasons]
            cleaned.append("ROAD_GAZE_CONFIRMED")
        elif banner == "DMS MONITOR":
            cleaned = ["SHORT_GLANCE_AWAY" if reason == "GAZE_OFF_ROAD" else reason for reason in cleaned]
            if inputs.driver_proposal_visible or inputs.driver_track_held:
                cleaned.extend(["NORMAL_ALLOWED_PROPOSAL_VISIBLE_HELD", "WEBCAM_DRIVER_TRACK_HELD"])
            if inputs.attention.ambiguous_attention_loss:
                cleaned.extend(["AMBIGUOUS_ATTENTION_HOLD", "AMBIGUOUS_TO_MONITOR"])
        elif banner in {"DISTRACTION WARNING", "DANGER"} and (
            "GAZE_OFF_ROAD" in reasons or inputs.attention.gaze_offroad_duration_ms > 0
        ):
            cleaned = [reason for reason in cleaned if reason != "GAZE_OFF_ROAD"]
            cleaned.extend(["GAZE_OFF_ROAD_SUSTAINED", "VISUAL_ATTENTION_LOSS"])
            if inputs.attention.ambiguous_attention_loss:
                cleaned.extend(["AMBIGUOUS_TO_DISTRACTION_WARNING", "AMBIGUOUS_NOT_DROWSINESS"])
        return list(dict.fromkeys(cleaned))

    def _monitor_distraction(self, inputs: DMSV02Inputs) -> bool:
        return (
            inputs.attention.head_down_duration_ms >= self.config.head_down_monitor_ms
            or inputs.attention.gaze_offroad_duration_ms >= self.config.single_offroad_glance_monitor_ms
            or inputs.attention.side_glance_duration_ms >= self.config.side_glance_monitor_ms
            or inputs.attention.attention_substate
            in {
                AttentionSubstate.SIDE_GLANCE_LEFT,
                AttentionSubstate.SIDE_GLANCE_RIGHT,
                AttentionSubstate.SIDE_PROFILE_TRACKED,
                AttentionSubstate.SIDE_PROFILE_RECOVERY,
            }
            or inputs.attention.phone_down_candidate_duration_ms >= self.config.phone_down_monitor_ms
            or inputs.attention.phone_texting_candidate_duration_ms >= self.config.phone_down_monitor_ms
            or (
                inputs.attention.ambiguous_attention_loss
                and inputs.attention.attention_lost_duration_ms >= self.config.ambiguous_to_monitor_ms
            )
        )

    def _warning_distraction(self, inputs: DMSV02Inputs) -> bool:
        return (
            inputs.attention.head_down_duration_ms >= self.config.head_down_warning_ms
            or inputs.attention.gaze_offroad_duration_ms >= self.config.single_offroad_glance_warning_ms
            or inputs.attention.side_glance_duration_ms >= self.config.side_glance_warning_ms
            or inputs.attention.attention_substate == AttentionSubstate.SIDE_PROFILE_ATTENTION_LOSS
            or inputs.attention.phone_down_candidate_duration_ms >= self.config.phone_down_warning_ms
            or inputs.attention.phone_texting_candidate_duration_ms >= self.config.phone_texting_warning_ms
            or inputs.phone_state in {
                "PHONE_TO_EAR_SUSPECTED",
                "PHONE_TO_EAR_CONFIRMED",
                "PHONE_DOWN_CONFIRMED",
                "PHONE_TEXTING_SCROLLING_SUSPECTED",
                "PHONE_TEXTING_SCROLLING_CONFIRMED",
            }
            or (
                inputs.attention.ambiguous_attention_loss
                and inputs.attention.attention_lost_duration_ms >= self.config.ambiguous_to_warning_ms
                and (
                    inputs.attention.head_down_duration_ms >= self.config.head_down_candidate_ms
                    or inputs.attention.gaze_offroad_duration_ms > 0
                    or inputs.attention.phone_down_candidate_duration_ms > 0
                )
            )
            or inputs.distraction_level in {DistractionLevel.MEDIUM, DistractionLevel.HIGH}
        )

    def _danger_distraction(self, inputs: DMSV02Inputs) -> bool:
        return (
            inputs.attention.head_down_duration_ms >= self.config.head_down_danger_ms
            or inputs.attention.gaze_offroad_duration_ms >= self.config.single_offroad_glance_danger_ms
            or inputs.attention.side_glance_duration_ms >= self.config.side_glance_danger_ms
            or inputs.attention.phone_down_candidate_duration_ms >= self.config.phone_down_danger_ms
            or inputs.attention.phone_texting_candidate_duration_ms >= self.config.phone_down_danger_ms
            or (
                inputs.phone_state in {
                    "PHONE_DOWN_SUSPECTED",
                    "PHONE_DOWN_CONFIRMED",
                    "PHONE_TEXTING_SCROLLING_SUSPECTED",
                    "PHONE_TEXTING_SCROLLING_CONFIRMED",
                    "PHONE_CONFIRMED",
                }
                and inputs.attention.gaze_offroad_duration_ms >= self.config.phone_down_danger_ms
            )
        )

    def _valid_microsleep_evidence(self, inputs: DMSV02Inputs) -> bool:
        return (
            inputs.drowsiness.perclos_valid
            and inputs.drowsiness.effective_eye_state == "CLOSED"
            and inputs.drowsiness.eye_visibility_score >= self.config.eye_visibility_min_confidence
            and "EYE_CLOSURE_SUPPRESSED_BY_DOWNWARD_GAZE" not in inputs.drowsiness.perclos_validity_reason_codes
        )

    def _drowsiness_warning_eligible(self, inputs: DMSV02Inputs) -> bool:
        if self._phone_or_downward_posture(inputs) and not self._valid_microsleep_evidence(inputs):
            return False
        if "EYE_CLOSURE_SUPPRESSED_BY_DOWNWARD_GAZE" in inputs.drowsiness.perclos_validity_reason_codes:
            return False
        perclos_valid = (
            inputs.drowsiness.perclos_valid
            and inputs.drowsiness.perclos_valid_time_5s_ms >= self.config.drowsiness_min_valid_eye_ms
            and inputs.drowsiness.eye_visibility_score >= self.config.eye_visibility_min_confidence
        )
        perclos_over_threshold = (
            inputs.drowsiness.perclos_5s >= self.config.perclos_5s_warning_threshold
            or inputs.drowsiness.perclos_60s >= self.config.perclos_60s_medium_threshold
        )
        sustained_eye_closure_valid = (
            inputs.drowsiness.effective_eye_state == "CLOSED"
            and inputs.drowsiness.eye_closure_duration_ms >= self.config.eye_closure_warning_ms
            and inputs.drowsiness.eye_closure_duration_ms > self.config.blink_normal_max_ms
            and inputs.drowsiness.eye_visibility_score >= self.config.eye_visibility_min_confidence
        )
        return (perclos_valid and perclos_over_threshold) or sustained_eye_closure_valid

    def _drowsiness_gate_reasons(self, inputs: DMSV02Inputs, drowsiness_state: str) -> list[str]:
        reasons: list[str] = []
        if drowsiness_state in {"DROWSY", "MICROSLEEP"}:
            if inputs.drowsiness.perclos_valid and inputs.drowsiness.perclos_5s >= self.config.perclos_5s_warning_threshold:
                reasons.append("DROWSINESS_VALID_PERCLOS")
            if inputs.drowsiness.eye_closure_duration_ms >= self.config.eye_closure_warning_ms:
                reasons.append("DROWSINESS_VALID_EYE_CLOSURE")
            if drowsiness_state == "MICROSLEEP":
                reasons.append("DROWSINESS_VALID_MICROSLEEP")
            return reasons
        if self._phone_or_downward_posture(inputs):
            reasons.append("DROWSINESS_SUPPRESSED_DOWNWARD_GAZE")
            if self._phone_like(inputs):
                reasons.append("DROWSINESS_SUPPRESSED_POSSIBLE_PHONE")
        if not inputs.drowsiness.perclos_valid or inputs.drowsiness.effective_eye_state == "UNKNOWN":
            reasons.append("DROWSINESS_SUPPRESSED_NO_VALID_EYE_EVIDENCE")
        return reasons

    def _phone_like(self, inputs: DMSV02Inputs) -> bool:
        return (
            inputs.phone_state
            in {
                "PHONE_SUSPECTED",
                "PHONE_CONFIRMED",
                "PHONE_TO_EAR_SUSPECTED",
                "PHONE_TO_EAR_CONFIRMED",
                "PHONE_DOWN_SUSPECTED",
                "PHONE_DOWN_CONFIRMED",
                "PHONE_TEXTING_SCROLLING_SUSPECTED",
                "PHONE_TEXTING_SCROLLING_CONFIRMED",
            }
            or inputs.attention.phone_suspicion_candidate
            or inputs.attention.phone_down_candidate_duration_ms >= self.config.phone_down_candidate_ms
            or inputs.attention.phone_texting_candidate_duration_ms >= self.config.phone_down_monitor_ms
        )

    def _phone_or_downward_posture(self, inputs: DMSV02Inputs) -> bool:
        return self._phone_like(inputs) or (
            inputs.attention.head_down_duration_ms >= self.config.head_down_candidate_ms
            and inputs.attention.gaze_offroad_duration_ms > 0
        )

    def _normal_blocked(self, inputs: DMSV02Inputs) -> bool:
        return (
            inputs.attention.attention_state.value == "DEGRADED"
            or inputs.attention.attention_substate
            in {
                AttentionSubstate.FACE_LOST,
                AttentionSubstate.AMBIGUOUS,
                AttentionSubstate.HEAD_POSE_UNRELIABLE,
            }
            and not self._side_profile_attention_available(inputs)
            or inputs.attention.head_down_duration_ms > self.config.head_down_candidate_ms
            or inputs.attention.phone_down_candidate_duration_ms > self.config.phone_down_monitor_ms
            or inputs.attention.phone_texting_candidate_duration_ms > self.config.phone_down_monitor_ms
            or inputs.attention.gaze_offroad_duration_ms > self.config.gaze_away_low_ms
            or inputs.driver_observability == "UNOBSERVABLE_TEMP"
            and not self._side_profile_attention_available(inputs)
        )

    def _side_profile_attention_available(self, inputs: DMSV02Inputs) -> bool:
        return (
            inputs.attention.yaw_classifiable
            or inputs.attention.side_profile_context_active
            or inputs.attention.attention_substate
            in {
                AttentionSubstate.SIDE_GLANCE_LEFT,
                AttentionSubstate.SIDE_GLANCE_RIGHT,
                AttentionSubstate.SIDE_PROFILE_TRACKED,
                AttentionSubstate.SIDE_PROFILE_ATTENTION_LOSS,
                AttentionSubstate.SIDE_PROFILE_RECOVERY,
            }
        )

    @staticmethod
    def _classification_reason_codes(reasons: list[str], banner: str) -> list[str]:
        raw_prefixes = ("RAW_",)
        return [reason for reason in reasons if not reason.startswith(raw_prefixes)]

    @staticmethod
    def _raw_observation_codes(inputs: DMSV02Inputs) -> list[str]:
        raw: list[str] = []
        if inputs.attention.gaze_offroad_duration_ms > 0 or "GAZE_OFF_ROAD" in inputs.attention.attention_reason_codes:
            raw.append("RAW_GAZE_OFF_ROAD")
        if inputs.attention.head_down_duration_ms > 0 or "HEAD_DOWN" in inputs.attention.attention_reason_codes:
            raw.append("RAW_HEAD_DOWN")
        if "POSSIBLE_PHONE_POSTURE" in inputs.attention.attention_reason_codes:
            raw.append("RAW_POSSIBLE_PHONE_POSTURE")
        if inputs.drowsiness.eye_visibility_score < 0.5 or "LOW_EYE_VISIBILITY" in inputs.attention.attention_reason_codes:
            raw.append("RAW_LOW_EYE_VISIBILITY")
        if inputs.attention.attention_substate in {
            AttentionSubstate.FACE_PARTIAL_SIDE_PROFILE,
            AttentionSubstate.FACE_LOST,
            AttentionSubstate.SIDE_GLANCE_LEFT,
            AttentionSubstate.SIDE_GLANCE_RIGHT,
            AttentionSubstate.SIDE_PROFILE_TRACKED,
            AttentionSubstate.SIDE_PROFILE_ATTENTION_LOSS,
        }:
            raw.append("RAW_SIDE_PROFILE")
        if not inputs.attention.pose_reliable:
            raw.append("RAW_LANDMARK_WEAK")
        return list(dict.fromkeys(raw))
