from __future__ import annotations

import copy
from dataclasses import dataclass

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import DMSState, DMSV02Level, VehicleRuntimeState


@dataclass
class VehicleInputSnapshot:
    speed_kph: float
    left_indicator_on: bool
    right_indicator_on: bool


class VehicleStateManager:
    def __init__(self, config: DMSConfig, *, output_fps_mode: str = "measured") -> None:
        self.config = config
        self.speed_kph = max(0.0, float(config.vehicle_speed_initial_kph))
        self.left_indicator_on = False
        self.right_indicator_on = False
        self.output_fps_mode = output_fps_mode
        self._first_timestamp_ms: int | None = None
        self._manual_speed_override = False
        self._active_since_ms: int | None = None
        self._gate_state = "STARTUP_INITIALIZING"

    def increase_speed(self, *, fast: bool = False) -> None:
        step = self.config.speed_adjust_fast_step_kph if fast else self.config.speed_adjust_step_kph
        self.speed_kph = max(0.0, self.speed_kph + step)
        self._manual_speed_override = True

    def decrease_speed(self, *, fast: bool = False) -> None:
        step = self.config.speed_adjust_fast_step_kph if fast else self.config.speed_adjust_step_kph
        self.speed_kph = max(0.0, self.speed_kph - step)
        self._manual_speed_override = True

    def toggle_left_indicator(self) -> None:
        self.left_indicator_on = not self.left_indicator_on
        if self.left_indicator_on:
            self.right_indicator_on = False

    def toggle_right_indicator(self) -> None:
        self.right_indicator_on = not self.right_indicator_on
        if self.right_indicator_on:
            self.left_indicator_on = False

    def update(
        self,
        state: DMSState,
        *,
        timestamp_ms: int,
        live_output_fps: float = 0.0,
        frame_capture_time_ms: float = 0.0,
        processing_time_ms: float = 0.0,
        frame_write_time_ms: float = 0.0,
    ) -> DMSState:
        if self._first_timestamp_ms is None:
            self._first_timestamp_ms = timestamp_ms
        self._apply_startup_ramp(timestamp_ms)
        gate_state, gate_reasons = self._speed_gate(timestamp_ms)
        vehicle = VehicleRuntimeState(
            ego_vehicle_speed_kph=self.speed_kph,
            ego_vehicle_speed_source="SIMULATED",
            vehicle_speed_sim_enabled=self.config.vehicle_speed_sim_enabled,
            dms_speed_gate_state=gate_state,
            dms_operational_mode=gate_state,
            dms_alerts_enabled=gate_state in {"DMS_ACTIVATED", "DMS_ACTIVE_MONITORING", "DMS_LIMITED"},
            dms_alert_suppression_reason="NONE",
            dms_activation_threshold_kph=self.config.dms_activation_speed_kph,
            dms_deactivation_threshold_kph=self.config.dms_deactivation_speed_kph,
            vehicle_speed_reason_codes=gate_reasons,
            left_indicator_on=self.left_indicator_on,
            right_indicator_on=self.right_indicator_on,
            indicator_reason_codes=self._indicator_reasons(),
            live_output_fps_mode=self.output_fps_mode,
            live_output_fps=live_output_fps,
            frame_capture_time_ms=frame_capture_time_ms,
            processing_time_ms=processing_time_ms,
            frame_write_time_ms=frame_write_time_ms,
            critical_unavailable_requires_no_face=self.config.critical_unavailable_requires_no_face,
            critical_unavailable_requires_no_body=self.config.critical_unavailable_requires_no_body,
        )
        if not vehicle.dms_alerts_enabled and self.config.standby_suppresses_alerts:
            vehicle.dms_alert_suppression_reason = gate_state
        state.vehicle = vehicle
        self._apply_indicator_mirror_arbitration(state, vehicle)
        self._apply_hmi_banner(state, vehicle)
        vehicle.vehicle_monitor_line = self._vehicle_monitor_line(vehicle)
        state.vehicle = vehicle
        return state

    def _apply_startup_ramp(self, timestamp_ms: int) -> None:
        if not self.config.vehicle_speed_startup_ramp_enabled or self._manual_speed_override:
            return
        if self._first_timestamp_ms is None:
            return
        elapsed_ms = max(0, timestamp_ms - self._first_timestamp_ms)
        if elapsed_ms > self.config.vehicle_speed_startup_ramp_ms:
            return
        span = max(1, self.config.vehicle_speed_startup_ramp_ms)
        ratio = min(1.0, elapsed_ms / span)
        target = max(0.0, self.config.vehicle_speed_startup_ramp_target_kph)
        self.speed_kph = self.config.vehicle_speed_initial_kph + (
            target - self.config.vehicle_speed_initial_kph
        ) * ratio

    def _speed_gate(self, timestamp_ms: int) -> tuple[str, list[str]]:
        reasons: list[str] = []
        if not self.config.vehicle_speed_sim_enabled:
            return "DMS_ACTIVE_MONITORING", ["VEHICLE_SPEED_GATE_DISABLED"]
        startup_elapsed_ms = 0 if self._first_timestamp_ms is None else timestamp_ms - self._first_timestamp_ms
        if self.config.vehicle_speed_startup_ramp_enabled and startup_elapsed_ms < self.config.vehicle_speed_startup_ramp_ms:
            self._gate_state = "STARTUP_INITIALIZING"
            return self._gate_state, ["STARTUP_INITIALIZING", "VEHICLE_SPEED_SIM_STARTUP_RAMP"]
        if self._gate_state in {"DMS_ACTIVATED", "DMS_ACTIVE_MONITORING", "DMS_LIMITED"}:
            if self.speed_kph < self.config.dms_deactivation_speed_kph:
                self._gate_state = "STANDBY"
                self._active_since_ms = None
                reasons.append("DMS_STANDBY_SPEED_BELOW_DEACTIVATION")
            elif self._active_since_ms is not None and timestamp_ms - self._active_since_ms >= self.config.dms_activation_banner_ms:
                self._gate_state = "DMS_ACTIVE_MONITORING"
                reasons.append("DMS_ACTIVE_MONITORING_SPEED_OK")
            else:
                reasons.append("DMS_ACTIVATED_SPEED_THRESHOLD_CROSSED")
        elif self.speed_kph > self.config.dms_activation_speed_kph:
            self._gate_state = "DMS_ACTIVATED"
            self._active_since_ms = timestamp_ms
            reasons.append("DMS_ACTIVATED_SPEED_THRESHOLD_CROSSED")
        else:
            self._gate_state = "STANDBY"
            reasons.append("DMS_STANDBY_SPEED_BELOW_THRESHOLD")
        return self._gate_state, reasons

    def _indicator_reasons(self) -> list[str]:
        if self.left_indicator_on:
            return ["LEFT_INDICATOR_ACTIVE"]
        if self.right_indicator_on:
            return ["RIGHT_INDICATOR_ACTIVE"]
        return ["INDICATORS_OFF"]

    def _apply_indicator_mirror_arbitration(self, state: DMSState, vehicle: VehicleRuntimeState) -> None:
        yaw = state.gaze.relative_yaw_deg
        pitch = abs(state.gaze.relative_pitch_deg)
        matching_left = vehicle.left_indicator_on and yaw < -self.config.side_glance_monitor_deg
        matching_right = vehicle.right_indicator_on and yaw > self.config.side_glance_monitor_deg
        matching_indicator = matching_left or matching_right
        side_glance_ms = state.attention.side_glance_duration_ms
        no_phone_or_drowsy = (
            state.dms_v02.final_banner not in {"DROWSINESS WARNING", "DANGER", "DRIVER UNAVAILABLE"}
            and state.phone_use.driver_state
            not in {"PHONE_TO_EAR_SUSPECTED", "PHONE_TO_EAR_CONFIRMED", "PHONE_CONFIRMED"}
        )
        if (
            self.config.mirror_check_requires_matching_indicator
            and matching_indicator
            and pitch <= self.config.mirror_check_max_abs_pitch_deg
            and no_phone_or_drowsy
        ):
            side = "LEFT" if matching_left else "RIGHT"
            if side_glance_ms <= self.config.mirror_check_allowed_ms:
                vehicle.sanctioned_task_state = f"{side}_MIRROR_CHECK_ALLOWED"
                vehicle.sanctioned_task_reason_codes = [
                    "INDICATOR_MATCHES_SIDE_GLANCE",
                    "MIRROR_CHECK_ALLOWED",
                    "DISTRACTION_WARNING_SUPPRESSED_MIRROR_CHECK",
                ]
                state.dms_v02.final_banner = "NORMAL"
                state.dms_v02.final_level = DMSV02Level.NORMAL
                state.dms_v02.final_decision_path = "NORMAL > sanctioned_mirror_check"
            elif side_glance_ms <= self.config.mirror_check_monitor_ms:
                vehicle.sanctioned_task_state = f"{side}_MIRROR_CHECK_MONITOR"
                vehicle.sanctioned_task_reason_codes = [
                    "INDICATOR_MATCHES_SIDE_GLANCE",
                    "MIRROR_CHECK_MONITOR",
                ]
                state.dms_v02.final_banner = "DMS MONITOR"
                state.dms_v02.final_level = DMSV02Level.MONITOR
                state.dms_v02.final_decision_path = "MONITOR > sanctioned_mirror_check"
            else:
                vehicle.sanctioned_task_state = f"{side}_MIRROR_CHECK_EXCEEDED"
                vehicle.sanctioned_task_reason_codes = [
                    "INDICATOR_MATCHES_SIDE_GLANCE",
                    "MIRROR_CHECK_DURATION_EXCEEDED",
                ]

    def _apply_hmi_banner(self, state: DMSState, vehicle: VehicleRuntimeState) -> None:
        raw_banner = state.dms_v02.final_banner or "NORMAL"
        if vehicle.dms_operational_mode == "STARTUP_INITIALIZING":
            hmi = "DMS STANDBY: Initializing vehicle speed simulation"
            subtype = "STARTUP_INITIALIZING"
            primary = "STARTUP_INITIALIZING"
        elif vehicle.dms_operational_mode == "STANDBY" and self.config.standby_suppresses_alerts:
            hmi = f"DMS STANDBY: Speed below {self.config.dms_activation_speed_kph:.0f} km/h"
            subtype = "SPEED_BELOW_ACTIVATION"
            primary = "DMS_STANDBY_SPEED_BELOW_THRESHOLD"
        elif vehicle.dms_operational_mode == "DMS_ACTIVATED":
            hmi = "DMS ACTIVATED: Monitoring enabled"
            subtype = "DMS_ACTIVATED"
            primary = "DMS_ACTIVATED_SPEED_THRESHOLD_CROSSED"
        else:
            subtype, reason = self._banner_subtype(state)
            hmi = f"{raw_banner}: {reason}"
            primary = subtype
        vehicle.hmi_banner_text = hmi
        vehicle.hmi_alert_subtype = subtype
        vehicle.hmi_primary_reason = primary
        vehicle.hmi_secondary_reason = state.dms_v02.final_decision_path
        state.dms_v02.hmi_banner_text = hmi
        state.dms_v02.alert_subtype = subtype
        state.dms_v02.alert_explanation = hmi
        state.dms_v02.hmi_primary_reason = primary
        state.dms_v02.hmi_secondary_reason = vehicle.hmi_secondary_reason

    def _banner_subtype(self, state: DMSState) -> tuple[str, str]:
        banner = state.dms_v02.final_banner
        if banner == "NORMAL":
            return "ROAD_ATTENTIVE", "Road attentive"
        if banner == "DMS MONITOR":
            if state.vehicle.sanctioned_task_state.endswith("MIRROR_CHECK_MONITOR"):
                return "MIRROR_CHECK_MONITOR", "Sanctioned mirror check"
            return "EARLY_ATTENTION_RISK", "Monitoring attention candidate"
        if banner == "DISTRACTION WARNING":
            sub = state.attention.attention_substate.value
            if "SIDE" in sub:
                return "HEAD_TURNED_SIDE", "Head turned side"
            if "HEAD_DOWN" in sub or "PHONE" in sub:
                return "HEAD_DOWN_OR_PHONE", "Head down / phone posture"
            return "VISUAL_DISTRACTION", "Visual distraction"
        if banner == "DROWSINESS WARNING":
            return "DROWSINESS_EVIDENCE", "Drowsiness evidence"
        if banner == "DMS DEGRADED":
            return "OBSERVATION_UNRELIABLE", "Observation unreliable"
        if banner == "DRIVER UNAVAILABLE":
            return "DRIVER_UNAVAILABLE", "Driver unavailable"
        if banner == "DANGER":
            return "DANGER", "Immediate risk"
        return "UNKNOWN", banner

    def _vehicle_monitor_line(self, vehicle: VehicleRuntimeState) -> str:
        left = "ON" if vehicle.left_indicator_on else "OFF"
        right = "ON" if vehicle.right_indicator_on else "OFF"
        alerts = "ON" if vehicle.dms_alerts_enabled else "OFF"
        return (
            f"Speed {vehicle.ego_vehicle_speed_kph:.1f} km/h | Gate {vehicle.dms_speed_gate_state} | "
            f"Alerts {alerts} | L {left} R {right} | Task {vehicle.sanctioned_task_state}"
        )

    def snapshot(self) -> VehicleInputSnapshot:
        return VehicleInputSnapshot(self.speed_kph, self.left_indicator_on, self.right_indicator_on)

    def clone_state_for_tests(self, state: DMSState) -> DMSState:
        return copy.deepcopy(state)
