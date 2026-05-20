from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WarningConfirmation:
    warning_candidate: str
    confirmed_warning_level: str
    confirmation_count: int
    confirmation_required: int


class WarningConfirmationGate:
    def __init__(self, enabled: bool = False, required_frames: dict[str, int] | None = None):
        self.enabled = enabled
        self.required_frames = required_frames or {}
        self._last_target_id: int | None = None
        self._last_candidate = "none"
        self._count = 0

    def update(self, target_id: int | None, raw_warning_level: str, aeb_candidate: bool) -> WarningConfirmation:
        if not self.enabled:
            return WarningConfirmation(raw_warning_level, raw_warning_level, 1, 1)

        candidate_key = self._candidate_key(raw_warning_level, aeb_candidate)
        if candidate_key == "none":
            self._reset()
            return WarningConfirmation(raw_warning_level, "none", 0, self._required(candidate_key))

        if target_id == self._last_target_id and candidate_key == self._last_candidate:
            self._count += 1
        else:
            self._last_target_id = target_id
            self._last_candidate = candidate_key
            self._count = 1

        required = self._required(candidate_key)
        confirmed = raw_warning_level if self._count >= required else "none"
        return WarningConfirmation(raw_warning_level, confirmed, self._count, required)

    def _candidate_key(self, raw_warning_level: str, aeb_candidate: bool) -> str:
        if aeb_candidate:
            return "aeb_ready"
        if raw_warning_level == "strong":
            return "strong_warning"
        if raw_warning_level in {"visual", "advisory"}:
            return "warning"
        return "none"

    def _required(self, candidate_key: str) -> int:
        if candidate_key == "none":
            return 0
        return int(self.required_frames.get(candidate_key, 1))

    def _reset(self) -> None:
        self._last_target_id = None
        self._last_candidate = "none"
        self._count = 0
