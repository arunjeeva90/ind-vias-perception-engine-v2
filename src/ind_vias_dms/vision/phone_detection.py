from __future__ import annotations

from ind_vias_dms.core.types import PlaceholderState


class PhoneDetectionPlaceholder:
    def process(self, frame: object) -> PlaceholderState:
        # TODO(v0.2): replace with a lightweight phone-use detector.
        return PlaceholderState(state="UNKNOWN", confidence=0.0)
