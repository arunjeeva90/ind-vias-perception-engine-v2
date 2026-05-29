from __future__ import annotations

from ind_vias_dms.core.types import SeatbeltAuthenticity


class SeatbeltDetectionPlaceholder:
    def process(self, frame: object) -> SeatbeltAuthenticity:
        # TODO(v0.3): add visual belt-path and buckle-signal fusion.
        return SeatbeltAuthenticity()
