from __future__ import annotations

from ind_vias_dms.core.types import AvailabilityState, DistractionLevel, DrowsinessLevel

GREEN = (60, 210, 90)
YELLOW = (40, 210, 240)
ORANGE = (0, 150, 255)
RED = (40, 40, 230)
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)
GRAY = (80, 80, 80)


def status_color(value: object) -> tuple[int, int, int]:
    if value in {DrowsinessLevel.MICROSLEEP, AvailabilityState.UNAVAILABLE, DistractionLevel.HIGH}:
        return RED
    if value in {DrowsinessLevel.HIGH, DrowsinessLevel.MEDIUM, AvailabilityState.DEGRADED, DistractionLevel.MEDIUM}:
        return ORANGE
    if value in {DrowsinessLevel.LOW, DistractionLevel.LOW}:
        return YELLOW
    return GREEN
