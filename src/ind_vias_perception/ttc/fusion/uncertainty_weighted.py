from __future__ import annotations

from collections.abc import Iterable

def fuse_ttc(values: Iterable[tuple[float | None, float]]) -> float | None:
    valid = [(v, max(s, 1e-3)) for v, s in values if v is not None and v > 0]
    if not valid:
        return None
    weights = [1.0 / s**2 for _, s in valid]
    return sum(w * v for w, (v, _) in zip(weights, valid)) / sum(weights)
