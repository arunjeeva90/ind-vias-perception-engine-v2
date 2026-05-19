from __future__ import annotations

import numpy as np


def cv_predict(state: np.ndarray, dt: float) -> np.ndarray:
    out = state.copy()
    out[0] += state[2] * dt
    out[1] += state[3] * dt
    return out
