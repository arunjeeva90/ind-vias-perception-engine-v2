from __future__ import annotations

import numpy as np


class KalmanFilterCV:
    def __init__(self, x: float, y: float):
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)
        self.p = np.eye(4) * 10.0

    def predict(self, dt: float) -> None:
        f = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        q = np.eye(4) * 0.05
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q

    def update(self, z: np.ndarray, r_scale: float = 1.0) -> None:
        h = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        r = np.eye(2) * max(0.1, r_scale)
        y = z - h @ self.x
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.p = (np.eye(4) - k @ h) @ self.p
