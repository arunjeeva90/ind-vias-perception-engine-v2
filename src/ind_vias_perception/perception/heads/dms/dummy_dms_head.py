from __future__ import annotations

import numpy as np


class DummyDMSHead:
    name = "dummy_dms"

    def forward(self, cabin_frame: np.ndarray | None) -> dict[str, object]:
        return {"available": cabin_frame is not None, "driver_attention": None}
