from __future__ import annotations

import numpy as np
from ind_vias_perception.common.types import FramePacket
from ind_vias_perception.config.settings import load_settings
from ind_vias_perception.pipeline.factory import build_pipeline


def test_pipeline_runs_with_all_atomic_components():
    settings = load_settings("configs/default.yaml")
    pipeline = build_pipeline(settings)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = pipeline.process(FramePacket(frame=frame, timestamp_s=0.0))
    assert out.detections
    assert out.safety_payload["cais_mode"] in {"nominal", "enhanced", "critical", "degraded"}
