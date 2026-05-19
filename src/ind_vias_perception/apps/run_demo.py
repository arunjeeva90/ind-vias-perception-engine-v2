from __future__ import annotations

import argparse
import cv2
from ind_vias_perception.common.types import FramePacket
from ind_vias_perception.config.settings import load_settings
from ind_vias_perception.pipeline.factory import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--video", default=None)
    parser.add_argument("--max-frames", type=int, default=3)
    args = parser.parse_args()

    settings = load_settings(args.config)
    pipeline = build_pipeline(settings)

    if args.video is None:
        import numpy as np
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(args.max_frames):
            out = pipeline.process(FramePacket(frame=frame, timestamp_s=i / 30.0, frame_id=i))
            print(out.safety_payload)
        return

    cap = cv2.VideoCapture(args.video)
    i = 0
    while i < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        out = pipeline.process(FramePacket(frame=frame, timestamp_s=i / 30.0, frame_id=i))
        print(out.safety_payload)
        i += 1


if __name__ == "__main__":
    main()
