from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def open_video_source(video: str | None, camera: int | None) -> cv2.VideoCapture:
    if video is None and camera is None:
        raise SystemExit("Provide either --video or --camera.")
    source: str | int = camera if camera is not None else str(video)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    return cap


def make_video_writer(path: str, fps: float, frame: np.ndarray) -> cv2.VideoWriter:
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not write video: {path}")
    return writer


def resize_to_width(frame: np.ndarray, width: int | None) -> np.ndarray:
    if width is None or width <= 0 or frame.shape[1] <= width:
        return frame
    scale = width / float(frame.shape[1])
    return cv2.resize(frame, (width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
