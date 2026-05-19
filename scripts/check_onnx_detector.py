from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ind_vias_perception.config.settings import load_settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the configured ONNX detector model.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    detection_cfg = settings.raw.get("detection", {})

    backend = str(detection_cfg.get("backend", "dummy"))
    model_path = _resolve_model_path(detection_cfg.get("onnx_model_path", "models/weights/detector.onnx"))
    input_size = detection_cfg.get("input_size", [640, 640])
    class_names = {int(k): str(v) for k, v in detection_cfg.get("class_names", {}).items()}

    print(f"model path: {model_path}")
    print(f"input size: {input_size}")
    print(f"class names: {class_names}")
    print(f"backend type: {backend}")

    exists = model_path.exists()
    print(f"model exists: {exists}")
    if not exists:
        print("OpenCV DNN loaded successfully: False")
        if backend == "onnx":
            print(f"ERROR: detection.backend=onnx but ONNX detector model is missing: {model_path}")
            return 1
        return 0

    try:
        cv2.dnn.readNetFromONNX(str(model_path))
    except cv2.error as exc:
        print("OpenCV DNN loaded successfully: False")
        print(f"ERROR: OpenCV DNN failed to load ONNX detector model: {exc}")
        return 1

    print("OpenCV DNN loaded successfully: True")
    return 0


def _resolve_model_path(path_value: str) -> Path:
    model_path = Path(path_value)
    if model_path.is_absolute():
        return model_path
    return REPO_ROOT / model_path


if __name__ == "__main__":
    raise SystemExit(main())
