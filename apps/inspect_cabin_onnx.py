from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2  # noqa: E402

from ind_vias_dms.core.config import DMSConfig  # noqa: E402
from ind_vias_dms.vision.cabin_object_detection import CabinClassMap, CabinObjectDetector  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a candidate cabin-object ONNX model safely.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", default=None)
    source.add_argument("--video", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--class-map", default="configs/dms/cabin_object_class_map.json")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-image", default=None)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=50)
    return parser


def inspect_model(args: argparse.Namespace) -> dict[str, Any]:
    class_map = CabinClassMap(args.class_map)
    report: dict[str, Any] = {
        "model_path": args.model,
        "model_exists": Path(args.model).exists(),
        "class_map_path": args.class_map,
        "class_map_loaded": class_map.status == "CLASS_MAP_READY",
        "class_map_status": class_map.status,
        "backend_status": "NOT_RUN",
        "input_width": args.input_width,
        "input_height": args.input_height,
        "raw_output_shapes": [],
        "parsed_detection_count": 0,
        "detections": [],
        "warnings": [],
        "errors": [],
        "parser_status": "NOT_RUN",
    }
    config = DMSConfig(
        cabin_evidence={
            "enabled": True,
            "detector_backend": "onnx",
            "model_path": args.model,
            "class_map_path": args.class_map,
            "input_width": args.input_width,
            "input_height": args.input_height,
            "min_confidence": args.conf,
            "nms_iou_threshold": args.nms,
            "max_detections": args.max_detections,
        }
    )
    detector = CabinObjectDetector(config)
    report["backend_status"] = detector.backend_status
    if class_map.status != "CLASS_MAP_READY":
        report["warnings"].append(class_map.status)
    if not report["model_exists"]:
        report["warnings"].append("MODEL_MISSING")
        report["parser_status"] = detector.backend_status
        return report
    frame = _load_frame(args)
    if frame is None:
        report["errors"].append("FRAME_NOT_AVAILABLE")
        report["parser_status"] = "FRAME_NOT_AVAILABLE"
        return report
    detections = detector.detect(frame, 0, context={"driver_roi_norm": [0.0, 0.0, 0.5, 1.0]})
    report["backend_status"] = detector.backend_status
    report["parser_status"] = detector.backend_status
    report["raw_output_shapes"] = getattr(detector, "last_raw_output_shapes", [])
    report["detections"] = [_jsonable(asdict(detection)) for detection in detections]
    report["parsed_detection_count"] = len(detections)
    if args.output_image:
        _write_output_image(frame, detections, args.output_image)
    return report


def _load_frame(args: argparse.Namespace):
    if args.image:
        return cv2.imread(args.image)
    if not args.video:
        return None
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.frame_index))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _write_output_image(frame, detections, path: str) -> None:
    out = frame.copy()
    height, width = out.shape[:2]
    for detection in detections:
        if not detection.bbox:
            continue
        x1, y1, x2, y2 = detection.bbox
        pt1 = int(x1 * width), int(y1 * height)
        pt2 = int(x2 * width), int(y2 * height)
        label = f"DET {detection.object_type.value} / {detection.relation_to_driver.value}"
        cv2.rectangle(out, pt1, pt2, (80, 220, 255), 2)
        cv2.putText(out, label, (pt1[0], max(18, pt1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 255), 1)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def write_report(report: dict[str, Any], path: str | None) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    report = inspect_model(args)
    write_report(report, args.output_json)
    print(report["backend_status"])


if __name__ == "__main__":
    main()
