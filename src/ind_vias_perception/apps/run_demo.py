from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ind_vias_perception.common.types import FramePacket
from ind_vias_perception.config.settings import load_settings
from ind_vias_perception.pipeline.factory import build_pipeline
from ind_vias_perception.apps.visualization import draw_perception_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--debug-overlay", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    settings = load_settings(args.config)
    detection_backend = settings.raw.get("detection", {}).get("backend", "dummy")
    ego_corridor = settings.raw.get("ego_corridor", {})
    validate_detector_config(settings.raw)
    pipeline = build_pipeline(settings)
    if hasattr(pipeline.detection_head, "debug"):
        pipeline.detection_head.debug = args.debug_overlay

    if args.image is not None:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read image: {args.image}")
        out = pipeline.process(FramePacket(frame=frame, timestamp_s=0.0, frame_id=0))
        annotated = draw_perception_output(
            frame,
            out,
            detection_backend=detection_backend,
            debug_overlay=args.debug_overlay,
            ego_corridor=ego_corridor,
        )
        print(out.safety_payload)
        if args.output is not None:
            if not cv2.imwrite(args.output, annotated):
                raise SystemExit(f"Could not write image: {args.output}")
        if args.show:
            cv2.imshow("IND-VIAS Perception", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    if args.video is None:
        import numpy as np
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(args.max_frames or 3):
            out = pipeline.process(FramePacket(frame=frame, timestamp_s=i / 30.0, frame_id=i))
            print(out.safety_payload)
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not read video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    writer = None
    i = 0
    while args.max_frames is None or i < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        out = pipeline.process(FramePacket(frame=frame, timestamp_s=i / fps, frame_id=i))
        annotated = draw_perception_output(
            frame,
            out,
            detection_backend=detection_backend,
            debug_overlay=args.debug_overlay,
            ego_corridor=ego_corridor,
        )
        print(out.safety_payload)
        if args.output is not None:
            if writer is None:
                height, width = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
                if not writer.isOpened():
                    cap.release()
                    raise SystemExit(f"Could not write video: {args.output}")
            writer.write(annotated)
        if args.show:
            cv2.imshow("IND-VIAS Perception", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        i += 1
    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()


def validate_detector_config(raw_settings: dict[str, object]) -> None:
    detection_cfg = raw_settings.get("detection", {})
    if not isinstance(detection_cfg, dict):
        return
    if detection_cfg.get("backend", "dummy") != "onnx":
        return

    model_path = Path(str(detection_cfg.get("onnx_model_path", "models/weights/detector.onnx")))
    if model_path.exists():
        return

    raise SystemExit(
        "ONNX detector model is missing. Place the ONNX model at "
        "models/weights/detector.onnx or switch back to configs/default.yaml."
    )


if __name__ == "__main__":
    main()
