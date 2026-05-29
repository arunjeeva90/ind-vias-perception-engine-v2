from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2  # noqa: E402

from ind_vias_dms.core.config import load_dms_config  # noqa: E402
from ind_vias_dms.core.pipeline import DMSPipeline  # noqa: E402
from ind_vias_dms.interface.dms_packet import serialize_dms_state  # noqa: E402
from ind_vias_dms.utils.jsonl_writer import JSONLWriter  # noqa: E402
from ind_vias_dms.utils.video_io import make_video_writer, open_video_source, resize_to_width  # noqa: E402
from ind_vias_dms.visualization.overlay import OverlayRenderer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone IND-VIAS DualSight DMS v0.1.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", default=None)
    source.add_argument("--camera", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--config", default="configs/dms/dualsight_dms_v0_1.yaml")
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--display", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_dms_config(args.config)
    cap = open_video_source(args.video, args.camera)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = config.output_fps

    try:
        pipeline = DMSPipeline(config)
    except RuntimeError as exc:
        cap.release()
        raise SystemExit(str(exc)) from exc
    overlay = OverlayRenderer()
    jsonl = JSONLWriter(args.jsonl)
    writer = None
    frame_id = 0

    try:
        while args.max_frames is None or frame_id < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_to_width(frame, config.frame_resize_width)
            timestamp_ms = int((frame_id / fps) * 1000)
            state, context = pipeline.process(frame, timestamp_ms, frame_id)
            jsonl.write(serialize_dms_state(state))

            annotated = frame
            if args.debug_overlay or config.overlay_enabled:
                annotated = overlay.draw(
                    frame,
                    state,
                    context["face"],
                    context["head_pose"],
                    float(context["fps"]),
                    telemetry_enabled=config.telemetry_panel_enabled,
                )
            if args.output is not None:
                if writer is None:
                    writer = make_video_writer(args.output, fps, annotated)
                writer.write(annotated)
            if args.display:
                cv2.imshow("IND-VIAS DualSight DMS", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_id += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        jsonl.close()
        pipeline.close()
        if args.display:
            cv2.destroyAllWindows()

    if args.output is not None:
        print(f"Wrote overlay video: {Path(args.output)}")
    if args.jsonl is not None:
        print(f"Wrote DMS JSONL: {Path(args.jsonl)}")


if __name__ == "__main__":
    main()
