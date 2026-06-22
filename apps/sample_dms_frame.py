from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract one representative cabin frame for ONNX inspection.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", default=None)
    source.add_argument("--camera", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--time-ms", type=int, default=None)
    parser.add_argument("--output", required=True)
    return parser


def sample_frame(args: argparse.Namespace) -> tuple[bool, str, tuple[int, int, int] | None]:
    cap = cv2.VideoCapture(args.video if args.video is not None else args.camera)
    if not cap.isOpened():
        return False, "SOURCE_OPEN_FAILED", None
    try:
        if args.video is not None:
            if args.time_ms is not None:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0, args.time_ms))
            elif args.frame_index is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, "FRAME_READ_FAILED", None
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), frame):
            return False, "IMAGE_WRITE_FAILED", None
        return True, str(output_path), tuple(frame.shape)
    finally:
        cap.release()


def main() -> None:
    args = build_parser().parse_args()
    ok, message, shape = sample_frame(args)
    if not ok:
        print(message)
        raise SystemExit(1)
    print(f"Saved frame: {message}")
    print(f"Shape: {shape}")


if __name__ == "__main__":
    main()
