#!/usr/bin/env python3
"""Train the one-class phone detector from the verified materialized dataset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np


AUTHORITATIVE_PHONE_ROOT = Path(
    "/home/vicharak/Mobility_ADAS/ADVIS/DMS/"
    "DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/03_Phone_detection"
)
PREPARED_DATA = Path(
    "local_data/dms_handoff_20260730/phone_yolo/data.yaml"
)
CABIN_RUN_NAME = "cabin_specific_phone_yolov8n_20260730"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PREPARED_DATA)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/dms_phone_cabin_reviewed_20260730"),
    )
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Prepared YOLO data.yaml not found: {args.data}")
    required_data = PREPARED_DATA.resolve()
    if args.data.resolve() != required_data:
        raise ValueError(
            f"Refusing non-authoritative YOLO data file {args.data.resolve()}; "
            f"required {required_data}"
        )
    dataset_root = args.data.resolve().parent
    authoritative_root = AUTHORITATIVE_PHONE_ROOT.resolve()
    for kind in ("images", "labels"):
        paths = sorted((dataset_root / kind).glob("*/*"))
        if not paths:
            raise ValueError(f"Prepared phone YOLO {kind} are missing")
        invalid = [
            path
            for path in paths
            if authoritative_root not in path.resolve().parents
        ]
        if invalid:
            raise ValueError(
                f"Prepared phone YOLO {kind} escape {authoritative_root}: "
                f"{invalid[:3]}"
            )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install it in an isolated training "
            "environment, then rerun this command; the runtime does not depend on it."
        ) from exc
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = YOLO(args.base_model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        seed=args.seed,
        deterministic=True,
        single_cls=True,
        device=args.device,
        project=str(args.output_dir.resolve()),
        name=CABIN_RUN_NAME,
        exist_ok=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
