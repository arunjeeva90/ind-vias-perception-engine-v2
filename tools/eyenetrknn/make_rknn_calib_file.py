from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


AUTHORITATIVE_ROOTS = {
    "eye_state": Path(
        "/home/vicharak/Mobility_ADAS/ADVIS/DMS/"
        "DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/01_Eye_state_dataset"
    ),
    "seat_belt": Path(
        "/home/vicharak/Mobility_ADAS/ADVIS/DMS/"
        "DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/02_Seat_Belt_detection"
    ),
    "phone_classifier": Path(
        "/home/vicharak/Mobility_ADAS/ADVIS/DMS/"
        "DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/03_Phone_detection"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Verified prepared manifest; only matching training rows are used",
    )
    parser.add_argument(
        "--task", choices=sorted(AUTHORITATIVE_ROOTS), default="eye_state"
    )
    parser.add_argument("--out", type=str, default="models/eyenetrknn/rknn_calib_dataset.txt")
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def main():
    args = parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, newline="", encoding="utf-8") as stream:
        images = [
            Path(row["image_path"]).resolve()
            for row in csv.DictReader(stream)
            if row["task"] == args.task and row["split"] == "train"
        ]
    allowed_root = AUTHORITATIVE_ROOTS[args.task].resolve()
    invalid = [path for path in images if allowed_root not in path.parents]
    if invalid:
        raise ValueError(
            f"Calibration contains paths outside {allowed_root}: {invalid[:3]}"
        )
    if not images:
        raise ValueError(f"No authoritative training rows found for {args.task}")

    random.Random(args.seed).shuffle(images)
    images = images[: args.max_images]

    with open(out_path, "w") as f:
        for p in images:
            f.write(str(p.resolve()) + "\n")

    print(f"Wrote {len(images)} images to {out_path}")


if __name__ == "__main__":
    main()
