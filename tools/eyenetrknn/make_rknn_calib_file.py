from __future__ import annotations

import argparse
import random
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="datasets/eye_state/train")
    parser.add_argument("--out", type=str, default="models/eyenetrknn/rknn_calib_dataset.txt")
    parser.add_argument("--max-images", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    random.seed(42)
    random.shuffle(images)
    images = images[: args.max_images]

    with open(out_path, "w") as f:
        for p in images:
            f.write(str(p.resolve()) + "\n")

    print(f"Wrote {len(images)} images to {out_path}")


if __name__ == "__main__":
    main()
