from pathlib import Path
import random
import shutil

SRC_DIRS = [
    Path("datasets/eye_state/train"),
    Path("datasets/eye_state_live/train"),
]

OUT_ROOT = Path("datasets/eye_state_finetune")

CLASSES = [
    "bad_crop",
    "eye_closed",
    "eye_open",
]

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

VAL_RATIO = 0.20
SEED = 42

random.seed(SEED)

# Clean output
if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

for split in ["train", "val"]:
    for cls in CLASSES:
        (OUT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

for cls in CLASSES:
    images = []

    for src_root in SRC_DIRS:
        cls_dir = src_root / cls
        if not cls_dir.exists():
            continue

        images.extend(
            [
                p for p in cls_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in EXTS
            ]
        )

    images = sorted(set(images))
    random.shuffle(images)

    if not images:
        print(f"[WARN] No images found for class: {cls}")
        continue

    n_val = max(1, int(len(images) * VAL_RATIO))
    val_images = images[:n_val]
    train_images = images[n_val:]

    for i, src in enumerate(train_images):
        dst = OUT_ROOT / "train" / cls / f"{cls}_{i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    for i, src in enumerate(val_images):
        dst = OUT_ROOT / "val" / cls / f"{cls}_{i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    print(
        f"{cls}: total={len(images)} train={len(train_images)} val={len(val_images)}"
    )

print("Done:", OUT_ROOT)
