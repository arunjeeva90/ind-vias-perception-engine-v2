from pathlib import Path
from PIL import Image
import sys

roots = [
    Path("datasets/eye_state_finetune"),
    Path("datasets/eye_state_live"),
    Path("datasets/eye_state"),
]

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

bad = []

for root in roots:
    if not root.exists():
        continue

    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue

        try:
            with Image.open(p) as img:
                img.verify()

            # verify() does not fully decode, so reopen and load
            with Image.open(p) as img:
                img.convert("RGB").load()

        except Exception as e:
            bad.append((p, str(e)))

print(f"Bad images found: {len(bad)}")

for p, err in bad:
    print(f"{p} :: {err}")

if bad:
    Path("bad_images_list.txt").write_text(
        "\n".join(str(p) for p, _ in bad) + "\n"
    )
    print("Written: bad_images_list.txt")
    sys.exit(1)

print("All images OK")
