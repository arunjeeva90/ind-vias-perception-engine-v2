# Excluded phone hard-negative checksum investigation — 2026-07-30

## Exact record

- Package-relative path:
  `01_IMAGES/03_Phone_detection/object_detection_yolo/images/train/WIN_20260728_11_33_25_Pro_t0000005000_f0000150.jpg`
- Expected SHA-256:
  `e462f6ef3e3e01137b322847b86ddbb9ce80bc851334de306b0d1f842db67721`
- Actual SHA-256:
  `6fa96089a6ddd518f211eaa896c08fe2a1d129a11e0f44530b961585718b2c3c`
- File size: 253,416 bytes, matching the manifest
- Dimensions/channels: 1920×1080, 3-channel RGB/BGR
- Decode status: valid
  - OpenCV: decoded as `(1080, 1920, 3)`, `uint8`
  - Pillow: JPEG verification passed
  - JPEG markers: valid SOI `FFD8` and EOI `FFD9`
- Source video: `WIN_20260728_11_33_25_Pro.mp4`
- Timestamp: 5,000 ms
- Frame number: 150
- Final reviewed label: `hard_negative`
- Manual reviewed/recommended: true/true
- YOLO label: present and intentionally empty (0 bytes)

## Copy/provenance result

The current handoff contains only one file with this basename. A content-hash
scan of every file under `01_IMAGES` found:

- one copy with the actual hash, at the path above;
- zero copies with the expected hash;
- no renamed second copy with either hash.

The manifest reports two historical source matches under nested 2026-07-29
handoff locations, but those historical payload paths are not present in the
current package. The original source video is also not included locally, so
frame 150 cannot be re-extracted for conclusive byte-level restoration.

## Visual review

The frame decodes to a coherent cabin image. The driver is visible and wearing
a seat belt. No clearly identifiable mobile phone is visible in the driver
area, so the reviewed hard-negative label is visually plausible. A partially
visible dark rectangular object at the extreme lower-right boundary is
ambiguous and cannot restore integrity or provenance.

## Likely cause

The final reviewed manifest, package inventory, and `SHA256SUMS.txt` all agree
on the expected hash and size, while the packaged bytes produce a different
hash at the same size. This is most consistent with the image being changed or
replaced after inventory/hash generation, or an archive/copy-stage byte
alteration that preserved file length. The available evidence cannot
distinguish those causes conclusively.

## Recommended handling

Continue excluding this file from all training, calibration, validation, and
benchmarking. Do not repair the manifest to match the current bytes. Restore it
only if an original historical source copy or source video independently
reproduces the expected content and provenance. Its empty YOLO label remains
untouched.
