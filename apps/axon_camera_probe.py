#!/usr/bin/env python3
"""
axon_camera_probe.py
OpenCV camera probe for Vicharak AXON board.

Tries camera indices 0 to max_index, reports success/failure for each,
prints resolution and FPS if available, and saves a sample frame.

Usage:
    python apps/axon_camera_probe.py
    python apps/axon_camera_probe.py --max-index 3
    python apps/axon_camera_probe.py --output-dir outputs/axon_camera_probe

Exit codes:
    0 - at least one camera is readable
    1 - no cameras could be opened
"""

import argparse
import os
import sys

import cv2


def probe_camera(index: int, output_dir: str) -> bool:
    """Attempt to open a camera and capture one frame.

    Args:
        index: Camera device index.
        output_dir: Directory to save sample frames.

    Returns:
        True if camera was successfully opened and a frame was captured.
    """
    print(f"\n--- Camera {index} (/dev/video{index}) ---")

    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print(f"  [FAIL] Cannot open camera {index}")
        return False

    # Read one frame
    ret, frame = cap.read()

    if not ret or frame is None:
        print(f"  [FAIL] Camera {index} opened but cannot read frame")
        cap.release()
        return False

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  [PASS] Camera {index} is readable")
    print(f"         Resolution: {width}x{height}")
    print(f"         FPS:        {fps:.1f}")
    print(f"         Frame shape: {frame.shape}")

    # Save sample frame
    os.makedirs(output_dir, exist_ok=True)
    frame_path = os.path.join(output_dir, f"camera_{index}.jpg")
    cv2.imwrite(frame_path, frame)
    print(f"         Saved:      {frame_path}")

    cap.release()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="AXON Camera Probe - test available cameras with OpenCV"
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=5,
        help="Maximum camera index to probe (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/axon_camera_probe",
        help="Directory to save sample frames (default: outputs/axon_camera_probe)",
    )
    args = parser.parse_args()

    print("========================================")
    print(" AXON Camera Probe")
    print("========================================")
    print(f"  OpenCV version: {cv2.__version__}")
    print(f"  Max index:      {args.max_index}")
    print(f"  Output dir:     {args.output_dir}")

    success_count = 0
    for i in range(args.max_index + 1):
        if probe_camera(i, args.output_dir):
            success_count += 1

    print("\n========================================")
    print(f" Results: {success_count} camera(s) readable out of {args.max_index + 1} probed")
    print("========================================")

    if success_count > 0:
        print("\nNext step:")
        print("  bash scripts/axon/run_dms_webcam_axon.sh 0")
        print("")
        sys.exit(0)
    else:
        print("\n[ERROR] No cameras could be opened.")
        print("")
        print("Troubleshooting:")
        print("  - Check if a USB camera is connected")
        print("  - Check permissions: ls -la /dev/video*")
        print("  - Add user to video group: sudo usermod -aG video $USER")
        print("  - Try: sudo chmod 666 /dev/video0")
        print("")
        sys.exit(1)


if __name__ == "__main__":
    main()
