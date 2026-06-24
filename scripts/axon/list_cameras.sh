#!/usr/bin/env bash
# list_cameras.sh
# List available cameras on AXON board
#
# Usage:
#   bash scripts/axon/list_cameras.sh

echo "========================================"
echo " AXON Camera Listing"
echo "========================================"
echo ""

# List /dev/video* devices
echo "--- /dev/video* devices ---"
DEVS=$(ls /dev/video* 2>/dev/null)
if [ -n "$DEVS" ]; then
    echo "$DEVS"
else
    echo "No /dev/video* devices found."
    echo ""
    echo "Possible causes:"
    echo "  - No USB camera connected"
    echo "  - Camera not recognized by kernel"
    echo "  - Permission issue (try: sudo chmod 666 /dev/video*)"
fi

echo ""

# v4l2-ctl --list-devices
echo "--- v4l2-ctl --list-devices ---"
if command -v v4l2-ctl &> /dev/null; then
    v4l2-ctl --list-devices 2>/dev/null || echo "v4l2-ctl --list-devices returned an error."
else
    echo "v4l2-ctl not available. Install with: sudo apt install v4l-utils"
fi

echo ""

# Check individual devices
echo "--- Device Details ---"
for i in 0 1 2 3 4 5; do
    DEV="/dev/video${i}"
    if [ -e "$DEV" ]; then
        echo ""
        echo "=== $DEV ==="
        if command -v v4l2-ctl &> /dev/null; then
            v4l2-ctl --device="$DEV" --all 2>/dev/null | head -30 || echo "  Could not query $DEV"
        else
            echo "  Device exists (v4l2-ctl not available for detailed info)"
        fi
    fi
done

echo ""
echo "========================================"
echo " Next Step"
echo "========================================"
echo ""
echo "  python apps/axon_camera_probe.py"
echo ""
echo "This will attempt to open each camera with OpenCV and capture a test frame."
echo ""
