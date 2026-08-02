"""Retained compatibility entry point for obsolete legacy eye folders."""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "Legacy eye folders were removed from the active workspace. Image "
        "readability, dimensions, channels, and SHA-256 are now verified by "
        "tools/dms_models/prepare_handoff_datasets.py."
    )


if __name__ == "__main__":
    main()
