"""Retained compatibility entry point for the obsolete legacy dataset builder."""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "Legacy EyeNet dataset mixing is disabled. Use "
        "tools/dms_models/prepare_handoff_datasets.py; eye training may consume "
        "only the final-reviewed 2026-07-30 handoff manifest."
    )


if __name__ == "__main__":
    main()
