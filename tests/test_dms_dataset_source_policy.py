from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparer_accepts_only_task_specific_authoritative_root():
    prepare = _load_tool(
        "prepare_handoff_source_policy",
        "tools/dms_models/prepare_handoff_datasets.py",
    )
    root = prepare.AUTHORITATIVE_HANDOFF_ROOT.resolve()
    allowed = root / "01_IMAGES/01_Eye_state_dataset/open/example.png"

    prepare.require_task_source("eye_state", allowed, root)

    with pytest.raises(ValueError, match="outside authoritative"):
        prepare.require_task_source(
            "eye_state",
            REPO_ROOT / "datasets/eye_state/train/eye_open/example.png",
            root,
        )


def test_preparer_rejects_cross_task_source():
    prepare = _load_tool(
        "prepare_handoff_cross_task_policy",
        "tools/dms_models/prepare_handoff_datasets.py",
    )
    root = prepare.AUTHORITATIVE_HANDOFF_ROOT.resolve()
    phone_image = root / "01_IMAGES/03_Phone_detection/example.jpg"

    with pytest.raises(ValueError, match="outside authoritative"):
        prepare.require_task_source("seat_belt", phone_image, root)


def test_legacy_eye_dataset_builder_is_disabled():
    legacy = _load_tool(
        "legacy_eye_builder_policy",
        "tools/eyenetrknn/build_finetune_dataset.py",
    )

    with pytest.raises(RuntimeError, match="Legacy EyeNet dataset mixing is disabled"):
        legacy.main()


def test_phone_training_declares_distinct_authoritative_source():
    phone = _load_tool(
        "phone_training_source_policy",
        "tools/dms_models/train_phone_yolo.py",
    )

    assert phone.AUTHORITATIVE_PHONE_ROOT == Path(
        "/home/vicharak/Mobility_ADAS/ADVIS/DMS/"
        "DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/03_Phone_detection"
    )
    assert "phone_yolo" in str(phone.PREPARED_DATA)
    assert phone.CABIN_RUN_NAME == "cabin_specific_phone_yolov8n_20260730"
