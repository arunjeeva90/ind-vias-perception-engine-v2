from __future__ import annotations

import pytest

from ind_vias_perception.apps.run_demo import build_parser, validate_detector_config


def test_cli_parser_accepts_image_video_output_show_and_max_frames():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--image",
            "frame.jpg",
            "--video",
            "clip.mp4",
            "--output",
            "annotated.mp4",
            "--show",
            "--debug-overlay",
            "--max-frames",
            "7",
        ]
    )

    assert args.image == "frame.jpg"
    assert args.video == "clip.mp4"
    assert args.output == "annotated.mp4"
    assert args.show is True
    assert args.debug_overlay is True
    assert args.max_frames == 7


def test_onnx_detector_config_missing_model_has_clear_error(tmp_path):
    missing_model = tmp_path / "detector.onnx"

    with pytest.raises(SystemExit) as exc:
        validate_detector_config(
            {
                "detection": {
                    "backend": "onnx",
                    "onnx_model_path": str(missing_model),
                }
            }
        )

    message = str(exc.value)
    assert "ONNX detector model is missing" in message
    assert "models/weights/detector.onnx" in message
    assert "configs/default.yaml" in message
