from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_onnx_detector.py"
    spec = importlib.util.spec_from_file_location("check_onnx_detector", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_onnx_detector_fails_when_onnx_backend_model_is_missing(tmp_path, capsys):
    module = _load_script_module()
    missing_model = tmp_path / "missing_detector.onnx"
    missing_model_text = str(missing_model).replace("\\", "/")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
camera:
  fx_px: 1100.0
  fy_px: 1100.0
  cx_px: 640.0
  cy_px: 360.0
  height_m: 1.25
  pitch_deg: 2.0
  horizon_v_px: 374.0
detection:
  backend: onnx
  onnx_model_path: "{missing_model_text}"
  input_size: [640, 640]
  class_names:
    0: car
""",
        encoding="utf-8",
    )

    exit_code = module.main(["--config", str(config)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "detection.backend=onnx" in captured.out
    assert "ONNX detector model is missing" in captured.out
    assert missing_model_text in captured.out.replace("\\", "/")
