from __future__ import annotations

import cv2
import numpy as np

from ind_vias_perception.common.types import BBox2D, PerceptionOutput


_BOX_COLOR = (0, 220, 0)
_TEXT_COLOR = (255, 255, 255)
_TEXT_BG_COLOR = (0, 0, 0)
_STATUS_BG_COLOR = (32, 32, 32)


def draw_perception_output(
    frame: np.ndarray,
    output: PerceptionOutput,
    detection_backend: str = "unknown",
    debug_overlay: bool = False,
    ego_corridor: dict[str, object] | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    if debug_overlay and ego_corridor:
        _draw_ego_corridor(annotated, ego_corridor)

    for det in output.detections:
        x1, y1, x2, y2 = _clamp_bbox(det.bbox, annotated.shape)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        label = f"{det.label.value} {det.confidence:.2f}"
        if debug_overlay and det.track_id is not None:
            label += f" id:{det.track_id}"
        if det.distance_m is not None:
            label += f" {det.distance_m:.1f}m"
        if det.ttc_s is not None:
            label += f" TTC {det.ttc_s:.1f}s"
        _draw_label(annotated, label, (x1, max(0, y1 - 8)))

    payload = output.safety_payload
    status_parts = [
        f"warning: {payload.get('warning_level', 'unknown')}",
        f"sentinel: {payload.get('sentinel_state', 'unknown')}",
        f"cais: {payload.get('cais_mode', output.mode)}",
    ]
    if debug_overlay:
        status_parts.insert(0, f"detection: {detection_backend}")
        status_parts.append(
            "ego_motion: "
            f"{output.scene_quality.ego_motion_state} "
            f"yaw={output.scene_quality.yaw_score:.2f} "
            f"conf={output.scene_quality.yaw_confidence:.2f} "
            f"confirm={output.scene_quality.turning_confirmation_count} "
            f"flow={output.scene_quality.flow_points}"
        )
        for det in output.detections[:3]:
            status_parts.append(
                "gc: "
                f"u={det.metadata.get('u_gc', 'n/a')} "
                f"v={det.metadata.get('v_gc', 'n/a')} "
                f"Dg={_format_float(det.metadata.get('distance_ground_m'))}m "
                f"Ds={_format_float(det.metadata.get('distance_semantic_m'))}m "
                f"Df={_format_float(det.metadata.get('distance_fused_camera_m'))}m "
                f"Dbump={_format_float(det.metadata.get('distance_bumper_m'))}m "
                f"src={det.metadata.get('distance_source', 'n/a')} "
                f"confD={_format_float(det.metadata.get('distance_confidence'))} "
                f"rel={_format_float(det.metadata.get('target_relevance'))} "
                f"valid={det.metadata.get('distance_valid_for_safety', 'n/a')} "
                f"reason={det.metadata.get('reason_codes', 'n/a')} "
                f"bbox_clipped={det.metadata.get('bbox_clipped', 'n/a')}"
            )
        status_parts.append(f"safety: {payload}")
        status_parts.append(
            "confirm: "
            f"raw={payload.get('raw_warning_level', 'n/a')} "
            f"confirmed={payload.get('confirmed_warning_level', 'n/a')} "
            f"count={payload.get('confirmation_count', 'n/a')}/"
            f"{payload.get('confirmation_required', 'n/a')}"
        )
    _draw_status(annotated, status_parts)
    return annotated


def _clamp_bbox(bbox: BBox2D, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x1 = int(max(0, min(width - 1, round(bbox.x1))))
    y1 = int(max(0, min(height - 1, round(bbox.y1))))
    x2 = int(max(0, min(width - 1, round(bbox.x2))))
    y2 = int(max(0, min(height - 1, round(bbox.y2))))
    return x1, y1, x2, y2


def _draw_label(frame: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - text_h - baseline - 4)
    right = min(frame.shape[1] - 1, x + text_w + 8)
    cv2.rectangle(frame, (x, top), (right, y + baseline), _TEXT_BG_COLOR, -1)
    cv2.putText(frame, text, (x + 4, y - 4), font, scale, _TEXT_COLOR, thickness, cv2.LINE_AA)


def _draw_status(frame: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    line_height = max((height for _, height in sizes), default=12) + 8
    panel_width = min(frame.shape[1] - 1, max((width for width, _ in sizes), default=0) + 16)
    panel_height = min(frame.shape[0] - 1, line_height * len(lines) + 8)
    cv2.rectangle(frame, (0, 0), (panel_width, panel_height), _STATUS_BG_COLOR, -1)
    for idx, line in enumerate(lines):
        y = 8 + line_height * (idx + 1) - 8
        cv2.putText(frame, line, (8, y), font, scale, _TEXT_COLOR, thickness, cv2.LINE_AA)


def _draw_ego_corridor(frame: np.ndarray, cfg: dict[str, object]) -> None:
    if not cfg.get("enabled", False):
        return
    height, width = frame.shape[:2]
    center_x = float(cfg.get("center_x_norm", 0.5)) * width
    top_y = float(cfg.get("top_y_norm", 0.45)) * height
    bottom_y = float(cfg.get("bottom_y_norm", 1.0)) * height
    top_w = float(cfg.get("top_width_norm", 0.18)) * width
    bottom_w = float(cfg.get("bottom_width_norm", 0.45)) * width
    polygon = np.array(
        [
            [center_x - top_w * 0.5, top_y],
            [center_x + top_w * 0.5, top_y],
            [center_x + bottom_w * 0.5, bottom_y],
            [center_x - bottom_w * 0.5, bottom_y],
        ],
        dtype=np.int32,
    )
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 180, 255), thickness=2)


def _format_float(value: object) -> str:
    if not isinstance(value, (float, int)):
        return "n/a"
    return f"{float(value):.1f}"
