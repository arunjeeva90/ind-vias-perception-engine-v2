from __future__ import annotations

import math

from ind_vias_perception.common.types import Detection


def evaluate_distance_quality(
    det: Detection,
    image_width: int,
    image_height: int,
    horizon_y: float,
    cfg: dict[str, object] | None = None,
) -> tuple[float, float, bool, list[str]]:
    cfg = cfg or {}
    reasons: list[str] = []
    confidence = 1.0
    near_horizon_margin = float(cfg.get("near_horizon_margin_px", 40))
    ground = float(det.metadata.get("distance_ground_m", float("inf")))
    semantic = float(det.metadata.get("distance_semantic_m", float("inf")))
    ratio = _agreement_ratio(ground, semantic)
    area_ratio = det.bbox.width * det.bbox.height / max(image_width * image_height, 1)
    min_area = _min_area_for(det, cfg)
    v_gc = float(det.metadata.get("v_gc", det.bbox.y2))
    is_side_object = not bool(det.metadata.get("in_ego_corridor", False))
    is_near_boundary = _near_boundary(det, image_width, image_height)
    is_tiny_bbox = area_ratio < min_area

    if is_side_object:
        reasons.append("side_object")
        confidence *= float(cfg.get("side_object_confidence_scale", 0.5))

    if is_near_boundary:
        reasons.append("near_boundary")
        confidence *= 0.65

    if is_tiny_bbox:
        reasons.append("tiny_bbox")
        confidence *= 0.35

    if v_gc <= horizon_y + near_horizon_margin:
        reasons.append("near_horizon")
        confidence *= 0.45

    if not math.isfinite(ground):
        reasons.append("non_finite_ground_distance")
    if not math.isfinite(semantic):
        reasons.append("non_finite_semantic_distance")
    if math.isfinite(ratio) and ratio > float(cfg.get("disagreement_ratio_limit", 2.0)):
        reasons.append("distance_disagreement")
        confidence *= 0.55

    confidence = max(0.0, min(1.0, confidence))
    if confidence < 0.35:
        reasons.append("low_distance_confidence")
    _add_outside_limits_reason(det, cfg, reasons)
    target_relevance = confidence * (1.0 if det.metadata.get("in_ego_corridor", False) else 0.45)
    valid = confidence >= 0.35 and "tiny_bbox" not in reasons and "near_horizon" not in reasons
    reason_codes = reasons or ["ok"]
    det.metadata["distance_reason_codes"] = ",".join(reason_codes)
    det.metadata["ground_semantic_ratio"] = ratio
    det.metadata["bbox_area_ratio"] = area_ratio
    det.metadata["ground_contact_row"] = v_gc
    det.metadata["horizon_y"] = float(horizon_y)
    det.metadata["near_horizon_margin_px"] = near_horizon_margin
    det.metadata["is_side_object"] = is_side_object
    det.metadata["is_near_boundary"] = is_near_boundary
    det.metadata["is_tiny_bbox"] = is_tiny_bbox
    return confidence, target_relevance, valid, reasons or ["ok"]


def _near_boundary(det: Detection, image_width: int, image_height: int) -> bool:
    margin_x = 0.02 * image_width
    margin_y = 0.02 * image_height
    return (
        det.bbox.x1 <= margin_x
        or det.bbox.x2 >= image_width - 1 - margin_x
        or det.bbox.y1 <= margin_y
        or det.bbox.y2 >= image_height - 1 - margin_y
    )


def _min_area_for(det: Detection, cfg: dict[str, object]) -> float:
    raw = cfg.get("min_bbox_area_ratio", {})
    if not isinstance(raw, dict):
        return 0.001
    if det.label.value == "bicycle" and "cyclist" in raw:
        return float(raw["cyclist"])
    return float(raw.get(det.label.value, 0.001))


def _agreement_ratio(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0 or b <= 0:
        return float("inf")
    return max(a, b) / min(a, b)


def _add_outside_limits_reason(det: Detection, cfg: dict[str, object], reasons: list[str]) -> None:
    distance = det.metadata.get("distance_bumper_m", det.distance_m)
    if distance is None:
        return
    distance = float(distance)
    min_distance = cfg.get("min_distance_m")
    max_distance = cfg.get("max_distance_m")
    if min_distance is not None and distance < float(min_distance):
        reasons.append("outside_distance_limits")
    if max_distance is not None and distance > float(max_distance):
        reasons.append("outside_distance_limits")
