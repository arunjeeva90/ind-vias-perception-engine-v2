from __future__ import annotations

from ind_vias_perception.common.types import BBox2D, Detection, ObjectClass


def group_compound_agents(
    detections: list[Detection],
    cfg: dict[str, object] | None = None,
) -> list[Detection]:
    cfg = cfg or {}
    if not cfg.get("enable_two_wheeler_grouping", False):
        return detections

    motorcycle_expand_scale = float(cfg.get("motorcycle_expand_scale", 1.6))
    min_iou = float(cfg.get("min_iou", 0.02))
    used: set[int] = set()
    grouped: list[Detection] = []

    for moto_idx, motorcycle in enumerate(detections):
        if moto_idx in used or motorcycle.label != ObjectClass.MOTORCYCLE:
            continue
        expanded_motorcycle = _expand_bbox(motorcycle.bbox, motorcycle_expand_scale)
        rider_idx = _best_rider_index(detections, used, expanded_motorcycle, min_iou)
        if rider_idx is None:
            continue
        rider = detections[rider_idx]
        used.add(moto_idx)
        used.add(rider_idx)
        grouped.append(_make_two_wheeler_agent(rider, motorcycle))

    if not grouped:
        return detections

    result = [det for idx, det in enumerate(detections) if idx not in used]
    result.extend(grouped)
    return result


def _best_rider_index(
    detections: list[Detection],
    used: set[int],
    expanded_motorcycle: BBox2D,
    min_iou: float,
) -> int | None:
    best_idx = None
    best_iou = 0.0
    for idx, det in enumerate(detections):
        if idx in used or det.label != ObjectClass.PEDESTRIAN:
            continue
        iou = _iou(expanded_motorcycle, det.bbox)
        if iou >= min_iou and iou > best_iou:
            best_idx = idx
            best_iou = iou
    return best_idx


def _make_two_wheeler_agent(rider: Detection, motorcycle: Detection) -> Detection:
    bbox = _union_bbox(rider.bbox, motorcycle.bbox)
    confidence = min(1.0, (rider.confidence + motorcycle.confidence) * 0.5)
    return Detection(
        bbox=bbox,
        label=ObjectClass.TWO_WHEELER_AGENT,
        confidence=confidence,
        metadata={
            "compound_source": "rider_motorcycle",
            "grouped_from": ["pedestrian", "motorcycle"],
            "rider_confidence": rider.confidence,
            "motorcycle_confidence": motorcycle.confidence,
        },
    )


def _expand_bbox(bbox: BBox2D, scale: float) -> BBox2D:
    cx = (bbox.x1 + bbox.x2) * 0.5
    cy = (bbox.y1 + bbox.y2) * 0.5
    half_w = bbox.width * scale * 0.5
    half_h = bbox.height * scale * 0.5
    return BBox2D(cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _union_bbox(a: BBox2D, b: BBox2D) -> BBox2D:
    return BBox2D(min(a.x1, b.x1), min(a.y1, b.y1), max(a.x2, b.x2), max(a.y2, b.y2))


def _iou(a: BBox2D, b: BBox2D) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    union = a.width * a.height + b.width * b.height - intersection
    if union <= 0:
        return 0.0
    return intersection / union
