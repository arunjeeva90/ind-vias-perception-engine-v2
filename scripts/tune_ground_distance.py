from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ind_vias_perception.config.settings import load_settings  # noqa: E402


@dataclass(frozen=True)
class GroundDistanceTuning:
    u_gc: float
    v_gc: float
    distance_camera_m: float
    distance_bumper_m: float
    suggested_fy: float
    suggested_horizon_y: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune monocular ground-plane distance from a known bbox.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="configs/phone_demo_1440.yaml")
    parser.add_argument("--bbox", required=True, help="x1,y1,x2,y2")
    parser.add_argument("--known-distance-m", type=float, required=True)
    parser.add_argument("--camera-height-m", type=float, default=None)
    parser.add_argument("--horizon-y", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    bbox = parse_bbox(args.bbox)
    camera_height_m = args.camera_height_m or settings.camera.height_m
    horizon_y = args.horizon_y if args.horizon_y is not None else settings.camera.horizon_v_px
    fy = args.fy if args.fy is not None else settings.camera.fy_px
    offset_m = settings.vehicle.camera_to_front_bumper_offset_m

    result = tune_ground_distance(
        bbox=bbox,
        known_distance_m=args.known_distance_m,
        camera_height_m=camera_height_m,
        horizon_y=horizon_y,
        fy=fy,
        camera_to_front_bumper_offset_m=offset_m,
    )

    print(f"image: {args.image}")
    print(f"config: {args.config}")
    print(f"u_gc: {result.u_gc:.2f}")
    print(f"v_gc: {result.v_gc:.2f}")
    print(f"current estimated Dcam: {_format_distance(result.distance_camera_m)}")
    print(f"current estimated Dbump: {_format_distance(result.distance_bumper_m)}")
    print(f"suggested fy: {result.suggested_fy:.2f}")
    print(f"suggested horizon_y: {result.suggested_horizon_y:.2f}")
    return 0


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bbox must be x1,y1,x2,y2")
    return parts[0], parts[1], parts[2], parts[3]


def tune_ground_distance(
    bbox: tuple[float, float, float, float],
    known_distance_m: float,
    camera_height_m: float,
    horizon_y: float,
    fy: float,
    camera_to_front_bumper_offset_m: float = 0.0,
) -> GroundDistanceTuning:
    x1, _, x2, y2 = bbox
    u_gc = (x1 + x2) * 0.5
    v_gc = y2
    distance_camera_m = estimate_camera_distance_m(v_gc, camera_height_m, horizon_y, fy)
    distance_bumper_m = max(distance_camera_m - camera_to_front_bumper_offset_m, 0.0)
    suggested_fy = known_distance_m * (v_gc - horizon_y) / camera_height_m
    suggested_horizon_y = v_gc - (fy * camera_height_m / known_distance_m)
    return GroundDistanceTuning(
        u_gc=u_gc,
        v_gc=v_gc,
        distance_camera_m=distance_camera_m,
        distance_bumper_m=distance_bumper_m,
        suggested_fy=suggested_fy,
        suggested_horizon_y=suggested_horizon_y,
    )


def estimate_camera_distance_m(v_gc: float, camera_height_m: float, horizon_y: float, fy: float) -> float:
    if v_gc <= horizon_y:
        return float("inf")
    return fy * camera_height_m / (v_gc - horizon_y)


def _format_distance(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.2f} m"


if __name__ == "__main__":
    raise SystemExit(main())
