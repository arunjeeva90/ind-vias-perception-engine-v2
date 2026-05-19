from __future__ import annotations


def main() -> None:
    print("IND-VIAS Metric Monocular ADAS Perception Engine v2.1")
    print("Atomic layers:")
    for item in [
        "backbones/{mobilenetv4_hybrid, efficientnet_lite, mobilevit, efficientvit}",
        "necks/{fpn, bifpn}",
        "heads/{detection, lane, freespace, ground_contact, depth, uncertainty, scene_quality, tsr, dms}",
        "geometry/{calibration, ground_plane, scale_anchors, scale_fusion}",
        "temporal/{trackers, motion_models, ugtf}",
        "ttc/{depth_ttc, expansion_ttc, flow_ttc, fusion}",
        "safety/{sentinel_fsm, safety_gate, can}",
        "runtime/{cais, profiling, logging}",
        "deployment/{onnx, tidl, openvx}",
    ]:
        print(f" - {item}")


if __name__ == "__main__":
    main()
