# Architecture v2.1 Atomic Layers

The repository intentionally splits the perception engine into atomic folders so that each block can evolve independently.

## Layer flow

```text
Camera frame
  -> preprocessing
  -> backbone
  -> neck / feature pyramid
  -> heads
  -> geometry anchors
  -> scale fusion
  -> temporal tracking / UGTF
  -> TTC fusion
  -> Sentinel FSM
  -> Safety gate
  -> CAN-ready outputs
```

## Backbone folders

- `mobilenetv4_hybrid/`: preferred production direction.
- `efficientnet_lite/`: safer first prototype.
- `mobilevit/`: optional hybrid candidate.
- `efficientvit/`: optional efficient transformer candidate.
- `adapters/`: ONNX/Torch/TIDL wrappers.

## Head folders

- `detection/`: vehicles, 2W, autos, pedestrians, animals.
- `lane/`: lane/road boundary.
- `freespace/`: drivable area.
- `ground_contact/`: predicts `(u_gc, v_gc)` for metric distance.
- `depth/`: sparse-dense hybrid depth.
- `uncertainty/`: `sigma_depth`, head confidence.
- `scene_quality/`: glare, rain, fog, night, occlusion, complexity.
- `tsr/`: traffic signs/signals.
- `dms/`: future driver monitoring heads.

## Production note

The present code uses deterministic dummy components. Replace one component at a time with trained inference adapters while keeping interfaces stable.
