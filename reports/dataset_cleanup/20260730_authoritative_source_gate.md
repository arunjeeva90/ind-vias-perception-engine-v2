# Authoritative model source gate — 2026-07-30

## Mandatory roots

| Task | Only permitted image root |
|---|---|
| Eye state | `/home/vicharak/Mobility_ADAS/ADVIS/DMS/DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/01_Eye_state_dataset` |
| Seat belt | `/home/vicharak/Mobility_ADAS/ADVIS/DMS/DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/02_Seat_Belt_detection` |
| Phone classifier/detector | `/home/vicharak/Mobility_ADAS/ADVIS/DMS/DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/03_Phone_detection` |

All 7,144 currently prepared rows resolve inside the corresponding root.
Preparation and training tooling now fail closed on a different handoff root,
cross-task path, legacy workspace dataset, or non-authoritative YOLO data file.

## Provenance and prepared splits

Overall prepared manifest SHA-256:
`510f29a06fbb8dd90c852636b3476e3b6fc83e2d92c36e819ce550002cb6ac4c`

| Task | Final manifest SHA-256 | Split SHA-256 | Train | Validation | Exclusions/errors |
|---|---|---|---:|---:|---|
| Eye state | `439806fe43abe06f9d7a5062b6c6bf93a18af99caffd2bee97359829bf82dca3` | `160a2a0e20ba04e9ca0da0facfb57b68595b26854e40bc43088bbfb8374ac61b` | 4,113 | 1,016 | 19 not recommended/uncertain |
| Seat belt | `735c2e54facfad2fc4e15492c086a7ce17f46022ddbed7f9e6ea4ce4bd806134` | `8e1ae40dd6e3b3508c3806449fbd25a94614e984de8995a2af81412d275e07b9` | 771 | 232 | 239 not recommended: 216 duplicate copies plus 23 uncertain |
| Phone classifier | `e78b9f809d243f7ae97be8eaaa50cdb89a78dfecda9f89b0baa7212867dc7bda` | `d9fb0ca4419e668c1bd0d13783c121b28f229a659a0a96a93299e16784ed72ca` | 550 | 130 | none |
| Phone detector | `0381f8da99648aa284b05dd06d0b1466e5385c38322f227204cb58804175a65c` | `92081333706e936dcf23947f4cd3901d4c3c209412f4b35462ed20728645c748` | 247 | 85 | one checksum-mismatched training hard negative excluded |

### Per-class counts

- Eye train: 375 closed, 3,738 open
- Eye validation: 70 closed, 946 open
- Seat-belt train: 364 off, 407 on
- Seat-belt validation: 54 off, 178 on
- Phone classifier train: 358 hard negative, 192 phone
- Phone classifier validation: 90 hard negative, 40 phone
- Phone detector train: 80 hard negative, 167 phone
- Phone detector validation: 20 hard negative, 65 phone

## Known conflicts before training

1. The phone detector manifest lists 81 training hard negatives, but one file
   fails SHA-256 verification. It remains excluded, leaving 80 verified
   training hard negatives.
2. Ultralytics is not installed in the current environments, so cabin-specific
   YOLO training cannot begin without installing it in an isolated training
   environment.
3. The existing COCO phone baseline has no colocated formal precision,
   recall, F1, mAP, or conversion report. It can be retained and evaluated, but
   prior metrics cannot be recovered from the available artifacts.
4. RKNN conversion and same-sample comparison must wait for a trained
   cabin-specific ONNX model that passes validation.

No new training, runtime model replacement, commit, or push has been performed
after applying this source gate.
