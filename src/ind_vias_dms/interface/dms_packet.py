from __future__ import annotations

import json

from ind_vias_dms.core.types import DMSState


def serialize_dms_state(state: DMSState) -> dict[str, object]:
    return state.to_dict()


def dumps_dms_state(state: DMSState) -> str:
    return json.dumps(serialize_dms_state(state), sort_keys=True)
