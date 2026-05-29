from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JSONLWriter:
    def __init__(self, path: str | None) -> None:
        self.file = None
        if path is not None:
            output_path = Path(path)
            if output_path.parent != Path("."):
                output_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(output_path, "w", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        if self.file is not None:
            self.file.write(json.dumps(payload, sort_keys=True) + "\n")

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
