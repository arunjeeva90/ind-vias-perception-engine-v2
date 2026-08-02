from __future__ import annotations

import importlib
import sys
import types
from typing import Any


def load_mediapipe_solutions() -> Any:
    """Load MediaPipe Solutions without the unused Tasks/audio import.

    MediaPipe 0.10 imports ``mediapipe.tasks.python`` from its package
    ``__init__``. That transitively imports audio/sound-device support and can
    block on an embedded board with no audio device. This DMS uses only the
    stable ``mp.solutions`` face/hand APIs, so a minimal Tasks namespace is
    installed before the first MediaPipe import.

    The change is process-local and does not patch the installed package. Code
    that needs MediaPipe Tasks must run separately without this loader.
    """

    if "mediapipe" not in sys.modules:
        tasks_module = types.ModuleType("mediapipe.tasks")
        tasks_python_module = types.ModuleType("mediapipe.tasks.python")
        tasks_module.python = tasks_python_module
        sys.modules.setdefault("mediapipe.tasks", tasks_module)
        sys.modules.setdefault("mediapipe.tasks.python", tasks_python_module)
    mediapipe = importlib.import_module("mediapipe")
    if not hasattr(mediapipe, "solutions"):
        raise ImportError("Installed MediaPipe package does not expose the Solutions API")
    return mediapipe
