from __future__ import annotations

from typing import Protocol, Sequence
import numpy as np
from .types import Detection, FramePacket, SceneQuality


class BackboneProtocol(Protocol):
    name: str
    def forward(self, frame: np.ndarray) -> dict[str, np.ndarray]: ...


class NeckProtocol(Protocol):
    name: str
    def forward(self, features: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


class HeadProtocol(Protocol):
    name: str
    def forward(self, features: dict[str, np.ndarray], packet: FramePacket) -> object: ...


class TrackerProtocol(Protocol):
    def update(self, detections: Sequence[Detection], timestamp_s: float) -> list[Detection]: ...


class SceneQualityHeadProtocol(Protocol):
    def forward(self, features: dict[str, np.ndarray], packet: FramePacket) -> SceneQuality: ...
