from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - depends on runtime image
    psutil = None


class RuntimePerfMonitor:
    def __init__(
        self,
        *,
        model_gops_per_frame: float = 0.0,
        jsonl_path: str | None = None,
        compute_backend: str = "CPU",
        npu_active: bool = False,
    ) -> None:
        self.model_gops_per_frame = max(0.0, float(model_gops_per_frame or 0.0))
        self.compute_backend = str(compute_backend)
        self.npu_active = bool(npu_active)
        self._process_times: deque[float] = deque(maxlen=120)
        self._inference_times: deque[float] = deque(maxlen=120)
        self._display_times: deque[float] = deque(maxlen=120)
        self._capture_samples: deque[tuple[float, int]] = deque(maxlen=120)
        self._snapshot: dict[str, Any] = {}
        self._process = psutil.Process() if psutil is not None else None
        if psutil is not None:
            psutil.cpu_percent(interval=None)
        self._jsonl = None
        if jsonl_path is not None:
            output_path = Path(jsonl_path)
            if output_path.parent != Path("."):
                output_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl = open(output_path, "w", encoding="utf-8")

    def update(
        self,
        *,
        timestamp_s: float,
        frame_id: int,
        frame_latency_ms: float,
        inference_time_ms: float,
        overlay_time_ms: float,
        loop_time_ms: float,
        inference_ran: bool,
        captured_frames: int,
        processed_frames: int,
        dropped_frames: int,
        displayed: bool = False,
    ) -> dict[str, Any]:
        now = time.time()
        self._process_times.append(now)
        if inference_ran:
            self._inference_times.append(now)
        if displayed:
            self._display_times.append(now)
        self._capture_samples.append((now, max(0, int(captured_frames))))

        inference_fps_actual = self._rate(self._inference_times)
        # Approximate AI compute estimate, not a hardware counter. This is useful
        # for relative profiling and product reporting, not exact silicon use.
        estimated_tops = self.model_gops_per_frame * inference_fps_actual / 1000.0
        cpu_percent = None
        ram_mb = None
        if self._process is not None:
            cpu_percent = psutil.cpu_percent(interval=None)
            ram_mb = self._process.memory_info().rss / (1024.0 * 1024.0)

        self._snapshot = {
            "ts": timestamp_s,
            "frame_id": frame_id,
            "capture_fps": self._capture_rate(),
            "processing_fps": self._rate(self._process_times),
            "inference_fps_actual": inference_fps_actual,
            "display_fps": self._rate(self._display_times),
            "frame_latency_ms": frame_latency_ms,
            "feature_latency_ms": frame_latency_ms,
            "inference_time_ms": inference_time_ms,
            "overlay_time_ms": overlay_time_ms,
            "loop_time_ms": loop_time_ms,
            "cpu_percent": cpu_percent,
            "ram_mb": ram_mb,
            "compute_backend": self.compute_backend,
            "npu_active": self.npu_active,
            # No RKNN performance-counter integration exists in this runtime.
            # Never present the GOPS-derived workload estimate as NPU usage.
            "npu_tops_utilized": None,
            "estimated_tops": estimated_tops,
            "captured_frames": captured_frames,
            "processed_frames": processed_frames,
            "dropped_frames": dropped_frames,
            "inference_ran": inference_ran,
        }
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return dict(self._snapshot)

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        if not self._snapshot:
            return frame
        lines = [
            f"DMS FPS: {self._snapshot.get('processing_fps', 0.0):.1f}",
            f"Inf FPS: {self._snapshot.get('inference_fps_actual', 0.0):.1f}",
            f"Latency: {self._snapshot.get('frame_latency_ms', 0.0):.0f} ms",
        ]
        cpu_percent = self._snapshot.get("cpu_percent")
        ram_mb = self._snapshot.get("ram_mb")
        if cpu_percent is not None:
            lines.append(f"CPU: {cpu_percent:.0f}%")
        if ram_mb is not None:
            lines.append(f"RAM: {ram_mb:.0f} MB")
        lines.extend(
            [
                f"Est TOPS: {self._snapshot.get('estimated_tops', 0.0):.2f}",
                f"Drop: {self._snapshot.get('dropped_frames', 0)}",
            ]
        )
        x, y = 12, 22
        line_h = 18
        panel_w = 178
        panel_h = line_h * len(lines) + 12
        cv2.rectangle(frame, (x - 6, y - 16), (x + panel_w, y - 16 + panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 6, y - 16), (x + panel_w, y - 16 + panel_h), (70, 170, 255), 1)
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x, y + index * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
        return frame

    def write_jsonl(self, payload: dict[str, Any] | None = None) -> None:
        if self._jsonl is None:
            return
        self._jsonl.write(json.dumps(payload or self._snapshot, sort_keys=True) + "\n")

    def close(self) -> None:
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None

    @staticmethod
    def _rate(samples: deque[float]) -> float:
        if len(samples) < 2:
            return 0.0
        elapsed = samples[-1] - samples[0]
        if elapsed <= 0:
            return 0.0
        return (len(samples) - 1) / elapsed

    def _capture_rate(self) -> float:
        if len(self._capture_samples) < 2:
            return 0.0
        start_ts, start_count = self._capture_samples[0]
        end_ts, end_count = self._capture_samples[-1]
        elapsed = end_ts - start_ts
        if elapsed <= 0:
            return 0.0
        return max(0.0, float(end_count - start_count) / elapsed)
