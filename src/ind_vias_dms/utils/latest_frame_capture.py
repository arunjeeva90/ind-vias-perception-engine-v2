from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class LatestFrameCapture:
    """Threaded camera reader that keeps only the newest captured frame."""

    def __init__(self, cap: cv2.VideoCapture, *, debug: bool = False, max_consecutive_failures: int = 10) -> None:
        self.cap = cap
        self.debug = debug
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_timestamp_s = 0.0
        self._latest_frame_id = -1
        self._last_read_frame_id = -1
        self._last_error: str | None = None
        self._first_frame_logged = False
        self.captured_frames = 0
        self.overwritten_frames = 0

    def start(self) -> "LatestFrameCapture":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        if self.debug:
            print("Latest-frame capture thread started")
        return self

    def read(
        self,
        timeout_s: float = 0.05,
        after_frame_id: int | None = None,
    ) -> tuple[bool, np.ndarray | None, float, int]:
        deadline = time.time() + max(0.0, timeout_s)
        while True:
            with self._lock:
                if self._latest_frame is not None and (
                    after_frame_id is None or self._latest_frame_id != after_frame_id
                ):
                    frame = self._latest_frame.copy()
                    frame_id = self._latest_frame_id
                    timestamp_s = self._latest_timestamp_s
                    self._last_read_frame_id = frame_id
                    return True, frame, timestamp_s, frame_id
                last_error = self._last_error
                running = self._running
                if last_error is not None or not running:
                    return False, None, 0.0, -1
            if time.time() >= deadline:
                return False, None, 0.0, -1
            time.sleep(0.002)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def release(self) -> None:
        self.stop()
        self.cap.release()

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def dropped_frames(self) -> int:
        return self.overwritten_frames

    def _capture_loop(self) -> None:
        consecutive_failures = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    with self._lock:
                        self._last_error = (
                            "camera read failed "
                            f"{consecutive_failures} consecutive times"
                        )
                    self._running = False
                    break
                time.sleep(0.01)
                continue
            consecutive_failures = 0
            timestamp_s = time.time()
            with self._lock:
                if self._latest_frame is not None and self._latest_frame_id > self._last_read_frame_id:
                    self.overwritten_frames += 1
                self.captured_frames += 1
                self._latest_frame = frame
                self._latest_timestamp_s = timestamp_s
                self._latest_frame_id = self.captured_frames - 1
                frame_id = self._latest_frame_id
                should_log_first_frame = self.debug and not self._first_frame_logged
                self._first_frame_logged = True
            if should_log_first_frame:
                print(f"Latest-frame first frame received: shape={frame.shape}, frame_id={frame_id}")
