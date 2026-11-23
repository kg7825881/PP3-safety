"""Video input/output.

One class handles every source type, because OpenCV's VideoCapture already
accepts files, webcams (integer index) and network streams (rtsp://, http://).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class VideoSource:
    def __init__(self, uri: str, max_frames: int | None = None) -> None:
        # "0", "1", ... means a webcam index; anything else is a path or URL.
        target: int | str = int(uri) if uri.isdigit() else uri
        if isinstance(target, str) and "://" not in target and not Path(target).exists():
            raise FileNotFoundError(f"Video source not found: {target}")
        self.cap = cv2.VideoCapture(target)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {uri}")
        self.max_frames = max_frames

    @property
    def fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0  # webcams often report 0

    @property
    def size(self) -> tuple[int, int]:
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        count = 0
        while self.max_frames is None or count < self.max_frames:
            ok, frame = self.cap.read()
            if not ok:
                break  # end of file or stream dropped
            count += 1
            yield frame

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VideoSink:
    """Writes annotated frames to an .mp4 file."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, size)

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()
