"""Core data types passed between pipeline stages.

Every stage (detector -> tracker -> rules -> API) speaks in these types instead of
raw framework objects. That keeps stages swappable: the detector can move from
PyTorch to CoreML without the tracker or API noticing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object in one frame.

    Box format is xyxy in pixel coordinates of the ORIGINAL frame
    (x1, y1) = top-left, (x2, y2) = bottom-right.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass(slots=True)
class StageTimings:
    """Per-frame latency breakdown in milliseconds.

    Splitting latency into stages is what lets us optimize the RIGHT thing:
    if postprocess (NMS) dominates, a faster backbone won't help.
    """

    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass(slots=True)
class FrameResult:
    """Everything the pipeline knows about a single frame."""

    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    timings: StageTimings = field(default_factory=StageTimings)
