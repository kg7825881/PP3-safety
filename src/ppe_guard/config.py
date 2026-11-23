"""Typed configuration loaded from YAML.

Why pydantic instead of a plain dict?
  * Typos fail loudly at startup ("conf_threshold: 1.5" -> validation error),
    not silently mid-stream.
  * One source of truth for defaults, readable in code and in configs/*.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Device = Literal["auto", "cpu", "mps"]


class ModelConfig(BaseModel):
    weights: str = Field(
        "weights/yolo26n.pt",
        description="Local path; official model names are auto-downloaded to that path",
    )
    device: Device = "auto"
    image_size: int = Field(640, ge=32, description="Square input size; must be a multiple of 32")
    conf_threshold: float = Field(0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0, description="NMS overlap threshold")
    classes: list[int] | None = Field(
        None, description="Keep only these class ids (None = all)"
    )
    half: bool = Field(False, description="FP16 inference (ignored on CPU)")


class SourceConfig(BaseModel):
    uri: str = Field("0", description="Video file path, RTSP/HTTP URL, or webcam index")
    max_frames: int | None = Field(None, ge=1, description="Stop after N frames (None = all)")


class OutputConfig(BaseModel):
    display: bool = True
    save_path: str | None = None


class AppConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    source: SourceConfig = SourceConfig()
    output: OutputConfig = OutputConfig()


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from YAML; missing keys fall back to defaults."""
    if path is None:
        return AppConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return AppConfig.model_validate(data)
