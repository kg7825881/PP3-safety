"""Compute-device selection.

On an M-series Mac, PyTorch can run on:
  * "cpu" - the performance/efficiency cores
  * "mps" - Metal Performance Shaders, i.e. the integrated GPU
The Apple Neural Engine (ANE) is NOT reachable from PyTorch; we reach it later
by exporting to CoreML.
"""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> str:
    """Turn "auto" into the best available device and validate explicit choices."""
    mps_ok = torch.backends.mps.is_available()
    if requested == "auto":
        return "mps" if mps_ok else "cpu"
    if requested == "mps" and not mps_ok:
        raise RuntimeError("MPS requested but not available (needs Apple Silicon + macOS 12.3+)")
    return requested
