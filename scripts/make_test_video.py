"""Generate a small synthetic test video (a slow pan/zoom across a bundled image).

Used for smoke-testing the video pipeline without any external download.
It is NOT a substitute for real footage when judging accuracy.

    python scripts/make_test_video.py data/videos/test_pan.mp4
"""

import sys
from pathlib import Path

import cv2
from ultralytics.utils import ASSETS


def main(out: str = "data/videos/test_pan.mp4", frames: int = 150, size=(960, 540)) -> None:
    img = cv2.imread(str(ASSETS / "bus.jpg"))
    h, w = img.shape[:2]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    for i in range(frames):
        t = i / (frames - 1)
        crop_h = int(h * (0.9 - 0.3 * t))           # zoom in over time
        crop_w = int(crop_h * size[0] / size[1])
        crop_w = min(crop_w, w)
        x = int((w - crop_w) * t)                   # pan left -> right
        y = (h - crop_h) // 2
        writer.write(cv2.resize(img[y:y + crop_h, x:x + crop_w], size))
    writer.release()
    print(f"Wrote {frames} frames to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
