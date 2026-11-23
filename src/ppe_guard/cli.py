"""Command-line entry point: `ppe-guard run ...` and `ppe-guard benchmark ...`."""

from __future__ import annotations

import argparse
import logging

from ppe_guard.config import load_config


def _cmd_run(args: argparse.Namespace) -> None:
    from ppe_guard.pipeline import run  # lazy import: torch loads only when needed

    cfg = load_config(args.config)
    # CLI flags override the YAML file (handy for quick experiments).
    if args.source is not None:
        cfg.source.uri = args.source
    if args.weights is not None:
        cfg.model.weights = args.weights
    if args.device is not None:
        cfg.model.device = args.device
    if args.save is not None:
        cfg.output.save_path = args.save
    if args.max_frames is not None:
        cfg.source.max_frames = args.max_frames
    if args.no_display:
        cfg.output.display = False
    run(cfg)


def _cmd_benchmark(args: argparse.Namespace) -> None:
    from ppe_guard.benchmark import run_benchmarks

    results = run_benchmarks(
        weights=args.weights,
        devices=args.devices,
        source=args.source,
        runs=args.runs,
        image_size=args.imgsz,
        tag=args.tag,
    )
    for r in results:
        print(f"{r.weights:>12} {r.device:>4}  mean {r.mean_ms:6.2f} ms  "
              f"p95 {r.p95_ms:6.2f} ms  {r.fps:6.1f} FPS")
    print(f"Saved benchmarks/{args.tag}.json and .md")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ppe-guard", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run detection on a video, webcam or stream")
    r.add_argument("-c", "--config", default="configs/default.yaml")
    r.add_argument("-s", "--source", help="File path, URL, or webcam index (overrides config)")
    r.add_argument("-w", "--weights")
    r.add_argument("-d", "--device", choices=["auto", "cpu", "mps"])
    r.add_argument("--save", help="Write annotated video to this .mp4 path")
    r.add_argument("--max-frames", type=int)
    r.add_argument("--no-display", action="store_true", help="Headless (no window)")
    r.set_defaults(func=_cmd_run)

    b = sub.add_parser("benchmark", help="Measure latency across models/devices")
    b.add_argument(
        "-w", "--weights", nargs="+", default=["weights/yolo11n.pt", "weights/yolo26n.pt"]
    )
    b.add_argument("-d", "--devices", nargs="+", default=["cpu", "mps"])
    b.add_argument("-s", "--source", help="Video for frames (default: bundled image)")
    b.add_argument("--runs", type=int, default=200)
    b.add_argument("--imgsz", type=int, default=640)
    b.add_argument("--tag", default="baseline", help="Output file name in benchmarks/")
    b.set_defaults(func=_cmd_benchmark)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
