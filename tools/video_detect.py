"""
Run YOLO on a video file and save an annotated output video (no Flask).

Usage (from project root, venv active):
  python tools/video_detect.py --source path/to/input.mp4 --conf 0.7

Or with Ultralytics CLI only:
  yolo predict model=models/best.pt source=path/to/input.mp4 conf=0.7 save=True
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="YOLO video inference → saved annotated video")
    parser.add_argument("--source", "-s", required=True, type=Path, help="Input video file (.mp4, .avi, ...)")
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        default=root / "models" / "best.pt",
        help="Weights file (default: models/best.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold (default 0.7)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size")
    parser.add_argument(
        "--project",
        type=Path,
        default=root / "runs" / "detect",
        help="Where to create predict/ output folder",
    )
    parser.add_argument("--name", type=str, default="video_out", help="Subfolder name under project")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Video not found: {args.source}")
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    model.predict(
        source=str(args.source),
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )
    out_dir = args.project / args.name
    print(f"Done. Look for saved video under: {out_dir}")


if __name__ == "__main__":
    main()
