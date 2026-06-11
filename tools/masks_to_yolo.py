"""
Convert grayscale/RGB mask images in GT/ to YOLO detection labels (.txt).

Expects layout (per split):
  images/*.jpg
  GT/<stem>.jpg or .png   (same stem as image)

Foreground = mask intensity > threshold (default 10). One box per connected
component above min_area (filters 1-pixel noise).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def find_mask_path(gt_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        p = gt_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def mask_to_boxes(
    mask_bgr: np.ndarray,
    threshold: int = 10,
    min_area: int = 100,
) -> list[tuple[float, float, float, float]]:
    """Return list of normalized YOLO boxes (cx, cy, w, h) in [0,1]."""
    if mask_bgr.ndim == 2:
        gray = mask_bgr
    else:
        gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    boxes: list[tuple[float, float, float, float]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw_i = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_i = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx = (x + bw_i / 2) / w
        cy = (y + bh_i / 2) / h
        nw = bw_i / w
        nh = bh_i / h
        boxes.append((cx, cy, nw, nh))
    return boxes


def process_split(
    split_dir: Path,
    class_id: int,
    threshold: int,
    min_area: int,
    dry_run: bool,
) -> tuple[int, int]:
    images_dir = split_dir / "images"
    gt_dir = split_dir / "GT"
    labels_dir = split_dir / "labels"
    if not dry_run:
        labels_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        stem = img_path.stem
        mask_path = find_mask_path(gt_dir, stem)
        if mask_path is None:
            print(f"[WARN] No GT for {img_path.name}")
            skipped += 1
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask is None:
            print(f"[WARN] Cannot read mask {mask_path}")
            skipped += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot read image {img_path}")
            skipped += 1
            continue
        if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        boxes = mask_to_boxes(mask, threshold=threshold, min_area=min_area)
        label_path = labels_dir / f"{stem}.txt"
        lines = [f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes]
        if not dry_run:
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
    return written, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="Masks in GT/ -> YOLO labels in labels/")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "dataset-splitM",
        help="Root folder containing Training/ and Testing/",
    )
    ap.add_argument("--class-id", type=int, default=0, help="Single-class YOLO id (default 0)")
    ap.add_argument("--threshold", type=int, default=10, help="Mask pixel > threshold = foreground")
    ap.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Min connected-component area in pixels (noise filter)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only report counts, do not write")
    args = ap.parse_args()

    root = args.dataset.resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset not found: {root}")

    total_w = total_s = 0
    for name in ("Training", "Testing"):
        split = root / name
        if not split.is_dir():
            print(f"[WARN] Missing split: {split}")
            continue
        w, s = process_split(split, args.class_id, args.threshold, args.min_area, args.dry_run)
        print(f"{name}: wrote {w} label files, skipped {s}")
        total_w += w
        total_s += s
    print(f"Total: {total_w} label files, skipped {total_s}")
    if args.dry_run:
        print("Dry run — no files written. Run without --dry-run to create labels/.")


if __name__ == "__main__":
    main()
