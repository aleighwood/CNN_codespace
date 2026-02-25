#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import image_crop as ic
import eval_roi_tile_skip as ers


def overlay_mask(rgb, mask, color=(0, 255, 0), alpha=0.35):
    out = rgb.copy().astype(np.float32)
    overlay = np.zeros_like(out)
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * overlay[m]
    return out.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Visualize ROI bbox and bitmap mask.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--auto-threshold", action="store_true")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--edge-threshold", type=float, default=0.2)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--close-iter", type=int, default=1)
    parser.add_argument("--open-iter", type=int, default=0)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--keep-largest", action="store_true", default=True)
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--roi-align", type=int, default=16)
    parser.add_argument("--roi-margin-px", type=int, default=8)
    parser.add_argument("--roi-margin-frac", type=float, default=0.0)
    parser.add_argument("--out-bbox", type=str, default="rtl/mem/roi_bbox_vis.png")
    parser.add_argument("--out-mask", type=str, default="rtl/mem/roi_bitmap_vis.png")
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    img = img.resize((args.image_size, args.image_size), resample=Image.BILINEAR)
    rgb = np.asarray(img).astype(np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    if args.edge:
        mag = ers.sobel_magnitude(gray)
        thr = args.edge_threshold
        if args.auto_threshold:
            thr = ic.otsu_threshold(mag)
        mask = (mag >= thr).astype(np.uint8)
    else:
        thr = args.threshold
        if args.auto_threshold:
            thr = ic.otsu_threshold(gray)
        mask = (gray >= thr).astype(np.uint8)
    if args.invert:
        mask = 1 - mask
    if args.close_iter > 0:
        mask = ic.binary_close(mask, k=args.kernel, iters=args.close_iter)
    if args.open_iter > 0:
        mask = ic.binary_open(mask, k=args.kernel, iters=args.open_iter)
    if args.center_weight > 0.0:
        mask = ers.select_component_center(mask, center_weight=args.center_weight, min_area=args.min_area)
    elif args.keep_largest or args.min_area > 0:
        mask = ic.keep_largest_component(mask, min_area=args.min_area)

    roi = ers.bbox_from_mask(mask)
    roi = ers.expand_roi(roi, args.roi_margin_px, args.roi_margin_frac, args.image_size, args.image_size)
    roi = ers.align_roi(roi, args.roi_align, args.image_size, args.image_size)

    # BBox visualization
    bbox_img = img.copy()
    if roi is not None:
        x, y, rw, rh = roi
        d = ImageDraw.Draw(bbox_img)
        d.rectangle([x, y, x + rw - 1, y + rh - 1], outline=(255, 0, 0))
    out_bbox = Path(args.out_bbox)
    out_bbox.parent.mkdir(parents=True, exist_ok=True)
    bbox_img.save(out_bbox)

    # Bitmap visualization
    mask_vis = overlay_mask((rgb * 255).astype(np.uint8), mask, color=(0, 255, 0), alpha=0.35)
    mask_img = Image.fromarray(mask_vis, mode="RGB")
    out_mask = Path(args.out_mask)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(out_mask)

    print(f"Saved bbox overlay: {out_bbox}")
    print(f"Saved bitmap overlay: {out_mask}")


if __name__ == "__main__":
    raise SystemExit(main())
