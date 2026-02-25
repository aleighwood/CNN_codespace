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


def tile_mask(mask, tile):
    h, w = mask.shape
    out_h = (h + tile - 1) // tile
    out_w = (w + tile - 1) // tile
    pad_h = out_h * tile - h
    pad_w = out_w * tile - w
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    reshaped = mask.reshape(out_h, tile, out_w, tile)
    return reshaped.max(axis=(1, 3)).astype(np.uint8)


def dilate_tiles(tmask, halo_tiles):
    if halo_tiles <= 0:
        return tmask
    k = 2 * halo_tiles + 1
    return ic.binary_dilate(tmask, k=k)


def main():
    parser = argparse.ArgumentParser(description="Draw bitmap tile-skip overlay.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--halo-tiles", type=int, default=1)
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
    parser.add_argument("--out", type=str, default="rtl/mem/roi_tile_overlay.png")
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

    tmask = tile_mask(mask, args.tile_size)
    if args.halo_tiles > 0:
        tmask = dilate_tiles(tmask, args.halo_tiles)

    out = img.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    tiles_y, tiles_x = tmask.shape
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x0 = tx * args.tile_size
            y0 = ty * args.tile_size
            x1 = min(args.image_size, x0 + args.tile_size) - 1
            y1 = min(args.image_size, y0 + args.tile_size) - 1
            if tmask[ty, tx] == 1:
                draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 200), fill=(255, 0, 0, 60))
            else:
                draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180, 80))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"Saved tile overlay: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
