#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import image_crop as ic
import eval_roi_tile_skip as ers


def downsample_mask(mask, stride):
    if stride == 1:
        return mask
    h, w = mask.shape
    out_h = (h + stride - 1) // stride
    out_w = (w + stride - 1) // stride
    pad_h = out_h * stride - h
    pad_w = out_w * stride - w
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    reshaped = mask.reshape(out_h, stride, out_w, stride)
    return reshaped.max(axis=(1, 3)).astype(np.uint8)


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


def estimate_bitmap_savings(mask, args):
    # MobileNet v1 spec
    specs = [
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (1024, 2),
        (1024, 1),
    ]

    h = w = args.image_size
    total_full = 0.0
    total_roi = 0.0
    cur_mask = mask.copy().astype(np.uint8)

    # Conv1
    _, _, out_h, out_w = ers.compute_same_padding(h, w, 3, 2)
    tmask = tile_mask(cur_mask, args.tile_size)
    if args.roi_halo_tiles and args.roi_halo_layers >= 0:
        tmask = dilate_tiles(tmask, args.roi_halo_tiles if args.roi_halo_layers >= 0 else 0)
    ratio = tmask.sum() / max(1, tmask.size)
    macs = out_h * out_w * 32 * 3 * 3 * 3
    total_full += macs
    total_roi += macs * ratio
    cur_mask = downsample_mask(cur_mask, 2)
    in_h, in_w, in_c = out_h, out_w, 32

    for layer_id, (out_c, stride) in enumerate(specs, start=1):
        _, _, out_h, out_w = ers.compute_same_padding(in_h, in_w, 3, stride)
        if stride == 2:
            cur_mask = downsample_mask(cur_mask, 2)
        tmask = tile_mask(cur_mask, args.tile_size)
        if args.roi_halo_tiles and args.roi_halo_layers and layer_id <= args.roi_halo_layers:
            tmask = dilate_tiles(tmask, args.roi_halo_tiles)
        ratio = tmask.sum() / max(1, tmask.size)
        macs_dw = out_h * out_w * in_c * 3 * 3
        macs_pw = out_h * out_w * in_c * out_c
        total_full += macs_dw + macs_pw
        total_roi += (macs_dw + macs_pw) * ratio
        in_h, in_w, in_c = out_h, out_w, out_c

    return total_full, total_roi


def main():
    parser = argparse.ArgumentParser(description="Estimate compute savings from ROI tile skipping on a single image.")
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
    parser.add_argument("--roi-halo-tiles", type=int, default=1)
    parser.add_argument("--roi-halo-layers", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--bitmap", action="store_true", help="Estimate savings using ROI bitmap instead of bbox.")
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
    if roi is None:
        roi = (0, 0, args.image_size, args.image_size)

    roi_xyxy = (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
    full, roi_cost = ers.estimate_compute_savings(args.image_size, args.image_size, roi_xyxy, args)
    savings = 100.0 * (1.0 - (roi_cost / max(1.0, full)))

    print(f"ROI: {roi} (aligned to {args.roi_align}px, margin {args.roi_margin_px}px)")
    print(f"Estimated compute (bbox): full={full/1e6:.2f}M MACs, roi={roi_cost/1e6:.2f}M MACs, savings={savings:.1f}%")

    if args.bitmap:
        full_b, roi_b = estimate_bitmap_savings(mask, args)
        savings_b = 100.0 * (1.0 - (roi_b / max(1.0, full_b)))
        print(
            f"Estimated compute (bitmap): full={full_b/1e6:.2f}M MACs, roi={roi_b/1e6:.2f}M MACs, savings={savings_b:.1f}%"
        )


if __name__ == "__main__":
    raise SystemExit(main())
