#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def parse_shape(value):
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("input-shape must be H,W,C")
    return tuple(parts)


def ceil_div(a, b):
    return (a + b - 1) // b


def downsample_mask(mask, stride):
    if stride == 1:
        return mask
    h, w = mask.shape
    out_h = ceil_div(h, stride)
    out_w = ceil_div(w, stride)
    pad_h = out_h * stride - h
    pad_w = out_w * stride - w
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    reshaped = mask.reshape(out_h, stride, out_w, stride)
    return reshaped.max(axis=(1, 3)).astype(np.uint8)


def tile_mask(mask, tile):
    h, w = mask.shape
    out_h = ceil_div(h, tile)
    out_w = ceil_div(w, tile)
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
    pad = k // 2
    padded = np.pad(tmask, pad, mode="constant", constant_values=0)
    out = np.zeros_like(tmask)
    h, w = tmask.shape
    for y in range(h):
        for x in range(w):
            if np.any(padded[y : y + k, x : x + k]):
                out[y, x] = 1
    return out


def to_hex8(v):
    return f"{int(v) & 0xFF:02X}"


def write_mem8(path, values):
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            f.write(to_hex8(v))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Generate per-layer tile mask memory from ROI bitmap.")
    parser.add_argument("--mask-npy", type=str, required=True, help="Input ROI mask (.npy) at input size.")
    parser.add_argument("--input-shape", type=parse_shape, default="224,224,3")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--roi-halo-tiles", type=int, default=0)
    parser.add_argument("--roi-halo-layers", type=int, default=0)
    parser.add_argument("--out-mem", type=str, default="rtl/mem/tile_mask.mem")
    parser.add_argument("--out-info", type=str, default="rtl/mem/tile_mask_info.txt")
    args = parser.parse_args()

    in_h, in_w, _ = args.input_shape
    mask = np.load(args.mask_npy)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[0] != in_h or mask.shape[1] != in_w:
        raise SystemExit(f"Mask shape {mask.shape} != input {(in_h, in_w)}")
    mask = (mask > 0).astype(np.uint8)

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

    out_vals = []
    info_lines = []
    base = 0

    # Conv1 (stride 2)
    mask = downsample_mask(mask, 2)
    out_h = ceil_div(in_h, 2)
    out_w = ceil_div(in_w, 2)
    tmask = tile_mask(mask, args.tile_size)
    if args.roi_halo_tiles and args.roi_halo_layers >= 0:
        tmask = dilate_tiles(tmask, args.roi_halo_tiles)
    th, tw = tmask.shape
    info_lines.append(f"layer0 conv1 out={out_h}x{out_w} tiles={th}x{tw} base={base}")
    out_vals.extend(tmask.reshape(-1).tolist())
    base += th * tw

    cur_h = out_h
    cur_w = out_w
    for layer_id, (_, stride) in enumerate(specs, start=1):
        if stride == 2:
            mask = downsample_mask(mask, 2)
        cur_h = ceil_div(cur_h, stride)
        cur_w = ceil_div(cur_w, stride)
        tmask = tile_mask(mask, args.tile_size)
        if args.roi_halo_tiles and args.roi_halo_layers and layer_id <= args.roi_halo_layers:
            tmask = dilate_tiles(tmask, args.roi_halo_tiles)
        th, tw = tmask.shape
        info_lines.append(
            f"layer{layer_id} out={cur_h}x{cur_w} tiles={th}x{tw} base={base}"
        )
        out_vals.extend(tmask.reshape(-1).tolist())
        base += th * tw

    out_path = Path(args.out_mem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_mem8(out_path, out_vals)

    if args.out_info:
        info_path = Path(args.out_info)
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text("\n".join(info_lines) + "\n", encoding="utf-8")

    print(f"Wrote tile mask mem: {out_path} ({len(out_vals)} entries)")
    if args.out_info:
        print(f"Wrote info: {args.out_info}")


if __name__ == "__main__":
    raise SystemExit(main())
