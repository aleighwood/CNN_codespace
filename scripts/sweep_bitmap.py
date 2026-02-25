#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import image_crop as ic
import roi_compute_savings as rcs


def parse_shape(value):
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("input-shape must be H,W,C")
    return tuple(parts)


class SimpleArgs:
    def __init__(self, image_size, tile_size, roi_halo_tiles, roi_halo_layers):
        self.image_size = image_size
        self.tile_size = tile_size
        self.roi_halo_tiles = roi_halo_tiles
        self.roi_halo_layers = roi_halo_layers


def build_mask(img_path, args):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((args.image_size, args.image_size), resample=Image.BILINEAR)
    rgb = np.asarray(img).astype(np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    if args.edge:
        mag = rcs.ers.sobel_magnitude(gray)
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
        mask = rcs.ers.select_component_center(mask, center_weight=args.center_weight, min_area=args.min_area)
    elif args.keep_largest or args.min_area > 0:
        mask = ic.keep_largest_component(mask, min_area=args.min_area)
    return mask


def read_logits(mem_path):
    vals = []
    with open(mem_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = int(line, 16)
            if v & 0x80000000:
                v = v - (1 << 32)
            vals.append(v)
    return np.array(vals, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Sweep bitmap tile-skip options and report savings + top5.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--input-shape", type=parse_shape, default="224,224,3")
    parser.add_argument("--tflite", type=str, required=True)
    parser.add_argument("--mem-dir", type=str, default="rtl/mem")
    parser.add_argument("--mask-npy", type=str, default="rtl/mem/roi_mask.npy")
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
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--halo-tiles", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--halo-layers", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    h, w, c = args.input_shape
    args.image_size = h
    mem_dir = Path(args.mem_dir)
    mem_dir.mkdir(parents=True, exist_ok=True)

    # Build and save mask once
    mask = build_mask(args.image, args)
    mask_path = Path(args.mask_npy)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask_path, mask.astype(np.uint8))

    # Prepare input mem once (full image)
    input_mem = mem_dir / "input_sweep.mem"
    img_cmd = [
        sys.executable,
        "scripts/image_to_input_mem.py",
        "--image",
        args.image,
        "--input-shape",
        f"{h},{w},{c}",
        "--output-mem",
        str(input_mem),
        "--tflite",
        args.tflite,
    ]
    subprocess.run(img_cmd, check=True)

    class_index = json.load(open("scripts/imagenet_class_index.json", "r"))

    results = []
    for ht in args.halo_tiles:
        for hl in args.halo_layers:
            simple = SimpleArgs(args.input_shape[0], args.tile_size, ht, hl)
            full, roi_cost = rcs.estimate_bitmap_savings(mask, simple)
            savings = 100.0 * (1.0 - (roi_cost / max(1.0, full)))

            golden_cmd = [
                sys.executable,
                "scripts/gen_golden_fc.py",
                "--mem-dir",
                str(mem_dir),
                "--input-shape",
                f"{h},{w},{c}",
                "--input-mem-in",
                str(input_mem),
                "--input-mem",
                str(input_mem),
                "--expected-mem",
                str(mem_dir / "fc_expected.mem"),
                "--expected-logits-mem",
                str(mem_dir / "fc_logits_expected.mem"),
                "--q31",
                "--tflite",
                args.tflite,
                "--roi-mask-npy",
                str(mask_path),
                "--tile-size",
                str(args.tile_size),
                "--roi-halo-tiles",
                str(ht),
                "--roi-halo-layers",
                str(hl),
            ]
            subprocess.run(golden_cmd, check=True)

            logits = read_logits(mem_dir / "fc_logits_expected.mem")
            topk = min(args.topk, logits.size)
            top_idx = logits.argsort()[-topk:][::-1]
            top = []
            for i in top_idx:
                wnid, name = class_index[str(int(i))]
                top.append((int(i), wnid, name, int(logits[int(i)])))

            results.append((savings, ht, hl, top))

    results.sort(reverse=True, key=lambda x: x[0])
    for savings, ht, hl, top in results:
        print(f"halo_tiles={ht} halo_layers={hl} savings={savings:.1f}%")
        for cls_id, wnid, name, logit in top:
            print(f"  {cls_id} {wnid} {name} {logit}")


if __name__ == "__main__":
    raise SystemExit(main())
