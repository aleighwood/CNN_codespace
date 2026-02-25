#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_shape(value):
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("input-shape must be H,W,C")
    return tuple(parts)


def parse_roi(value):
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("roi must be x,y,w,h")
    return tuple(parts)


def bbox_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise SystemExit("Mask is empty; no ROI found.")
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


def align_roi_pixels(roi, align, w, h):
    if align is None or align <= 1:
        return roi
    x0, y0, rw, rh = roi
    x1 = x0 + rw
    y1 = y0 + rh
    x0 = (x0 // align) * align
    y0 = (y0 // align) * align
    x1 = ((x1 + align - 1) // align) * align
    y1 = ((y1 + align - 1) // align) * align
    x0 = max(0, min(x0, w))
    y0 = max(0, min(y0, h))
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        raise SystemExit(f"Aligned ROI empty: {(x0,y0,x1,y1)}")
    return x0, y0, x1 - x0, y1 - y0


def main():
    parser = argparse.ArgumentParser(description="Generate ROI input mem + golden outputs.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--input-shape", type=parse_shape, default="160,160,3")
    parser.add_argument("--tflite", type=str, default="quantized_models/mobilenet_int8_ilsvrc2012_5000.tflite")
    parser.add_argument("--mem-dir", type=str, default="rtl/mem")
    parser.add_argument("--output-mem", type=str, default="rtl/mem/input_roi.mem")
    parser.add_argument("--roi", type=parse_roi, default=None, help="ROI x,y,w,h (pixels).")
    parser.add_argument("--roi-normalized", action="store_true")
    parser.add_argument("--roi-margin", type=float, default=0.0)
    parser.add_argument("--roi-from-mask", action="store_true", help="Auto ROI from mask (thresholded image).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Grayscale threshold (0..1).")
    parser.add_argument("--auto-threshold", action="store_true", help="Use Otsu threshold.")
    parser.add_argument("--invert", action="store_true", help="Invert mask after threshold.")
    parser.add_argument("--close-iter", type=int, default=0, help="Apply binary close this many times.")
    parser.add_argument("--open-iter", type=int, default=0, help="Apply binary open this many times.")
    parser.add_argument("--kernel", type=int, default=3, help="Kernel size for morphology (odd).")
    parser.add_argument("--keep-largest", action="store_true", help="Keep only the largest component.")
    parser.add_argument("--min-area", type=int, default=0, help="Keep components with area >= this.")
    parser.add_argument("--roi-mode", type=str, default="tile-skip", choices=("crop", "tile-skip"))
    parser.add_argument("--roi-align", type=int, default=0, help="Align ROI to this pixel grid (e.g., 16).")
    parser.add_argument("--roi-halo-tiles", type=int, default=0, help="Halo in tiles for early layers.")
    parser.add_argument("--roi-halo-layers", type=int, default=0, help="Apply tile halo for first N layers.")
    parser.add_argument("--tile-size", type=int, default=16, help="Tile size in pixels for ROI halo.")
    parser.add_argument("--output-bbox-image", type=str, default="", help="Save RGB image with ROI bbox.")
    parser.add_argument("--output-crop-image", type=str, default="", help="Save preprocessed RGB input image.")
    parser.add_argument("--print-topk", action="store_true", help="Print top-k predictions from logits.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--q31", action="store_true", default=True)
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Enable automatic ROI-from-mask + bbox output + top-k (sensible defaults).",
    )
    args = parser.parse_args()

    mem_dir = Path(args.mem_dir)
    mem_dir.mkdir(parents=True, exist_ok=True)

    h, w, c = args.input_shape
    input_shape = f"{h},{w},{c}"

    argv = set(sys.argv[1:])
    if args.auto:
        args.roi_from_mask = True
        if "--auto-threshold" not in argv and "--threshold" not in argv:
            args.auto_threshold = True
        if "--keep-largest" not in argv and "--min-area" not in argv:
            args.keep_largest = True
        if "--close-iter" not in argv:
            args.close_iter = max(args.close_iter, 1)
        if "--kernel" not in argv:
            args.kernel = 5
        if "--output-bbox-image" not in argv:
            args.output_bbox_image = str(mem_dir / "roi_bbox.png")
        if "--output-crop-image" not in argv:
            if args.roi_mode == "tile-skip":
                args.output_crop_image = str(mem_dir / "input_image.png")
            else:
                args.output_crop_image = str(mem_dir / "roi_crop.png")
        if "--print-topk" not in argv:
            args.print_topk = True

    roi = args.roi
    roi_norm = None
    if args.roi_from_mask:
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import image_crop as ic  # local script

        mask, used_thresh, rgb = ic.load_image_mask(
            args.image, w, h, args.threshold, args.auto_threshold, args.invert
        )
        if args.close_iter > 0:
            mask = ic.binary_close(mask, k=args.kernel, iters=args.close_iter)
        if args.open_iter > 0:
            mask = ic.binary_open(mask, k=args.kernel, iters=args.open_iter)
        if args.keep_largest or args.min_area > 0:
            mask = ic.keep_largest_component(mask, min_area=args.min_area)
        roi = bbox_from_mask(mask)
        if args.roi_align and args.roi_align > 1:
            roi = align_roi_pixels(roi, args.roi_align, w, h)
            print(f"Aligned ROI to {args.roi_align}px grid: {roi}")
        roi_norm = (roi[0] / w, roi[1] / h, roi[2] / w, roi[3] / h)
        print(f"Auto ROI from mask (threshold={used_thresh:.3f}): {roi}")

        if args.output_bbox_image:
            img = Image.open(args.image).convert("RGB").resize((w, h), resample=Image.BILINEAR)
            out = img.copy()
            d = ImageDraw.Draw(out)
            x, y, rw, rh = roi
            for offset in range(2):
                d.rectangle([x - offset, y - offset, x + rw - 1 + offset, y + rh - 1 + offset], outline=(255, 0, 0))
            out_path = Path(args.output_bbox_image)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.save(out_path)
            print(f"Saved ROI bbox image to {out_path}")
    elif roi is not None and args.roi_align and args.roi_align > 1:
        if args.roi_normalized:
            roi_px = (roi[0] * w, roi[1] * h, roi[2] * w, roi[3] * h)
        else:
            roi_px = roi
        roi_px = align_roi_pixels(tuple(int(round(v)) for v in roi_px), args.roi_align, w, h)
        roi = roi_px
        roi_norm = (roi[0] / w, roi[1] / h, roi[2] / w, roi[3] / h)
        print(f"Aligned ROI to {args.roi_align}px grid: {roi}")

    if args.roi_mode == "crop":
        img_cmd = [
            sys.executable,
            "scripts/image_to_input_mem_roi.py",
            "--image",
            args.image,
            "--input-shape",
            input_shape,
            "--output-mem",
            args.output_mem,
            "--tflite",
            args.tflite,
        ]
        if roi is not None:
            if roi_norm is not None:
                img_cmd += ["--roi", ",".join(f"{v:.6f}" for v in roi_norm), "--roi-normalized"]
                print(f"Using normalized ROI for input crop: {roi_norm}")
            else:
                img_cmd += ["--roi", ",".join(str(v) for v in roi)]
                if args.roi_normalized:
                    img_cmd += ["--roi-normalized"]
        if args.output_crop_image:
            img_cmd += ["--output-image", args.output_crop_image]
    else:
        img_cmd = [
            sys.executable,
            "scripts/image_to_input_mem.py",
            "--image",
            args.image,
            "--input-shape",
            input_shape,
            "--output-mem",
            args.output_mem,
            "--tflite",
            args.tflite,
        ]
        if args.output_crop_image:
            img_cmd += ["--output-image", args.output_crop_image]
    if args.roi_margin:
        img_cmd += ["--roi-margin", str(args.roi_margin)]

    print("Running:", " ".join(img_cmd))
    subprocess.run(img_cmd, check=True)

    golden_cmd = [
        sys.executable,
        "scripts/gen_golden_fc.py",
        "--mem-dir",
        str(mem_dir),
        "--input-shape",
        input_shape,
        "--input-mem-in",
        args.output_mem,
        "--input-mem",
        args.output_mem,
        "--expected-mem",
        str(mem_dir / "fc_expected.mem"),
        "--expected-logits-mem",
        str(mem_dir / "fc_logits_expected.mem"),
    ]
    if args.q31 and args.tflite:
        golden_cmd += ["--q31", "--tflite", args.tflite]
    if roi is not None:
        golden_cmd += ["--roi", ",".join(str(v) for v in roi)]
    if args.roi_normalized:
        golden_cmd += ["--roi-normalized"]
    if args.roi_margin:
        golden_cmd += ["--roi-margin", str(args.roi_margin)]
    if args.roi_align and args.roi_align > 1:
        golden_cmd += ["--roi-align", str(args.roi_align)]
    if args.roi_halo_tiles and args.roi_halo_layers:
        golden_cmd += ["--roi-halo-tiles", str(args.roi_halo_tiles)]
        golden_cmd += ["--roi-halo-layers", str(args.roi_halo_layers)]
        golden_cmd += ["--tile-size", str(args.tile_size)]
    if args.roi_mode == "tile-skip":
        golden_cmd += ["--roi-skip"]

    print("Running:", " ".join(golden_cmd))
    subprocess.run(golden_cmd, check=True)

    print("Done. Golden outputs written to:")
    print(f"  {mem_dir / 'fc_expected.mem'}")
    print(f"  {mem_dir / 'fc_logits_expected.mem'}")

    if args.print_topk:
        logits_path = mem_dir / "fc_logits_expected.mem"
        if logits_path.exists():
            vals = []
            with open(logits_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    v = int(line, 16)
                    if v & 0x80000000:
                        v = v - (1 << 32)
                    vals.append(v)
            logits = np.array(vals, dtype=np.int64)
            topk = min(args.topk, logits.size)
            top_idx = logits.argsort()[-topk:][::-1]
            class_index = json.load(open("scripts/imagenet_class_index.json", "r"))
            print("Top predictions:")
            for i in top_idx:
                wnid, name = class_index[str(int(i))]
                print(f"  {int(i)} {wnid} {name} {int(logits[int(i)])}")


if __name__ == "__main__":
    raise SystemExit(main())
