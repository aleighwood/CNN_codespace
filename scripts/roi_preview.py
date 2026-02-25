#!/usr/bin/env python3
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import image_crop as ic


def list_images(val_dir):
    patterns = ("*.JPEG", "*.jpg", "*.jpeg", "*.png")
    files = []
    for pat in patterns:
        files.extend(glob.glob(str(Path(val_dir) / pat)))
    return sorted(files)


def align_roi(roi, align, w, h):
    if roi is None or align <= 1:
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
        return None
    return x0, y0, x1 - x0, y1 - y0


def expand_roi(roi, margin_px, margin_frac, w, h):
    if roi is None:
        return None
    x0, y0, rw, rh = roi
    x1 = x0 + rw
    y1 = y0 + rh
    if margin_frac and margin_frac > 0.0:
        mx = int(round(rw * margin_frac))
        my = int(round(rh * margin_frac))
    else:
        mx = int(margin_px)
        my = int(margin_px)
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w, x1 + mx)
    y1 = min(h, y1 + my)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def bbox_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


def sobel_magnitude(gray):
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = (
        -1 * gray[:, :-2] + 1 * gray[:, 2:]
        -2 * gray[:, :-2] + 2 * gray[:, 2:]
    )
    gy[1:-1, :] = (
        -1 * gray[:-2, :] + 1 * gray[2:, :]
        -2 * gray[:-2, :] + 2 * gray[2:, :]
    )
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag


def select_component_center(mask, center_weight=1.0, min_area=0):
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 0
    comps = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 1 and labels[y, x] == 0:
                label += 1
                stack = [(y, x)]
                labels[y, x] = label
                area = 0
                xs = []
                ys = []
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    xs.append(cx)
                    ys.append(cy)
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] == 1 and labels[ny, nx] == 0:
                                labels[ny, nx] = label
                                stack.append((ny, nx))
                if area >= min_area:
                    cx = float(sum(xs)) / len(xs)
                    cy = float(sum(ys)) / len(ys)
                    comps.append((label, area, cx, cy))

    if not comps:
        return mask

    cx0 = (w - 1) / 2.0
    cy0 = (h - 1) / 2.0
    best = None
    best_score = None
    for lab, area, cx, cy in comps:
        dx = (cx - cx0) / w
        dy = (cy - cy0) / h
        center_score = 1.0 - min(1.0, np.sqrt(dx * dx + dy * dy))
        score = area * (1.0 + center_weight * center_score)
        if best_score is None or score > best_score:
            best_score = score
            best = lab

    out = np.zeros_like(mask)
    out[labels == best] = 1
    return out


def main():
    parser = argparse.ArgumentParser(description="Preview ROI masks/bboxes on ILSVRC2012 val images.")
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--auto-select", action="store_true", help="Select images with clear foreground using heuristics.")
    parser.add_argument("--scan", type=int, default=500, help="Number of images to scan when auto-selecting.")
    parser.add_argument("--min-area-frac", type=float, default=0.08)
    parser.add_argument("--max-area-frac", type=float, default=0.6)
    parser.add_argument("--min-mask-frac", type=float, default=0.02)
    parser.add_argument("--max-mask-frac", type=float, default=0.5)
    parser.add_argument("--min-center-score", type=float, default=0.4)
    parser.add_argument("--output-dir", type=str, default="rtl/mem/roi_preview")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--auto-threshold", action="store_true")
    parser.add_argument("--edge", action="store_true", help="Use Sobel magnitude for mask.")
    parser.add_argument("--edge-threshold", type=float, default=0.2)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--close-iter", type=int, default=1)
    parser.add_argument("--open-iter", type=int, default=0)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--keep-largest", action="store_true", default=True)
    parser.add_argument("--center-weight", type=float, default=0.0, help="Weight center bias for component selection.")
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--roi-align", type=int, default=16)
    parser.add_argument("--roi-margin-px", type=int, default=0, help="Expand ROI by this many pixels.")
    parser.add_argument("--roi-margin-frac", type=float, default=0.0, help="Expand ROI by this fraction per side.")
    args = parser.parse_args()

    files = list_images(args.val_dir)
    if not files:
        raise SystemExit(f"No images found in {args.val_dir}")

    idx = np.arange(len(files))
    rng = np.random.default_rng(args.seed)
    if args.shuffle or args.auto_select:
        rng.shuffle(idx)
    if args.auto_select:
        scan = min(args.scan, len(idx))
        candidates = []
        for k in idx[:scan]:
            path = files[int(k)]
            if args.edge:
                img = Image.open(path).convert("RGB")
                img = img.resize((args.image_size, args.image_size), resample=Image.BILINEAR)
                rgb = np.asarray(img).astype(np.float32) / 255.0
                gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
                mag = sobel_magnitude(gray)
                thr = args.edge_threshold
                if args.auto_threshold:
                    thr = ic.otsu_threshold(mag)
                mask = (mag >= thr).astype(np.uint8)
                if args.invert:
                    mask = 1 - mask
            else:
                mask, thr, rgb = ic.load_image_mask(
                    path, args.image_size, args.image_size, args.threshold, args.auto_threshold, args.invert
                )
            if args.close_iter > 0:
                mask = ic.binary_close(mask, k=args.kernel, iters=args.close_iter)
            if args.open_iter > 0:
                mask = ic.binary_open(mask, k=args.kernel, iters=args.open_iter)
            if args.center_weight > 0.0:
                mask = select_component_center(mask, center_weight=args.center_weight, min_area=args.min_area)
            elif args.keep_largest or args.min_area > 0:
                mask = ic.keep_largest_component(mask, min_area=args.min_area)

            roi = align_roi(bbox_from_mask(mask), args.roi_align, args.image_size, args.image_size)
            if roi is None:
                continue
            x0, y0, rw, rh = roi
            area_frac = (rw * rh) / float(args.image_size * args.image_size)
            mask_frac = float(mask.sum()) / float(args.image_size * args.image_size)
            cx = x0 + rw / 2.0
            cy = y0 + rh / 2.0
            center_score = 1.0 - min(
                1.0,
                np.sqrt(((cx - (args.image_size - 1) / 2.0) / args.image_size) ** 2 +
                        ((cy - (args.image_size - 1) / 2.0) / args.image_size) ** 2),
            )
            if not (args.min_area_frac <= area_frac <= args.max_area_frac):
                continue
            if not (args.min_mask_frac <= mask_frac <= args.max_mask_frac):
                continue
            if center_score < args.min_center_score:
                continue
            score = center_score * (1.0 - area_frac)
            candidates.append((score, int(k)))
        if not candidates:
            raise SystemExit("No candidates matched the selection filters.")
        candidates.sort(reverse=True, key=lambda x: x[0])
        idx = np.array([k for _, k in candidates[: args.num]], dtype=np.int64)
    else:
        idx = idx[: args.num]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, k in enumerate(idx):
        path = files[int(k)]
        if args.edge:
            img = Image.open(path).convert("RGB")
            img = img.resize((args.image_size, args.image_size), resample=Image.BILINEAR)
            rgb = np.asarray(img).astype(np.float32) / 255.0
            gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            mag = sobel_magnitude(gray)
            used_thr = args.edge_threshold
            if args.auto_threshold:
                used_thr = ic.otsu_threshold(mag)
            mask = (mag >= used_thr).astype(np.uint8)
            if args.invert:
                mask = 1 - mask
        else:
            mask, used_thr, rgb = ic.load_image_mask(
                path, args.image_size, args.image_size, args.threshold, args.auto_threshold, args.invert
            )
        if args.close_iter > 0:
            mask = ic.binary_close(mask, k=args.kernel, iters=args.close_iter)
        if args.open_iter > 0:
            mask = ic.binary_open(mask, k=args.kernel, iters=args.open_iter)
        if args.center_weight > 0.0:
            mask = select_component_center(mask, center_weight=args.center_weight, min_area=args.min_area)
        elif args.keep_largest or args.min_area > 0:
            mask = ic.keep_largest_component(mask, min_area=args.min_area)

        roi = bbox_from_mask(mask)
        roi = expand_roi(roi, args.roi_margin_px, args.roi_margin_frac, args.image_size, args.image_size)
        roi = align_roi(roi, args.roi_align, args.image_size, args.image_size)

        img = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(img)
        if roi is not None:
            x, y, rw, rh = roi
            draw.rectangle([x, y, x + rw - 1, y + rh - 1], outline=(255, 0, 0))

        out_path = out_dir / f"roi_{i:02d}.png"
        img.save(out_path)
        print(f"{Path(path).name}: threshold={used_thr:.3f} roi={roi} -> {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
