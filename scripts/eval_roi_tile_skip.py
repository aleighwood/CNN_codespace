#!/usr/bin/env python3
import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import image_crop as ic


def parse_shape(value):
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("input-shape must be H,W,C")
    return tuple(parts)


def load_labels(gt_path):
    labels = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(int(line))
    return np.array(labels, dtype=np.int64)


def list_val_images(val_dir):
    patterns = ("*.JPEG", "*.jpg", "*.jpeg", "*.png")
    files = []
    for pat in patterns:
        files.extend(glob.glob(str(Path(val_dir) / pat)))
    files = sorted(files)
    return files


def preprocess_image(path, out_h, out_w, center_crop=False, resize_shorter=256):
    img = Image.open(path).convert("RGB")
    if center_crop:
        w, h = img.size
        scale = float(resize_shorter) / float(min(w, h))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        left = max(0, (new_w - out_w) // 2)
        top = max(0, (new_h - out_h) // 2)
        img = img.crop((left, top, left + out_w, top + out_h))
    else:
        img = img.resize((out_w, out_h), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr, img


def bbox_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


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


def compute_same_padding(h, w, k, stride):
    out_h = (h + stride - 1) // stride
    out_w = (w + stride - 1) // stride
    pad_h = max((out_h - 1) * stride + k - h, 0)
    pad_w = max((out_w - 1) * stride + k - w, 0)
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    return pad_top, pad_left, out_h, out_w


def map_roi(roi, in_h, in_w, stride, halo):
    x0, y0, x1, y1 = roi
    x0 = max(0, x0 - halo)
    y0 = max(0, y0 - halo)
    x1 = min(in_w, x1 + halo)
    y1 = min(in_h, y1 + halo)
    out_w = (in_w + stride - 1) // stride
    out_h = (in_h + stride - 1) // stride
    out_x0 = max(0, min(out_w, x0 // stride))
    out_y0 = max(0, min(out_h, y0 // stride))
    out_x1 = max(out_x0, min(out_w, (x1 + stride - 1) // stride))
    out_y1 = max(out_y0, min(out_h, (y1 + stride - 1) // stride))
    return out_x0, out_y0, out_x1, out_y1


def ceil_div(a, b):
    return (a + b - 1) // b


def tile_ratio(roi, out_h, out_w, tile):
    if roi is None:
        return 1.0
    x0, y0, x1, y1 = roi
    rw = max(0, x1 - x0)
    rh = max(0, y1 - y0)
    if rw == 0 or rh == 0:
        return 0.0
    full_tiles = ceil_div(out_w, tile) * ceil_div(out_h, tile)
    roi_tiles = ceil_div(rw, tile) * ceil_div(rh, tile)
    return min(1.0, roi_tiles / max(1, full_tiles))


def layer_halo_px(layer_idx, args):
    halo = 1
    if args.roi_halo_tiles and args.roi_halo_layers:
        if layer_idx <= args.roi_halo_layers:
            halo = max(halo, args.roi_halo_tiles * args.tile_size)
    return halo


def estimate_compute_savings(h, w, roi, args):
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

    total_full = 0.0
    total_roi = 0.0

    # Conv1
    _, _, out_h, out_w = compute_same_padding(h, w, 3, 2)
    full_tiles = tile_ratio(None, out_h, out_w, args.tile_size)
    halo = layer_halo_px(0, args)
    if roi is not None:
        roi = map_roi(roi, h, w, stride=2, halo=halo)
    ratio = tile_ratio(roi, out_h, out_w, args.tile_size)
    macs = out_h * out_w * 32 * 3 * 3 * 3
    total_full += macs * full_tiles
    total_roi += macs * ratio

    in_h, in_w, in_c = out_h, out_w, 32

    for layer_id, (out_c, stride) in enumerate(specs, start=1):
        # depthwise
        _, _, out_h, out_w = compute_same_padding(in_h, in_w, 3, stride)
        halo = layer_halo_px(layer_id, args)
        if roi is not None:
            roi = map_roi(roi, in_h, in_w, stride=stride, halo=halo)
        ratio = tile_ratio(roi, out_h, out_w, args.tile_size)
        macs_dw = out_h * out_w * in_c * 3 * 3
        total_full += macs_dw
        total_roi += macs_dw * ratio

        # pointwise
        macs_pw = out_h * out_w * in_c * out_c
        total_full += macs_pw
        total_roi += macs_pw * ratio

        in_h, in_w, in_c = out_h, out_w, out_c

    return total_full, total_roi


def write_mem8(path, values):
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            f.write(f"{int(v) & 0xFF:02X}\n")


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
    parser = argparse.ArgumentParser(description="Evaluate ROI tile-skip golden model on ILSVRC2012 val subset.")
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--gt-file", type=str, required=True)
    parser.add_argument("--tflite", type=str, required=True)
    parser.add_argument("--mem-dir", type=str, default="rtl/mem")
    parser.add_argument("--input-shape", type=parse_shape, default="224,224,3")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--auto-select", action="store_true", help="Select images with clear foreground using heuristics.")
    parser.add_argument("--scan", type=int, default=500, help="Number of images to scan when auto-selecting.")
    parser.add_argument("--min-area-frac", type=float, default=0.08)
    parser.add_argument("--max-area-frac", type=float, default=0.6)
    parser.add_argument("--min-mask-frac", type=float, default=0.02)
    parser.add_argument("--max-mask-frac", type=float, default=0.5)
    parser.add_argument("--min-center-score", type=float, default=0.4)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--resize-shorter", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--auto-threshold", action="store_true")
    parser.add_argument("--edge", action="store_true", help="Use Sobel magnitude for mask.")
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
    parser.add_argument("--report-compute", action="store_true", help="Report estimated compute savings.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-examples", type=int, default=0, help="Save bbox overlays for first N images.")
    parser.add_argument("--output-dir", type=str, default="rtl/mem/roi_eval")
    args = parser.parse_args()

    h, w, c = args.input_shape
    if c != 3:
        raise SystemExit("Only 3-channel inputs supported.")

    labels = load_labels(args.gt_file)
    files = list_val_images(args.val_dir)
    if len(files) != labels.shape[0]:
        print(f"Warning: {len(files)} images vs {labels.shape[0]} labels")

    indices = np.arange(min(len(files), labels.shape[0]))
    rng = np.random.default_rng(args.seed)
    if args.shuffle or args.auto_select:
        rng.shuffle(indices)
    if args.auto_select:
        scan = min(args.scan, len(indices))
        candidates = []
        for k in indices[:scan]:
            img_path = files[int(k)]
            arr, _ = preprocess_image(
                img_path,
                h,
                w,
                center_crop=args.center_crop,
                resize_shorter=args.resize_shorter,
            )
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            if args.edge:
                mag = sobel_magnitude(gray)
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
                mask = select_component_center(mask, center_weight=args.center_weight, min_area=args.min_area)
            elif args.keep_largest or args.min_area > 0:
                mask = ic.keep_largest_component(mask, min_area=args.min_area)

            roi = bbox_from_mask(mask)
            roi = expand_roi(roi, args.roi_margin_px, args.roi_margin_frac, w, h)
            roi = align_roi(roi, args.roi_align, w, h)
            if roi is None:
                continue
            x0, y0, rw, rh = roi
            area_frac = (rw * rh) / float(w * h)
            mask_frac = float(mask.sum()) / float(w * h)
            cx = x0 + rw / 2.0
            cy = y0 + rh / 2.0
            center_score = 1.0 - min(
                1.0,
                np.sqrt(((cx - (w - 1) / 2.0) / w) ** 2 + ((cy - (h - 1) / 2.0) / h) ** 2),
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
        indices = np.array([k for _, k in candidates[: args.num_samples]], dtype=np.int64)
    elif args.num_samples > 0:
        indices = indices[: args.num_samples]

    interpreter = tf.lite.Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    q_scale, q_zp = input_details.get("quantization", (None, None))
    if q_scale is None:
        raise SystemExit("TFLite input quantization scale not found.")

    mem_dir = Path(args.mem_dir)
    mem_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir)
    if args.save_examples > 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = 0
    sum_full = 0.0
    sum_roi = 0.0

    for n, idx in enumerate(indices):
        img_path = files[int(idx)]
        gt = int(labels[int(idx)]) - 1

        arr, pil_img = preprocess_image(
            img_path,
            h,
            w,
            center_crop=args.center_crop,
            resize_shorter=args.resize_shorter,
        )

        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        if args.edge:
            mag = sobel_magnitude(gray)
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
            mask = select_component_center(mask, center_weight=args.center_weight, min_area=args.min_area)
        elif args.keep_largest or args.min_area > 0:
            mask = ic.keep_largest_component(mask, min_area=args.min_area)

        roi = bbox_from_mask(mask)
        roi = expand_roi(roi, args.roi_margin_px, args.roi_margin_frac, w, h)
        roi = align_roi(roi, args.roi_align, w, h)
        if roi is None:
            roi = (0, 0, w, h)

        if args.report_compute:
            roi_xyxy = (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
            full, roi_cost = estimate_compute_savings(h, w, roi_xyxy, args)
            sum_full += full
            sum_roi += roi_cost

        if args.save_examples > 0 and n < args.save_examples:
            vis = pil_img.copy()
            d = ImageDraw.Draw(vis)
            x0, y0, rw, rh = roi
            d.rectangle([x0, y0, x0 + rw - 1, y0 + rh - 1], outline=(255, 0, 0))
            vis.save(out_dir / f"roi_{n:03d}.png")

        img_norm = arr * 2.0 - 1.0
        q = np.round(img_norm / q_scale + q_zp)
        q = np.clip(q, -128, 127).astype(np.int8)
        chw = np.transpose(q, (2, 0, 1)).reshape(-1)
        input_mem = mem_dir / "input_eval.mem"
        write_mem8(input_mem, chw)

        cmd = [
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
            "--roi",
            f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}",
            "--roi-skip",
        ]
        if args.roi_align and args.roi_align > 1:
            cmd += ["--roi-align", str(args.roi_align)]
        if args.roi_halo_tiles and args.roi_halo_layers:
            cmd += ["--roi-halo-tiles", str(args.roi_halo_tiles)]
            cmd += ["--roi-halo-layers", str(args.roi_halo_layers)]
            cmd += ["--tile-size", str(args.tile_size)]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        logits = read_logits(mem_dir / "fc_logits_expected.mem")
        pred = int(np.argmax(logits))
        ok = pred == gt
        correct += int(ok)
        total += 1

        if args.verbose:
            print(f"{Path(img_path).name}: gt={gt} pred={pred} {'OK' if ok else 'MISS'}")

    acc = 100.0 * correct / max(1, total)
    print(f"Tile-skip accuracy: {correct}/{total} = {acc:.2f}%")
    if args.report_compute and total > 0:
        avg_full = sum_full / total
        avg_roi = sum_roi / total
        savings = 100.0 * (1.0 - (avg_roi / max(1.0, avg_full)))
        print(f"Estimated compute: full={avg_full/1e6:.2f}M MACs, roi={avg_roi/1e6:.2f}M MACs, savings={savings:.1f}%")


if __name__ == "__main__":
    raise SystemExit(main())
