#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


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


def to_hex8(v):
    return f"{int(v) & 0xFF:02X}"


def write_mem8(path, values):
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            f.write(to_hex8(v))
            f.write("\n")


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def main():
    parser = argparse.ArgumentParser(description="Convert an ROI from an image to int8 CHW .mem input.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file (jpg/png).")
    parser.add_argument("--input-shape", type=parse_shape, default="160,160,3")
    parser.add_argument("--output-mem", type=str, default="rtl/mem/input_roi.mem")
    parser.add_argument("--normalize", type=str, default="-1,1", help="Normalization range, default -1,1")
    parser.add_argument("--tflite", type=str, default="", help="Optional TFLite model to fetch input scale/zero-point.")
    parser.add_argument("--scale", type=float, default=None, help="Override quant scale (used with --zero-point).")
    parser.add_argument("--zero-point", type=int, default=None, help="Override quant zero-point.")
    parser.add_argument("--output-image", type=str, default="", help="Optional path to save cropped/resized RGB image.")
    parser.add_argument("--roi", type=parse_roi, default=None, help="ROI x,y,w,h (pixels).")
    parser.add_argument("--roi-normalized", action="store_true", help="Interpret --roi as normalized (0..1).")
    parser.add_argument(
        "--roi-margin",
        type=float,
        default=0.0,
        help="Expand ROI by this fraction on each side (e.g., 0.1 adds 10%%).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Unused, kept for parity")
    args = parser.parse_args()

    h, w, c = args.input_shape
    if c != 3:
        raise SystemExit("Only 3-channel RGB is supported right now.")

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    raw = tf.io.read_file(str(img_path))
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0..1
    img_h = int(img.shape[0])
    img_w = int(img.shape[1])

    if args.roi is None:
        x0, y0, rw, rh = 0.0, 0.0, float(img_w), float(img_h)
    else:
        x0, y0, rw, rh = args.roi
        if args.roi_normalized:
            x0 *= img_w
            y0 *= img_h
            rw *= img_w
            rh *= img_h

    if rw <= 0 or rh <= 0:
        raise SystemExit("ROI width/height must be > 0.")

    margin = max(0.0, float(args.roi_margin))
    x0 -= rw * margin
    y0 -= rh * margin
    rw *= (1.0 + 2.0 * margin)
    rh *= (1.0 + 2.0 * margin)

    x1 = clamp(int(round(x0 + rw)), 0, img_w)
    y1 = clamp(int(round(y0 + rh)), 0, img_h)
    x0 = clamp(int(round(x0)), 0, img_w)
    y0 = clamp(int(round(y0)), 0, img_h)

    if x1 <= x0 or y1 <= y0:
        raise SystemExit(f"Clamped ROI is empty: ({x0},{y0})-({x1},{y1})")

    roi = img[y0:y1, x0:x1, :]
    if (roi.shape[0] != h) or (roi.shape[1] != w):
        roi = tf.image.resize(roi, (h, w), method="bilinear", antialias=True)

    if args.output_image:
        vis = tf.clip_by_value(roi * 255.0, 0.0, 255.0)
        vis = tf.cast(tf.round(vis), tf.uint8).numpy()
        out_img = Image.fromarray(vis, mode="RGB")
        out_path = Path(args.output_image)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path)

    if args.normalize.strip() == "-1,1":
        roi = roi * 2.0 - 1.0
    else:
        raise SystemExit("Only normalize=-1,1 is supported right now.")

    q_scale = None
    q_zp = None
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=args.tflite)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        q_scale, q_zp = input_details.get("quantization", (None, None))
    if args.scale is not None and args.zero_point is not None:
        q_scale = args.scale
        q_zp = args.zero_point

    if q_scale is None:
        q = tf.round(roi * 127.0)
        q = tf.clip_by_value(q, -127.0, 127.0)
    else:
        q = tf.round(roi / q_scale + q_zp)
        q = tf.clip_by_value(q, -128.0, 127.0)
    q = tf.cast(q, tf.int8).numpy()  # HWC

    chw = np.transpose(q, (2, 0, 1)).reshape(-1)
    write_mem8(args.output_mem, chw)

    print(f"Image: {img_w}x{img_h}")
    print(f"ROI (clamped): x={x0} y={y0} w={x1-x0} h={y1-y0}")
    print(f"Output: {h}x{w}")
    print(f"Wrote {chw.size} values to {args.output_mem}")
    print(f"Min/Max int8: {int(q.min())}/{int(q.max())}")


if __name__ == "__main__":
    raise SystemExit(main())
