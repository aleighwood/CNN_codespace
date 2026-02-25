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


def to_hex8(v):
    return f"{int(v) & 0xFF:02X}"


def write_mem8(path, values):
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            f.write(to_hex8(v))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Convert an image to int8 CHW .mem input.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file (jpg/png).")
    parser.add_argument("--input-shape", type=parse_shape, default="16,16,3")
    parser.add_argument("--output-mem", type=str, default="rtl/mem/input_rand.mem")
    parser.add_argument("--normalize", type=str, default="-1,1", help="Normalization range, default -1,1")
    parser.add_argument("--tflite", type=str, default="", help="Optional TFLite model to fetch input scale/zero-point.")
    parser.add_argument("--scale", type=float, default=None, help="Override quant scale (used with --zero-point).")
    parser.add_argument("--zero-point", type=int, default=None, help="Override quant zero-point.")
    parser.add_argument("--output-image", type=str, default="", help="Optional path to save resized RGB image.")
    parser.add_argument(
        "--center-crop",
        action="store_true",
        help="Resize shorter side to --resize-shorter and center-crop to input size.",
    )
    parser.add_argument("--resize-shorter", type=int, default=256)
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
    if args.center_crop:
        shape = tf.shape(img)
        height = tf.cast(shape[0], tf.float32)
        width = tf.cast(shape[1], tf.float32)
        scale = tf.cast(args.resize_shorter, tf.float32) / tf.minimum(height, width)
        new_h = tf.cast(tf.math.round(height * scale), tf.int32)
        new_w = tf.cast(tf.math.round(width * scale), tf.int32)
        img = tf.image.resize(img, (new_h, new_w), method="bilinear", antialias=True)
        img = tf.image.resize_with_crop_or_pad(img, h, w)
    else:
        if (img.shape[0] != h) or (img.shape[1] != w):
            img = tf.image.resize(img, (h, w), method="bilinear", antialias=True)

    if args.output_image:
        vis = tf.clip_by_value(img * 255.0, 0.0, 255.0)
        vis = tf.cast(tf.round(vis), tf.uint8).numpy()
        out_img = Image.fromarray(vis, mode="RGB")
        out_path = Path(args.output_image)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path)

    if args.normalize.strip() == "-1,1":
        img = img * 2.0 - 1.0
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
        q = tf.round(img * 127.0)
        q = tf.clip_by_value(q, -127.0, 127.0)
    else:
        q = tf.round(img / q_scale + q_zp)
        q = tf.clip_by_value(q, -128.0, 127.0)
    q = tf.cast(q, tf.int8).numpy()  # HWC

    # Write CHW order
    chw = np.transpose(q, (2, 0, 1)).reshape(-1)
    write_mem8(args.output_mem, chw)

    print(f"Wrote {chw.size} values to {args.output_mem}")
    print(f"Min/Max int8: {int(q.min())}/{int(q.max())}")


if __name__ == "__main__":
    raise SystemExit(main())
