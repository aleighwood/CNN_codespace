#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from pathlib import Path

import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.MobileNet_tf import MobileNetFunctional
from tensorflow.keras.applications.mobilenet import preprocess_input


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize MobileNet using ILSVRC2012 val images.")
    parser.add_argument("--val-dir", type=str, required=True, help="Directory with ILSVRC2012 val JPEGs.")
    parser.add_argument("--weights", type=str, default="mobilenet_imagenet.weights.h5")
    parser.add_argument("--output", type=str, default="quantized_models/mobilenet_int8_ilsvrc2012.tflite")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of images to use (from start).")
    parser.add_argument(
        "--representative-samples",
        type=int,
        default=5000,
        help="Number of representative samples to use for calibration.",
    )
    parser.add_argument("--resize-shorter", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--no-center-crop", action="store_true", help="Use direct resize instead of center crop.")
    return parser.parse_args()


def build_file_list(val_dir, num_samples):
    files = sorted(glob.glob(os.path.join(val_dir, "*.JPEG")))
    if not files:
        raise RuntimeError(f"No JPEGs found in {val_dir}")
    if num_samples and len(files) > num_samples:
        files = files[:num_samples]
    return files


def make_preprocess(resize_shorter, crop_size, center_crop):
    def _fn(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32) * 255.0
        if center_crop:
            shape = tf.shape(img)
            height = tf.cast(shape[0], tf.float32)
            width = tf.cast(shape[1], tf.float32)
            scale = tf.cast(resize_shorter, tf.float32) / tf.minimum(height, width)
            new_h = tf.cast(tf.math.round(height * scale), tf.int32)
            new_w = tf.cast(tf.math.round(width * scale), tf.int32)
            img = tf.image.resize(img, (new_h, new_w), method="bilinear", antialias=True)
            img = tf.image.resize_with_crop_or_pad(img, crop_size, crop_size)
        else:
            img = tf.image.resize(img, (crop_size, crop_size), method="bilinear", antialias=True)
        img = preprocess_input(img)
        return img

    return _fn


def representative_dataset(ds, num_samples):
    count = 0
    for img in ds:
        yield [img.numpy()[None, ...]]
        count += 1
        if num_samples and count >= num_samples:
            break


def main():
    args = parse_args()
    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        raise SystemExit(f"val-dir not found: {val_dir}")

    files = build_file_list(str(val_dir), args.num_samples)
    preprocess = make_preprocess(args.resize_shorter, args.crop_size, not args.no_center_crop)
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    model = MobileNetFunctional(input_shape=(args.crop_size, args.crop_size, 3), classes=1000)
    if args.weights:
        model.load_weights(args.weights)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(ds, args.representative_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
