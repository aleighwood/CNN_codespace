#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.MobileNet_tf import MobileNetFunctional
from tensorflow.keras.applications.mobilenet import preprocess_input


def parse_args():
    parser = argparse.ArgumentParser(description="Compare FP32, TFLite int8, and golden int8 outputs.")
    parser.add_argument("--val-dir", type=str, default="ILSVRC2012_val")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--weights", type=str, default="mobilenet_imagenet.weights.h5")
    parser.add_argument("--tflite", type=str, default="quantized_models/mobilenet_int8_ilsvrc2012_5000.tflite")
    parser.add_argument("--class-index-json", type=str, default="scripts/imagenet_class_index.json")
    parser.add_argument("--resize-shorter", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    return parser.parse_args()


def read_mem32(path):
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = int(line, 16) & 0xFFFFFFFF
            if v & 0x80000000:
                v -= 0x100000000
            vals.append(v)
    return np.array(vals, dtype=np.int32)


def preprocess_image(path, resize_shorter, crop_size):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) * 255.0
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    scale = tf.cast(resize_shorter, tf.float32) / tf.minimum(tf.cast(h, tf.float32), tf.cast(w, tf.float32))
    new_h = tf.cast(tf.math.round(tf.cast(h, tf.float32) * scale), tf.int32)
    new_w = tf.cast(tf.math.round(tf.cast(w, tf.float32) * scale), tf.int32)
    img = tf.image.resize(img, (new_h, new_w), method="bilinear", antialias=True)
    img = tf.image.resize_with_crop_or_pad(img, crop_size, crop_size)
    img = preprocess_input(img)
    return img


def quantize_input(data, input_details):
    dtype = input_details["dtype"]
    if dtype == np.float32:
        return data.astype(np.float32)
    scale, zero_point = input_details.get("quantization", (0.0, 0))
    if scale == 0:
        return data.astype(dtype)
    quantized = np.round(data / scale + zero_point)
    if dtype == np.int8:
        quantized = np.clip(quantized, -128, 127)
    elif dtype == np.uint8:
        quantized = np.clip(quantized, 0, 255)
    return quantized.astype(dtype)


def dequantize_output(data, output_details):
    dtype = output_details["dtype"]
    if dtype == np.float32:
        return data.astype(np.float32)
    scale, zero_point = output_details.get("quantization", (0.0, 0))
    if scale == 0:
        return data.astype(np.float32)
    return (data.astype(np.float32) - zero_point) * scale


def topk_names(vec, class_index, k=5):
    idxs = np.argsort(-vec)[:k]
    out = []
    for idx in idxs:
        wnid, name = class_index[str(int(idx))]
        out.append((int(idx), name, wnid, float(vec[idx])))
    return out


def main():
    args = parse_args()
    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        raise SystemExit(f"val-dir not found: {val_dir}")

    with open(args.class_index_json, "r", encoding="utf-8") as f:
        class_index = json.load(f)

    files = sorted(val_dir.glob("*.JPEG"))[: args.num_samples]
    if not files:
        raise SystemExit("No JPEGs found")

    model = MobileNetFunctional(input_shape=(args.crop_size, args.crop_size, 3), classes=1000)
    model.load_weights(args.weights)

    interpreter = tf.lite.Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    for path in files:
        img = preprocess_image(str(path), args.resize_shorter, args.crop_size)
        img_np = img.numpy()[None, ...]

        fp32 = model.predict(img_np, verbose=0)[0]

        tflite_in = quantize_input(img_np, input_details)
        interpreter.set_tensor(input_details["index"], tflite_in)
        interpreter.invoke()
        tflite_out = interpreter.get_tensor(output_details["index"])
        tflite_out = dequantize_output(tflite_out, output_details)[0]

        subprocess.run(
            [
                sys.executable,
                "scripts/image_to_input_mem.py",
                "--image",
                str(path),
                "--input-shape",
                f"{args.crop_size},{args.crop_size},3",
                "--center-crop",
                "--resize-shorter",
                str(args.resize_shorter),
                "--output-mem",
                "rtl/mem/input_rand.mem",
                "--tflite",
                args.tflite,
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/gen_golden_fc.py",
                "--input-shape",
                f"{args.crop_size},{args.crop_size},3",
                "--input-mem-in",
                "rtl/mem/input_rand.mem",
                "--input-mem",
                "rtl/mem/input_rand.mem",
                "--expected-logits-mem",
                "rtl/mem/fc_logits_expected.mem",
                "--expected-mem",
                "rtl/mem/fc_expected.mem",
                "--mem-dir",
                "rtl/mem",
                "--tflite",
                args.tflite,
                "--q31",
            ],
            check=True,
        )
        golden_logits = read_mem32("rtl/mem/fc_logits_expected.mem").astype(np.float32)

        print(f"\nImage: {path.name}")
        print("FP32 top-5:")
        for idx, name, wnid, score in topk_names(fp32, class_index, 5):
            print(f"  {idx:4d} {name} ({wnid}) score={score:.4f}")
        print("TFLite int8 top-5:")
        for idx, name, wnid, score in topk_names(tflite_out, class_index, 5):
            print(f"  {idx:4d} {name} ({wnid}) score={score:.4f}")
        print("Golden int8 logits top-5:")
        for idx, name, wnid, score in topk_names(golden_logits, class_index, 5):
            print(f"  {idx:4d} {name} ({wnid}) logit={score:.0f}")


if __name__ == "__main__":
    main()
