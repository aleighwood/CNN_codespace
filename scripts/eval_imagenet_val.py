#!/usr/bin/env python3
import argparse
import glob
import io
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.MobileNet_tf import MobileNetFunctional
from tensorflow.keras.applications.mobilenet import preprocess_input


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MobileNet on ILSVRC2012 val set.")
    parser.add_argument("--val-dir", type=str, required=True, help="Directory with ILSVRC2012 val JPEGs.")
    parser.add_argument(
        "--gt-file",
        type=str,
        required=True,
        help="Path to ILSVRC2012_validation_ground_truth.txt (labels 1..1000).",
    )
    parser.add_argument("--weights", type=str, default="mobilenet_imagenet.weights.h5")
    parser.add_argument(
        "--class-index-json",
        type=str,
        default=str(Path(__file__).with_name("imagenet_class_index.json")),
        help="Path to imagenet_class_index.json (Keras class index mapping).",
    )
    parser.add_argument(
        "--meta-mat",
        type=str,
        default="",
        help="Path to ILSVRC2012 devkit meta.mat (for ILSVRC ID -> WNID mapping).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=0, help="Limit number of samples (0=all).")
    parser.add_argument("--resize-shorter", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--no-center-crop", action="store_true", help="Use direct resize instead of center crop.")
    parser.add_argument("--top5", action="store_true", help="Also compute top-5 accuracy.")
    parser.add_argument("--tflite", type=str, default="", help="Path to TFLite model to evaluate.")
    parser.add_argument("--tflite-only", action="store_true", help="Skip Keras evaluation; only TFLite.")
    return parser.parse_args()


def load_labels(gt_path, num_samples):
    labels = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(int(line))
    labels = np.array(labels, dtype=np.int64)
    if num_samples and labels.shape[0] > num_samples:
        labels = labels[:num_samples]
    return labels


MI = {
    "miINT8": 1,
    "miUINT8": 2,
    "miINT16": 3,
    "miUINT16": 4,
    "miINT32": 5,
    "miUINT32": 6,
    "miSINGLE": 7,
    "miDOUBLE": 9,
    "miINT64": 12,
    "miUINT64": 13,
    "miMATRIX": 14,
    "miCOMPRESSED": 15,
    "miUTF8": 16,
    "miUTF16": 17,
    "miUTF32": 18,
}
MI_TYPES = set(MI.values())

MX = {
    "mxCELL_CLASS": 1,
    "mxSTRUCT_CLASS": 2,
    "mxOBJECT_CLASS": 3,
    "mxCHAR_CLASS": 4,
    "mxSPARSE_CLASS": 5,
    "mxDOUBLE_CLASS": 6,
    "mxSINGLE_CLASS": 7,
    "mxINT8_CLASS": 8,
    "mxUINT8_CLASS": 9,
    "mxINT16_CLASS": 10,
    "mxUINT16_CLASS": 11,
    "mxINT32_CLASS": 12,
    "mxUINT32_CLASS": 13,
    "mxINT64_CLASS": 14,
    "mxUINT64_CLASS": 15,
}

DTYPE_MAP = {
    MI["miINT8"]: np.int8,
    MI["miUINT8"]: np.uint8,
    MI["miINT16"]: np.int16,
    MI["miUINT16"]: np.uint16,
    MI["miINT32"]: np.int32,
    MI["miUINT32"]: np.uint32,
    MI["miINT64"]: np.int64,
    MI["miUINT64"]: np.uint64,
    MI["miSINGLE"]: np.float32,
    MI["miDOUBLE"]: np.float64,
}


def _read_tag(fh):
    tag = fh.read(8)
    if len(tag) < 8:
        return None
    dtype, size = np.frombuffer(tag, dtype="<u4", count=2)
    small_bytes = dtype >> 16
    dt = dtype & 0xFFFF
    if small_bytes and dt in MI_TYPES and small_bytes <= 4:
        data = tag[4:4 + small_bytes]
        return int(dt), int(small_bytes), data, True
    return int(dtype), int(size), None, False


def _read_element(fh):
    tag = _read_tag(fh)
    if tag is None:
        return None
    dtype, size, data, is_small = tag
    if not is_small:
        data = fh.read(size)
        pad = (8 - (size % 8)) % 8
        if pad:
            fh.read(pad)
    return dtype, data


def _parse_matrix_header(data):
    buf = io.BytesIO(data)
    dtype, af = _read_element(buf)
    if dtype is None:
        return None
    flags = np.frombuffer(af, dtype="<u4", count=2)
    class_type = int(flags[0] & 0xFF)
    dtype, dims_data = _read_element(buf)
    dims = np.frombuffer(dims_data, dtype="<i4")
    dtype, name_data = _read_element(buf)
    name = name_data.decode("utf-8").strip("\x00")
    return class_type, dims, name, buf


def _parse_matrix_value(data):
    parsed = _parse_matrix_header(data)
    if parsed is None:
        return None
    class_type, dims, name, buf = parsed
    if class_type == MX["mxCHAR_CLASS"]:
        dtype, raw = _read_element(buf)
        if dtype in (MI["miINT8"], MI["miUINT8"], MI["miUTF8"]):
            return raw.decode("utf-8", errors="ignore").strip("\x00")
        if dtype in (MI["miINT16"], MI["miUINT16"], MI["miUTF16"]):
            arr = np.frombuffer(raw, dtype="<u2")
            return "".join(chr(c) for c in arr if c != 0)
        return ""
    if class_type in (
        MX["mxDOUBLE_CLASS"],
        MX["mxSINGLE_CLASS"],
        MX["mxINT8_CLASS"],
        MX["mxUINT8_CLASS"],
        MX["mxINT16_CLASS"],
        MX["mxUINT16_CLASS"],
        MX["mxINT32_CLASS"],
        MX["mxUINT32_CLASS"],
        MX["mxINT64_CLASS"],
        MX["mxUINT64_CLASS"],
    ):
        dtype, raw = _read_element(buf)
        if dtype is None:
            return None
        return np.frombuffer(raw, dtype=DTYPE_MAP.get(dtype, np.float64))
    return None


def _load_ilsvrc2012_wnids(meta_mat_path):
    with open(meta_mat_path, "rb") as fh:
        fh.read(128)
        while True:
            elem = _read_element(fh)
            if elem is None:
                break
            dtype, data = elem
            if dtype == MI["miCOMPRESSED"]:
                data = zlib.decompress(data)
                buf = io.BytesIO(data)
                elem = _read_element(buf)
                if elem is None:
                    continue
                dtype, data = elem
            if dtype != MI["miMATRIX"]:
                continue
            parsed = _parse_matrix_header(data)
            if parsed is None:
                continue
            class_type, dims, name, buf = parsed
            if name != "synsets":
                continue
            if class_type != MX["mxSTRUCT_CLASS"]:
                raise RuntimeError("meta.mat synsets is not a struct")
            dtype, fl_data = _read_element(buf)
            field_len = int(np.frombuffer(fl_data, dtype="<i4", count=1)[0])
            dtype, fn_data = _read_element(buf)
            fields = [
                fn_data[i:i + field_len].split(b"\x00", 1)[0].decode("utf-8")
                for i in range(0, len(fn_data), field_len)
            ]
            numel = int(np.prod(dims))
            wnids = [None] * numel
            ids = [None] * numel
            for idx in range(numel):
                for fname in fields:
                    elem = _read_element(buf)
                    if elem is None:
                        raise RuntimeError("Unexpected end of struct data")
                    f_dtype, f_data = elem
                    if f_dtype != MI["miMATRIX"]:
                        continue
                    if fname not in ("WNID", "ILSVRC2012_ID"):
                        continue
                    val = _parse_matrix_value(f_data)
                    if fname == "WNID":
                        wnids[idx] = val
                    else:
                        if val is None or len(val) == 0:
                            ids[idx] = None
                        else:
                            ids[idx] = int(val[0])
            low = [w for i, w in zip(ids, wnids) if i is not None and i <= 1000]
            if len(low) != 1000:
                raise RuntimeError("Unexpected number of low-level synsets")
            return low
    raise RuntimeError("synsets not found in meta.mat")


def build_keras_label_map(class_index_json, meta_mat_path):
    with open(class_index_json, "r", encoding="utf-8") as f:
        class_index = json.load(f)
    wnid_to_keras = {v[0]: int(k) for k, v in class_index.items()}
    if len(wnid_to_keras) != 1000:
        raise RuntimeError("Unexpected class index size; expected 1000 entries.")
    wnids = _load_ilsvrc2012_wnids(meta_mat_path)
    return wnids, wnid_to_keras


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


def evaluate_tflite(model_path, ds, labels, top5, log_every=1000):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    total = 0
    correct1 = 0
    correct5 = 0
    label_idx = 0
    for batch in ds:
        batch_np = batch.numpy()
        batch_size = batch_np.shape[0]
        for i in range(batch_size):
            sample = batch_np[i : i + 1]
            input_data = quantize_input(sample, input_details)
            interpreter.set_tensor(input_details["index"], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details["index"])
            output_data = dequantize_output(output_data, output_details)
            logits = output_data.reshape(-1)

            gt = labels[label_idx]
            label_idx += 1

            if int(np.argmax(logits)) == int(gt):
                correct1 += 1
            if top5:
                if int(gt) in np.argsort(-logits)[:5]:
                    correct5 += 1

            total += 1
            if log_every and total % log_every == 0:
                print(f"tflite progress: {total} samples")

    acc1 = correct1 / total if total else 0.0
    acc5 = correct5 / total if total else 0.0
    return acc1, acc5


def main():
    args = parse_args()
    val_dir = Path(args.val_dir)
    gt_file = Path(args.gt_file)
    class_index_json = Path(args.class_index_json)
    meta_mat = Path(args.meta_mat) if args.meta_mat else gt_file.parent / "meta.mat"

    if not val_dir.exists():
        raise SystemExit(f"val-dir not found: {val_dir}")
    if not gt_file.exists():
        raise SystemExit(f"gt-file not found: {gt_file}")
    if not class_index_json.exists():
        raise SystemExit(f"class-index-json not found: {class_index_json}")
    if not meta_mat.exists():
        raise SystemExit(f"meta-mat not found: {meta_mat}")

    files = build_file_list(str(val_dir), args.num_samples)
    labels = load_labels(str(gt_file), args.num_samples)

    if len(files) != labels.shape[0]:
        raise SystemExit(f"file count {len(files)} != label count {labels.shape[0]}")

    center_crop = not args.no_center_crop
    preprocess = make_preprocess(args.resize_shorter, args.crop_size, center_crop)

    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    wnids, wnid_to_keras = build_keras_label_map(str(class_index_json), str(meta_mat))
    gt = np.array([wnid_to_keras[wnids[i - 1]] for i in labels], dtype=np.int64)

    if not args.tflite_only:
        model = MobileNetFunctional(input_shape=(args.crop_size, args.crop_size, 3), classes=1000)
        if args.weights:
            model.load_weights(args.weights)

        preds = model.predict(ds, verbose=0)
        top1 = np.argmax(preds, axis=1)
        acc1 = float(np.mean(top1 == gt))
        print(f"top1 accuracy: {acc1:.4f}")

        if args.top5:
            top5 = np.argsort(-preds, axis=1)[:, :5]
            acc5 = float(np.mean([gt[i] in top5[i] for i in range(gt.shape[0])]))
            print(f"top5 accuracy: {acc5:.4f}")

    if args.tflite:
        tflite_acc1, tflite_acc5 = evaluate_tflite(args.tflite, ds, gt, args.top5)
        print(f"tflite top1 accuracy: {tflite_acc1:.4f}")
        if args.top5:
            print(f"tflite top5 accuracy: {tflite_acc5:.4f}")


if __name__ == "__main__":
    main()
