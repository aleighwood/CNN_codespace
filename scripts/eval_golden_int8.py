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

import scripts.eval_imagenet_val as eval_imagenet_val


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate golden int8 model on ILSVRC2012 val images.")
    parser.add_argument("--val-dir", type=str, default="ILSVRC2012_val")
    parser.add_argument(
        "--gt-file",
        type=str,
        default="ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
    )
    parser.add_argument(
        "--class-index-json",
        type=str,
        default=str(Path(__file__).with_name("imagenet_class_index.json")),
    )
    parser.add_argument(
        "--meta-mat",
        type=str,
        default="ILSVRC2012_devkit_t12/data/meta.mat",
    )
    parser.add_argument("--mem-dir", type=str, default="rtl/mem")
    parser.add_argument(
        "--tflite",
        type=str,
        default="quantized_models/mobilenet_int8_ilsvrc2012_5000.tflite",
        help="TFLite model for input quantization params.",
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--resize-shorter", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument(
        "--sanity-index",
        type=int,
        default=447,
        help="0-based index into sorted val images to sanity-check first.",
    )
    parser.add_argument("--skip-sanity", action="store_true")
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


def run_golden_for_image(img_path, mem_dir, crop_size, resize_shorter, input_scale, input_zp, tflite_path):
    input_mem = str(Path(mem_dir) / "input_rand.mem")
    logits_mem = str(Path(mem_dir) / "fc_logits_expected.mem")

    subprocess.run(
        [
            sys.executable,
            "scripts/image_to_input_mem.py",
            "--image",
            str(img_path),
            "--input-shape",
            f"{crop_size},{crop_size},3",
            "--center-crop",
            "--resize-shorter",
            str(resize_shorter),
            "--output-mem",
            input_mem,
            "--scale",
            str(input_scale),
            "--zero-point",
            str(input_zp),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/gen_golden_fc.py",
            "--input-shape",
            f"{crop_size},{crop_size},3",
            "--input-mem-in",
            input_mem,
            "--input-mem",
            input_mem,
            "--expected-logits-mem",
            logits_mem,
            "--expected-mem",
            str(Path(mem_dir) / "fc_expected.mem"),
            "--mem-dir",
            mem_dir,
            "--tflite",
            str(tflite_path),
            "--q31",
        ],
        check=True,
    )

    logits = read_mem32(logits_mem)
    return int(np.argmax(logits))


def main():
    args = parse_args()
    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        raise SystemExit(f"val-dir not found: {val_dir}")

    gt_file = Path(args.gt_file)
    if not gt_file.exists():
        raise SystemExit(f"gt-file not found: {gt_file}")

    class_index_json = Path(args.class_index_json)
    if not class_index_json.exists():
        raise SystemExit(f"class-index-json not found: {class_index_json}")

    meta_mat = Path(args.meta_mat)
    if not meta_mat.exists():
        raise SystemExit(f"meta-mat not found: {meta_mat}")

    wnids, wnid_to_keras = eval_imagenet_val.build_keras_label_map(
        str(class_index_json), str(meta_mat)
    )

    tflite_path = Path(args.tflite)
    if not tflite_path.exists():
        raise SystemExit(f"tflite not found: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    input_scale, input_zp = input_details.get("quantization", (None, None))
    if input_scale is None:
        raise SystemExit("TFLite input quantization not found.")

    labels = []
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            labels.append(int(line.strip()))
    labels = np.array(labels, dtype=np.int64)

    files = sorted(val_dir.glob("*.JPEG"))

    if not args.skip_sanity:
        if args.sanity_index < 0 or args.sanity_index >= len(files):
            raise SystemExit("sanity-index out of range")
        sanity_path = files[args.sanity_index]
        ilsvrc_id = labels[args.sanity_index]
        gt = wnid_to_keras[wnids[ilsvrc_id - 1]]
        pred = run_golden_for_image(
            sanity_path,
            args.mem_dir,
            args.crop_size,
            args.resize_shorter,
            input_scale,
            input_zp,
            tflite_path,
        )
        print(
            f"Sanity image idx={args.sanity_index} file={sanity_path.name} pred={pred} gt={gt} {'OK' if pred==gt else 'NO'}"
        )
        if pred != gt:
            raise SystemExit("Sanity check failed; aborting before full eval.")

    num_samples = min(args.num_samples, len(files))
    files = files[:num_samples]
    labels = labels[:num_samples]

    correct = 0
    for idx, path in enumerate(files):
        pred = run_golden_for_image(
            path,
            args.mem_dir,
            args.crop_size,
            args.resize_shorter,
            input_scale,
            input_zp,
            tflite_path,
        )
        ilsvrc_id = labels[idx]
        gt = wnid_to_keras[wnids[ilsvrc_id - 1]]
        correct += int(pred == gt)
        print(f"{idx + 1}/{num_samples} pred={pred} gt={gt} {'OK' if pred==gt else 'NO'}")

    acc = correct / num_samples if num_samples else 0.0
    print(f"Golden int8 top-1 accuracy on {num_samples} images: {acc:.4f}")


if __name__ == "__main__":
    main()
