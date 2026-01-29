import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.MobileNet_tf import MobileNetFunctional


def parse_shape(value):
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("input-shape must be H,W,C")
    return tuple(parts)


def load_dataset(npz_path, num_samples, input_shape, seed):
    if npz_path:
        data = np.load(npz_path)
        images = data["images"].astype(np.float32)
        labels = data["labels"].astype(np.int64) if "labels" in data else None
    else:
        rng = np.random.default_rng(seed)
        images = rng.standard_normal((num_samples, *input_shape), dtype=np.float32)
        labels = None

    if num_samples and images.shape[0] > num_samples:
        images = images[:num_samples]
        if labels is not None:
            labels = labels[:num_samples]

    return images, labels


def top1_accuracy(predictions, labels):
    if labels is None:
        return None
    labels = labels.reshape(-1)
    pred_labels = np.argmax(predictions, axis=1)
    return float(np.mean(pred_labels == labels))


def evaluate_keras(model, images, labels, batch_size):
    start = time.perf_counter()
    predictions = model.predict(images, batch_size=batch_size, verbose=0)
    duration = time.perf_counter() - start
    latency_ms = (duration / images.shape[0]) * 1000
    return {
        "latency_ms": latency_ms,
        "accuracy": top1_accuracy(predictions, labels),
        "model_size_kb": None,
    }


def write_tflite(converter, output_path):
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return output_path


def representative_dataset(images, input_shape, num_samples, seed):
    if images is None:
        rng = np.random.default_rng(seed)
        for _ in range(num_samples):
            yield [rng.standard_normal((1, *input_shape), dtype=np.float32)]
        return
    limit = min(images.shape[0], num_samples)
    for i in range(limit):
        yield [images[i : i + 1]]


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


def evaluate_tflite(model_path, images, labels):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    outputs = []
    start = time.perf_counter()
    for sample in images:
        input_data = sample[np.newaxis, ...]
        input_data = quantize_input(input_data, input_details)
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details["index"])
        output_data = dequantize_output(output_data, output_details)
        outputs.append(output_data.reshape(output_data.shape[0], -1)[0])
    duration = time.perf_counter() - start
    outputs = np.stack(outputs, axis=0)
    latency_ms = (duration / images.shape[0]) * 1000

    return {
        "latency_ms": latency_ms,
        "accuracy": top1_accuracy(outputs, labels),
        "model_size_kb": os.path.getsize(model_path) / 1024.0,
    }


def format_metric(value, precision=4):
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def print_results(results):
    for name, metrics in results.items():
        latency = format_metric(metrics["latency_ms"], precision=2)
        accuracy = format_metric(metrics["accuracy"], precision=4)
        size = format_metric(metrics["model_size_kb"], precision=2)
        print(f"{name}: latency_ms={latency}, accuracy={accuracy}, size_kb={size}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MobileNet quantization variants.")
    parser.add_argument("--weights", type=str, default=None, help="Path to Keras weights (.h5).")
    parser.add_argument(
        "--keras-app-mobilenet",
        action="store_true",
        help="Use tf.keras.applications.MobileNet instead of MobileNetFunctional.",
    )
    parser.add_argument(
        "--keras-weights",
        type=str,
        default=None,
        help="Pass through to tf.keras.applications.MobileNet weights arg (e.g., 'imagenet').",
    )
    parser.add_argument("--npz", type=str, default=None, help="NPZ with 'images' and optional 'labels'.")
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--input-shape", type=parse_shape, default="224,224,3")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--representative-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="quantized_models")
    parser.add_argument("--skip-keras", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    images, labels = load_dataset(args.npz, args.num_samples, args.input_shape, args.seed)

    if args.keras_app_mobilenet:
        model = tf.keras.applications.MobileNet(
            input_shape=args.input_shape,
            classes=args.classes,
            weights=args.keras_weights,
        )
    else:
        model = MobileNetFunctional(input_shape=args.input_shape, classes=args.classes)
    if args.weights:
        model.load_weights(args.weights)

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    if not args.skip_keras:
        results["keras_fp32"] = evaluate_keras(model, images, labels, args.batch_size)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    dynamic_path = os.path.join(args.output_dir, "mobilenet_dynamic.tflite")
    write_tflite(converter, dynamic_path)
    results["tflite_dynamic"] = evaluate_tflite(dynamic_path, images, labels)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    float16_path = os.path.join(args.output_dir, "mobilenet_float16.tflite")
    write_tflite(converter, float16_path)
    results["tflite_float16"] = evaluate_tflite(float16_path, images, labels)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        images, args.input_shape, args.representative_samples, args.seed
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_path = os.path.join(args.output_dir, "mobilenet_int8.tflite")
    write_tflite(converter, int8_path)
    results["tflite_int8"] = evaluate_tflite(int8_path, images, labels)

    print_results(results)


if __name__ == "__main__":
    main()
