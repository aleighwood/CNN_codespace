import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an ImageNet NPZ file for quantization benchmarking."
    )
    parser.add_argument("--output", default="imagenet_val.npz", help="Output NPZ path.")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples to export.")
    parser.add_argument("--split", default="validation", help="TFDS split to use.")
    parser.add_argument("--dataset", default="imagenet2012", help="TFDS dataset name.")
    parser.add_argument(
        "--imagenet-v2",
        action="store_true",
        help="Use imagenet_v2/matched-frequency with the test split.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Crop/resize size (square).")
    parser.add_argument(
        "--center-crop",
        action="store_true",
        help="Resize shorter side to --resize-shorter then center crop to --image-size.",
    )
    parser.add_argument(
        "--resize-shorter",
        type=int,
        default=256,
        help="Shorter-side resize for center crop.",
    )
    parser.add_argument("--data-dir", default=None, help="TFDS data_dir for prepared datasets.")
    parser.add_argument(
        "--manual-dir",
        default=None,
        help="Directory with manually downloaded ImageNet files (TFDS manual_dir).",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset order.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffle.")
    return parser.parse_args()


def preprocess(image, image_size, center_crop, resize_shorter):
    image = tf.cast(image, tf.float32)
    if center_crop:
        shape = tf.shape(image)
        height = tf.cast(shape[0], tf.float32)
        width = tf.cast(shape[1], tf.float32)
        scale = tf.cast(resize_shorter, tf.float32) / tf.minimum(height, width)
        new_height = tf.cast(tf.math.round(height * scale), tf.int32)
        new_width = tf.cast(tf.math.round(width * scale), tf.int32)
        image = tf.image.resize(image, (new_height, new_width), method="bilinear")
        image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    else:
        image = tf.image.resize(image, (image_size, image_size), method="bilinear")
    return tf.keras.applications.mobilenet.preprocess_input(image)


def build_dataset(args):
    download_config = None
    if args.manual_dir:
        download_config = tfds.download.DownloadConfig(manual_dir=args.manual_dir)

    builder = tfds.builder(args.dataset, data_dir=args.data_dir)
    builder.download_and_prepare(download_config=download_config)
    return builder.as_dataset(
        split=args.split,
        shuffle_files=args.shuffle,
        as_supervised=True,
    )


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)

    if args.imagenet_v2:
        args.dataset = "imagenet_v2/matched-frequency"
        args.split = "test"
        args.manual_dir = None

    dataset = build_dataset(args)
    images = []
    labels = []

    for image, label in dataset:
        image = preprocess(
            image,
            image_size=args.image_size,
            center_crop=args.center_crop,
            resize_shorter=args.resize_shorter,
        )
        images.append(image.numpy())
        labels.append(int(label.numpy()))
        if args.num_samples and len(images) >= args.num_samples:
            break

    if not images:
        raise RuntimeError("No samples found. Check dataset availability and split name.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        images=np.stack(images, axis=0),
        labels=np.array(labels, dtype=np.int64),
    )
    print(f"Wrote {len(images)} samples to {output_path}")


if __name__ == "__main__":
    main()
