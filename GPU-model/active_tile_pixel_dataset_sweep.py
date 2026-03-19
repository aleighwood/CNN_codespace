import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mobile_net import MobileNetV1
from sparse_mobilenet import SparseMobileNetRunner, build_layer_masks, image_to_normalized_tensor, label_from_roi_input_path


def list_roi_inputs(roi_dataset_dir: str) -> list[Path]:
    return sorted(Path(roi_dataset_dir).rglob("roi_input.npz"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(rows: list[dict], x_key: str, title: str, output_path: Path) -> None:
    x = [row[x_key] for row in rows]
    sparse_ms = [row["sparse_ms"] for row in rows]
    active_tiles = [row["active_tiles"] for row in rows]
    sparse_top1 = [row["sparse_top1_acc"] for row in rows]
    dense_masked_top1 = [row["dense_masked_top1_acc"] for row in rows]
    dense_unmasked_top1 = [row["dense_unmasked_top1_acc"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(x, sparse_ms, color="tab:blue", marker="o", label="mean sparse wall time (ms)")
    ax1.set_xlabel(x_key.replace("_", " "))
    ax1.set_ylabel("wall time (ms)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(x, active_tiles, color="tab:red", marker="s", label="mean active tiles")
    ax2.set_ylabel("active tiles", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    ax3.plot(x, sparse_top1, color="tab:green", marker="^", label="sparse masked top-1 (%)")
    ax3.plot(x, dense_masked_top1, color="tab:olive", marker="v", label="dense masked top-1 (%)")
    ax3.plot(x, dense_unmasked_top1, color="tab:purple", marker="d", label="dense unmasked top-1 (%)")
    ax3.set_ylabel("top-1 accuracy (%)", color="tab:green")
    ax3.tick_params(axis="y", labelcolor="tab:green")

    lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_eval_bundle_worker(args: tuple[str, int, int, int, str]) -> dict:
    roi_input_path, tile_width, tile_height, min_active_pixels, tile_count_method = args
    bundle = np.load(roi_input_path)
    if "label" in bundle:
        label = int(bundle["label"])
    else:
        label = label_from_roi_input_path(roi_input_path)
    pixel_masks, tile_masks = build_layer_masks(
        roi_mask=bundle["roi_mask"],
        tile_width=tile_width,
        tile_height=tile_height,
        min_active_pixels=min_active_pixels,
        tile_count_method=tile_count_method,
    )
    return {
        "rgb": bundle["rgb"],
        "masked_rgb": bundle["masked_rgb"],
        "label": label,
        "pixel_masks": pixel_masks,
        "tile_masks": tile_masks,
        "active_tiles": int(sum(mask.sum() for mask in tile_masks)),
        "total_tiles": int(sum(mask.size for mask in tile_masks)),
    }


def topk_hits(probs: torch.Tensor, label: int) -> tuple[int, int]:
    pred_top1 = int(torch.argmax(probs, dim=1).item())
    _, pred_top5 = probs.topk(5, dim=1)
    top1 = int(pred_top1 == label)
    top5 = int((pred_top5 == label).any().item())
    return top1, top5


def prediction_agreement(a: torch.Tensor, b: torch.Tensor) -> int:
    return int(torch.argmax(a, dim=1).item() == torch.argmax(b, dim=1).item())


def evaluate_config(
    roi_input_paths: list[Path],
    tile_width: int,
    tile_height: int,
    min_active_pixels: int,
    tile_count_method: str,
    workers: int,
    runner: SparseMobileNetRunner,
) -> dict:
    total_images = len(roi_input_paths)
    job_args = [(str(path), tile_width, tile_height, min_active_pixels, tile_count_method) for path in roi_input_paths]

    print(f"  building per-layer tile masks on {workers} CPU workers")
    eval_bundles = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for image_index, bundle in enumerate(executor.map(build_eval_bundle_worker, job_args), start=1):
            eval_bundles.append(bundle)
            if image_index % 50 == 0 or image_index == total_images:
                print(f"  tile generation {image_index}/{total_images} images")

    sparse_top1 = 0
    sparse_top5 = 0
    dense_masked_top1 = 0
    dense_masked_top5 = 0
    dense_unmasked_top1 = 0
    dense_unmasked_top5 = 0
    sparse_times = []
    dense_masked_times = []
    dense_unmasked_times = []
    active_tiles = []
    active_ratios = []
    sparse_vs_dense_masked_mean_abs_diffs = []
    sparse_vs_dense_masked_max_abs_diffs = []
    dense_masked_vs_unmasked_mean_abs_diffs = []
    dense_masked_vs_unmasked_max_abs_diffs = []
    sparse_vs_unmasked_mean_abs_diffs = []
    sparse_vs_unmasked_max_abs_diffs = []
    sparse_vs_dense_masked_agree = 0
    dense_masked_vs_unmasked_agree = 0
    sparse_vs_unmasked_agree = 0

    for image_index, bundle in enumerate(eval_bundles, start=1):
        x_masked = image_to_normalized_tensor(bundle["masked_rgb"], runner.device)
        x_unmasked = image_to_normalized_tensor(bundle["rgb"], runner.device)

        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        sparse_probs, active_tile_count, total_tile_count = runner.sparse_forward(
            image_tensor=x_masked,
            pixel_masks=bundle["pixel_masks"],
            tile_masks=bundle["tile_masks"],
            tile_width=tile_width,
            tile_height=tile_height,
        )
        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        sparse_times.append((time.perf_counter() - start) * 1000.0)

        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        dense_masked_probs = runner.dense_semantic_forward(x_masked, bundle["pixel_masks"])
        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        dense_masked_times.append((time.perf_counter() - start) * 1000.0)

        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        dense_unmasked_probs = runner.dense_unmasked_forward(x_unmasked)
        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        dense_unmasked_times.append((time.perf_counter() - start) * 1000.0)

        top1, top5 = topk_hits(sparse_probs, bundle["label"])
        sparse_top1 += top1
        sparse_top5 += top5

        top1, top5 = topk_hits(dense_masked_probs, bundle["label"])
        dense_masked_top1 += top1
        dense_masked_top5 += top5

        top1, top5 = topk_hits(dense_unmasked_probs, bundle["label"])
        dense_unmasked_top1 += top1
        dense_unmasked_top5 += top5

        active_tiles.append(active_tile_count)
        active_ratios.append(active_tile_count / max(1, total_tile_count))
        sparse_vs_dense_masked = torch.abs(sparse_probs - dense_masked_probs)
        dense_masked_vs_unmasked = torch.abs(dense_masked_probs - dense_unmasked_probs)
        sparse_vs_unmasked = torch.abs(sparse_probs - dense_unmasked_probs)
        sparse_vs_dense_masked_mean_abs_diffs.append(float(sparse_vs_dense_masked.mean().item()))
        sparse_vs_dense_masked_max_abs_diffs.append(float(sparse_vs_dense_masked.max().item()))
        dense_masked_vs_unmasked_mean_abs_diffs.append(float(dense_masked_vs_unmasked.mean().item()))
        dense_masked_vs_unmasked_max_abs_diffs.append(float(dense_masked_vs_unmasked.max().item()))
        sparse_vs_unmasked_mean_abs_diffs.append(float(sparse_vs_unmasked.mean().item()))
        sparse_vs_unmasked_max_abs_diffs.append(float(sparse_vs_unmasked.max().item()))
        sparse_vs_dense_masked_agree += prediction_agreement(sparse_probs, dense_masked_probs)
        dense_masked_vs_unmasked_agree += prediction_agreement(dense_masked_probs, dense_unmasked_probs)
        sparse_vs_unmasked_agree += prediction_agreement(sparse_probs, dense_unmasked_probs)

        if image_index % 25 == 0 or image_index == total_images:
            print(f"  full-model eval {image_index}/{total_images} images")

    n = max(1, total_images)
    return {
        "tile_width": tile_width,
        "tile_height": tile_height,
        "min_active_pixels": min_active_pixels,
        "num_images": total_images,
        "sparse_ms": float(np.mean(sparse_times)),
        "dense_masked_ms": float(np.mean(dense_masked_times)),
        "dense_unmasked_ms": float(np.mean(dense_unmasked_times)),
        "active_tiles": float(np.mean(active_tiles)),
        "active_ratio": float(np.mean(active_ratios)),
        "sparse_vs_dense_masked_mean_abs_diff": float(np.mean(sparse_vs_dense_masked_mean_abs_diffs)),
        "sparse_vs_dense_masked_max_abs_diff": float(np.max(sparse_vs_dense_masked_max_abs_diffs)),
        "dense_masked_vs_unmasked_mean_abs_diff": float(np.mean(dense_masked_vs_unmasked_mean_abs_diffs)),
        "dense_masked_vs_unmasked_max_abs_diff": float(np.max(dense_masked_vs_unmasked_max_abs_diffs)),
        "sparse_vs_unmasked_mean_abs_diff": float(np.mean(sparse_vs_unmasked_mean_abs_diffs)),
        "sparse_vs_unmasked_max_abs_diff": float(np.max(sparse_vs_unmasked_max_abs_diffs)),
        "sparse_top1_acc": 100.0 * sparse_top1 / n,
        "sparse_top5_acc": 100.0 * sparse_top5 / n,
        "dense_masked_top1_acc": 100.0 * dense_masked_top1 / n,
        "dense_masked_top5_acc": 100.0 * dense_masked_top5 / n,
        "dense_unmasked_top1_acc": 100.0 * dense_unmasked_top1 / n,
        "dense_unmasked_top5_acc": 100.0 * dense_unmasked_top5 / n,
        "sparse_vs_dense_masked_pred_agreement": 100.0 * sparse_vs_dense_masked_agree / n,
        "dense_masked_vs_unmasked_pred_agreement": 100.0 * dense_masked_vs_unmasked_agree / n,
        "sparse_vs_unmasked_pred_agreement": 100.0 * sparse_vs_unmasked_agree / n,
    }


def evaluate_sweep(
    roi_input_paths: list[Path],
    sweep_key: str,
    sweep_values: list[int],
    default_tile_width: int,
    default_tile_height: int,
    default_min_active_pixels: int,
    tile_count_method: str,
    workers: int,
    runner: SparseMobileNetRunner,
) -> list[dict]:
    results = []
    total_configs = len(sweep_values)
    for config_index, sweep_value in enumerate(sweep_values, start=1):
        tile_width = default_tile_width
        tile_height = default_tile_height
        min_active_pixels = default_min_active_pixels

        if sweep_key == "tile_width":
            tile_width = sweep_value
        elif sweep_key == "tile_height":
            tile_height = sweep_value
        elif sweep_key == "min_active_pixels":
            min_active_pixels = sweep_value

        print(f"[{sweep_key}] config {config_index}/{total_configs}: value={sweep_value}")
        summary = evaluate_config(
            roi_input_paths=roi_input_paths,
            tile_width=tile_width,
            tile_height=tile_height,
            min_active_pixels=min_active_pixels,
            tile_count_method=tile_count_method,
            workers=workers,
            runner=runner,
        )
        summary[sweep_key] = sweep_value
        results.append(summary)
        print(
            f"{sweep_key}={sweep_value}: dense_unmasked_top1={summary['dense_unmasked_top1_acc']:.2f}% dense_masked_top1={summary['dense_masked_top1_acc']:.2f}% sparse_top1={summary['sparse_top1_acc']:.2f}% sparse_ms={summary['sparse_ms']:.3f}"
        )
    return results


def load_runner(weights_path: str, device_name: str, chunk_tiles: int) -> SparseMobileNetRunner:
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MobileNetV1()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return SparseMobileNetRunner(model=model, device=device, chunk_tiles=chunk_tiles)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep full sparse MobileNet parameters across the ROI-generated image dataset.")
    parser.add_argument("--roi-dataset-dir", type=str, default="dataset_roi_frames")
    parser.add_argument("--weights", type=str, default="my_mobilenet_with_weights.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk-tiles", type=int, default=32)
    parser.add_argument("--default-tile-width", type=int, default=16)
    parser.add_argument("--default-tile-height", type=int, default=16)
    parser.add_argument("--default-min-active-pixels", type=int, default=1)
    parser.add_argument("--tile-count-method", type=str, choices=["direct", "scanline"], default="direct")
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--min-active-pixels-start", type=int, default=1)
    parser.add_argument("--min-active-pixels-stop", type=int, default=5)
    parser.add_argument("--tile-width-start", type=int, default=14)
    parser.add_argument("--tile-width-stop", type=int, default=18)
    parser.add_argument("--tile-height-start", type=int, default=14)
    parser.add_argument("--tile-height-stop", type=int, default=18)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="active_tile_pixel_dataset_sweep")
    args = parser.parse_args()

    roi_input_paths = list_roi_inputs(args.roi_dataset_dir)
    if args.max_images is not None:
        roi_input_paths = roi_input_paths[: max(0, args.max_images)]
    if not roi_input_paths:
        raise SystemExit(f"No roi_input.npz files found under {args.roi_dataset_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = load_runner(args.weights, args.device, args.chunk_tiles)

    threshold_rows = evaluate_sweep(
        roi_input_paths=roi_input_paths,
        sweep_key="min_active_pixels",
        sweep_values=list(range(args.min_active_pixels_start, args.min_active_pixels_stop + 1)),
        default_tile_width=args.default_tile_width,
        default_tile_height=args.default_tile_height,
        default_min_active_pixels=args.default_min_active_pixels,
        tile_count_method=args.tile_count_method,
        workers=args.workers,
        runner=runner,
    )
    tile_width_rows = evaluate_sweep(
        roi_input_paths=roi_input_paths,
        sweep_key="tile_width",
        sweep_values=list(range(args.tile_width_start, args.tile_width_stop + 1)),
        default_tile_width=args.default_tile_width,
        default_tile_height=args.default_tile_height,
        default_min_active_pixels=args.default_min_active_pixels,
        tile_count_method=args.tile_count_method,
        workers=args.workers,
        runner=runner,
    )
    tile_height_rows = evaluate_sweep(
        roi_input_paths=roi_input_paths,
        sweep_key="tile_height",
        sweep_values=list(range(args.tile_height_start, args.tile_height_stop + 1)),
        default_tile_width=args.default_tile_width,
        default_tile_height=args.default_tile_height,
        default_min_active_pixels=args.default_min_active_pixels,
        tile_count_method=args.tile_count_method,
        workers=args.workers,
        runner=runner,
    )

    write_csv(output_dir / "threshold_sweep.csv", threshold_rows)
    write_csv(output_dir / "tile_width_sweep.csv", tile_width_rows)
    write_csv(output_dir / "tile_height_sweep.csv", tile_height_rows)

    plot_sweep(threshold_rows, "min_active_pixels", "Dataset Sweep: Min Active Pixels", output_dir / "threshold_sweep.png")
    plot_sweep(tile_width_rows, "tile_width", "Dataset Sweep: Tile Width", output_dir / "tile_width_sweep.png")
    plot_sweep(tile_height_rows, "tile_height", "Dataset Sweep: Tile Height", output_dir / "tile_height_sweep.png")

    print(f"Processed images: {len(roi_input_paths)}")
    print(f"Saved dataset sweep outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
