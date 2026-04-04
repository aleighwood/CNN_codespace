import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
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


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.28,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_grid_overview(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return

    _apply_plot_style()
    min_active_values = sorted({int(row["min_active_pixels"]) for row in rows})
    tile_width_values = sorted({int(row["tile_width"]) for row in rows})
    tile_height_values = sorted({int(row["tile_height"]) for row in rows})
    row_lookup = {
        (int(row["min_active_pixels"]), int(row["tile_width"]), int(row["tile_height"])): row
        for row in rows
    }

    metric_specs = [
        ("sparse_ms", "Mean Sparse Latency (ms)", lambda row: float(row["sparse_ms"]), "viridis"),
        ("active_ratio", "Mean Active Ratio (%)", lambda row: 100.0 * float(row["active_ratio"]), "magma_r"),
        ("sparse_top1_acc", "Sparse Top-1 Accuracy (%)", lambda row: float(row["sparse_top1_acc"]), "plasma"),
    ]
    metric_ranges = []
    for _, _, value_fn, _ in metric_specs:
        values = [value_fn(row) for row in rows]
        metric_ranges.append((min(values), max(values)))

    fig, axes = plt.subplots(
        len(min_active_values),
        len(metric_specs),
        figsize=(11.2, 2.55 * len(min_active_values) + 0.6),
        squeeze=False,
    )
    colorbar_images = [None] * len(metric_specs)

    for col_idx, (_, title, _, _) in enumerate(metric_specs):
        axes[0][col_idx].set_title(title)

    for row_idx, min_active_pixels in enumerate(min_active_values):
        for col_idx, (_, _, value_fn, cmap) in enumerate(metric_specs):
            vmin, vmax = metric_ranges[col_idx]
            grid = np.full((len(tile_height_values), len(tile_width_values)), np.nan, dtype=float)
            for height_idx, tile_height in enumerate(tile_height_values):
                for width_idx, tile_width in enumerate(tile_width_values):
                    row = row_lookup.get((min_active_pixels, tile_width, tile_height))
                    if row is not None:
                        grid[height_idx, width_idx] = value_fn(row)

            ax = axes[row_idx][col_idx]
            image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            colorbar_images[col_idx] = image
            ax.set_xticks(range(len(tile_width_values)))
            ax.set_xticklabels(tile_width_values)
            ax.set_yticks(range(len(tile_height_values)))
            ax.set_yticklabels(tile_height_values)
            ax.set_xlabel("tile width")
            if col_idx == 0:
                ax.set_ylabel(f"tile height\nminpix={min_active_pixels}")
            else:
                ax.set_ylabel("tile height")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    for col_idx, image in enumerate(colorbar_images):
        cbar = fig.colorbar(image, ax=axes[:, col_idx], fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

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


def evaluate_grid(
    roi_input_paths: list[Path],
    tile_width_values: list[int],
    tile_height_values: list[int],
    min_active_pixel_values: list[int],
    tile_count_method: str,
    workers: int,
    runner: SparseMobileNetRunner,
) -> list[dict]:
    results = []
    configs = list(product(min_active_pixel_values, tile_width_values, tile_height_values))
    total_configs = len(configs)
    for config_index, (min_active_pixels, tile_width, tile_height) in enumerate(configs, start=1):
        print(
            f"[grid] config {config_index}/{total_configs}: min_active_pixels={min_active_pixels} tile_width={tile_width} tile_height={tile_height}"
        )
        summary = evaluate_config(
            roi_input_paths=roi_input_paths,
            tile_width=tile_width,
            tile_height=tile_height,
            min_active_pixels=min_active_pixels,
            tile_count_method=tile_count_method,
            workers=workers,
            runner=runner,
        )
        results.append(summary)
        print(
            f"min_active_pixels={min_active_pixels} tile_width={tile_width} tile_height={tile_height}: dense_unmasked_top1={summary['dense_unmasked_top1_acc']:.2f}% dense_masked_top1={summary['dense_masked_top1_acc']:.2f}% sparse_top1={summary['sparse_top1_acc']:.2f}% sparse_ms={summary['sparse_ms']:.3f}"
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

    grid_rows = evaluate_grid(
        roi_input_paths=roi_input_paths,
        tile_width_values=list(range(args.tile_width_start, args.tile_width_stop + 1)),
        tile_height_values=list(range(args.tile_height_start, args.tile_height_stop + 1)),
        min_active_pixel_values=list(range(args.min_active_pixels_start, args.min_active_pixels_stop + 1)),
        tile_count_method=args.tile_count_method,
        workers=args.workers,
        runner=runner,
    )

    write_csv(output_dir / "grid_search.csv", grid_rows)

    plot_grid_overview(grid_rows, output_dir / "grid_search_overview.png")

    print(f"Processed images: {len(roi_input_paths)}")
    print(f"Saved dataset sweep outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
