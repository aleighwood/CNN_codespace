import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from active_tile_pixel_dataset_sweep import plot_sweep
from roi_tiles import calculate_tile_counts_direct


INPUT_DIR = Path("active_tile_pixel_dataset_sweep")
ROI_DIR = Path("dataset_roi_frames")


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = value
                    continue
                try:
                    number = float(value)
                    parsed[key] = int(number) if number.is_integer() else number
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
        return rows


def write_csv_rows(path: Path, rows: list[dict]) -> None:
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
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.24,
            "lines.linewidth": 1.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def make_config_label(family: str, tile_width: int, tile_height: int, min_active_pixels: int) -> str:
    if family == "threshold":
        return f"minpix={min_active_pixels}"
    if family == "tile_width":
        return f"tile_w={tile_width}"
    return f"tile_h={tile_height}"


def load_dataset_sweep_rows() -> list[dict]:
    rows = []
    for csv_name, family in (
        ("threshold_sweep.csv", "threshold"),
        ("tile_width_sweep.csv", "tile_width"),
        ("tile_height_sweep.csv", "tile_height"),
    ):
        for row in read_csv_rows(INPUT_DIR / csv_name):
            row["family"] = family
            row["config_label"] = make_config_label(
                family=family,
                tile_width=int(row["tile_width"]),
                tile_height=int(row["tile_height"]),
                min_active_pixels=int(row["min_active_pixels"]),
            )
            rows.append(row)
    return rows


def compute_mean_coverage_by_config(sweep_rows: list[dict]) -> list[dict]:
    roi_input_paths = sorted(ROI_DIR.rglob("roi_input.npz"))
    if not roi_input_paths:
        raise SystemExit(f"No roi_input.npz files found under {ROI_DIR}")

    configs = []
    for row in sweep_rows:
        configs.append(
            {
                "family": row["family"],
                "config_label": row["config_label"],
                "tile_width": int(row["tile_width"]),
                "tile_height": int(row["tile_height"]),
                "min_active_pixels": int(row["min_active_pixels"]),
                "sparse_ms": float(row["sparse_ms"]),
                "active_tiles_mean": float(row["active_tiles"]),
                "sparse_top1_acc": float(row["sparse_top1_acc"]),
                "dense_masked_top1_acc": float(row["dense_masked_top1_acc"]),
                "dense_unmasked_top1_acc": float(row["dense_unmasked_top1_acc"]),
            }
        )

    coverage_sums = [0.0 for _ in configs]
    total_images = len(roi_input_paths)
    for image_index, roi_input_path in enumerate(roi_input_paths, start=1):
        bundle = np.load(roi_input_path)
        roi_mask = bundle["roi_mask"]
        for idx, config in enumerate(configs):
            tile_pixel_counts, _ = calculate_tile_counts_direct(
                mask=roi_mask,
                tile_w=config["tile_width"],
                tile_h=config["tile_height"],
            )
            active_mask = tile_pixel_counts >= max(1, config["min_active_pixels"])
            active_tile_count = int(active_mask.sum())
            if active_tile_count == 0:
                coverage = 0.0
            else:
                actual_pixels = int(tile_pixel_counts[active_mask].sum())
                tile_capacity = active_tile_count * config["tile_width"] * config["tile_height"]
                coverage = float(actual_pixels / tile_capacity)
            coverage_sums[idx] += coverage
        if image_index % 250 == 0 or image_index == total_images:
            print(f"coverage stats {image_index}/{total_images} images")

    summary_rows = []
    for idx, config in enumerate(configs):
        mean_coverage = coverage_sums[idx] / total_images
        summary_rows.append(
            {
                "family": config["family"],
                "config_label": config["config_label"],
                "tile_width": config["tile_width"],
                "tile_height": config["tile_height"],
                "min_active_pixels": config["min_active_pixels"],
                "mean_coverage_efficiency": mean_coverage,
                "mean_active_tiles": config["active_tiles_mean"],
                "mean_sparse_ms": config["sparse_ms"],
                "sparse_top1_acc": config["sparse_top1_acc"],
                "dense_masked_top1_acc": config["dense_masked_top1_acc"],
                "dense_unmasked_top1_acc": config["dense_unmasked_top1_acc"],
            }
        )
    return summary_rows


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(x), rankdata(y))


def plot_coverage_efficiency(summary_rows: list[dict]) -> None:
    _apply_plot_style()
    x = np.arange(len(summary_rows))
    y = [100.0 * row["mean_coverage_efficiency"] for row in summary_rows]
    labels = [row["config_label"] for row in summary_rows]

    family_colors = {"threshold": "#1f77b4", "tile_width": "#ff7f0e", "tile_height": "#2ca02c"}
    colors = [family_colors[row["family"]] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title("Mean Tile Coverage Efficiency by Configuration")
    ax.set_ylabel("Coverage efficiency (%)")
    ax.set_xlabel("Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=6, label=family.replace("_", " "))
        for family, color in family_colors.items()
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=True, title="Sweep family")
    fig.tight_layout()
    fig.savefig(INPUT_DIR / "coverage_efficiency_by_config.png", dpi=160)
    plt.close(fig)


def plot_runtime_vs_active_tiles(summary_rows: list[dict]) -> None:
    _apply_plot_style()
    family_colors = {"threshold": "#1f77b4", "tile_width": "#ff7f0e", "tile_height": "#2ca02c"}
    family_markers = {"threshold": "o", "tile_width": "s", "tile_height": "^"}

    x = np.array([row["mean_active_tiles"] for row in summary_rows], dtype=float)
    y = np.array([row["mean_sparse_ms"] for row in summary_rows], dtype=float)
    pearson = pearson_corr(x, y)
    spearman = spearman_corr(x, y)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for row in summary_rows:
        ax.scatter(
            row["mean_active_tiles"],
            row["mean_sparse_ms"],
            color=family_colors[row["family"]],
            marker=family_markers[row["family"]],
            s=70,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(
            row["config_label"],
            (row["mean_active_tiles"], row["mean_sparse_ms"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_title("Mean Sparse Latency vs Mean Active Tiles")
    ax.set_xlabel("Mean active tiles")
    ax.set_ylabel("Mean sparse latency (ms)")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    text = f"Pearson r = {pearson:.3f}\nSpearman rho = {spearman:.3f}"
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.35"},
    )

    legend_handles = [
        plt.Line2D([0], [0], color=family_colors[family], marker=family_markers[family], linestyle="", markersize=7, label=family.replace("_", " "))
        for family in ("threshold", "tile_width", "tile_height")
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, title="Sweep family")
    fig.tight_layout()
    fig.savefig(INPUT_DIR / "runtime_vs_active_tiles.png", dpi=160)
    plt.close(fig)


def main() -> int:
    for csv_name, x_key, title, png_name in (
        ("threshold_sweep.csv", "min_active_pixels", "Dataset Sweep: Min Active Pixels", "threshold_sweep.png"),
        ("tile_width_sweep.csv", "tile_width", "Dataset Sweep: Tile Width", "tile_width_sweep.png"),
        ("tile_height_sweep.csv", "tile_height", "Dataset Sweep: Tile Height", "tile_height_sweep.png"),
    ):
        csv_path = INPUT_DIR / csv_name
        if not csv_path.exists():
            raise SystemExit(f"Missing CSV file: {csv_path}")
        rows = read_csv_rows(csv_path)
        if not rows:
            raise SystemExit(f"CSV file is empty: {csv_path}")
        plot_sweep(rows=rows, x_key=x_key, title=title, output_path=INPUT_DIR / png_name)

    sweep_rows = load_dataset_sweep_rows()
    summary_rows = compute_mean_coverage_by_config(sweep_rows)
    write_csv_rows(INPUT_DIR / "config_tile_summary.csv", summary_rows)
    plot_coverage_efficiency(summary_rows)
    plot_runtime_vs_active_tiles(summary_rows)

    print(f"Replotted graphs from CSV files under: {INPUT_DIR}")
    print(f"Saved config summary CSV to: {INPUT_DIR / 'config_tile_summary.csv'}")
    print(f"Saved coverage plot to: {INPUT_DIR / 'coverage_efficiency_by_config.png'}")
    print(f"Saved runtime correlation plot to: {INPUT_DIR / 'runtime_vs_active_tiles.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
