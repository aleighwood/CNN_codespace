import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from active_tile_pixel_dataset_sweep import plot_grid_overview
from roi_tiles import calculate_tile_counts_direct


INPUT_DIR = Path("active_tile_pixel_dataset_sweep")
ROI_DIR = Path("dataset_roi_frames")
MINPIX_PALETTE = {
    1: "#0072B2",
    2: "#E69F00",
    3: "#009E73",
    4: "#D55E00",
    5: "#CC79A7",
}


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
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.28,
            "lines.linewidth": 1.4,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfd",
            "savefig.facecolor": "white",
        }
    )


def make_config_label(tile_width: int, tile_height: int, min_active_pixels: int) -> str:
    return f"w={tile_width}, h={tile_height}, minpix={min_active_pixels}"


def load_dataset_sweep_rows() -> list[dict]:
    rows = read_csv_rows(INPUT_DIR / "grid_search.csv")
    for row in rows:
        row["config_label"] = make_config_label(
            tile_width=int(row["tile_width"]),
            tile_height=int(row["tile_height"]),
            min_active_pixels=int(row["min_active_pixels"]),
        )
    return rows


def compute_mean_coverage_by_config(sweep_rows: list[dict]) -> list[dict]:
    roi_input_paths = sorted(ROI_DIR.rglob("roi_input.npz"))
    if not roi_input_paths:
        raise SystemExit(f"No roi_input.npz files found under {ROI_DIR}")

    configs = []
    for row in sweep_rows:
        configs.append(
            {
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


def _scale_marker_sizes(values: np.ndarray, size_min: float = 70.0, size_max: float = 240.0) -> np.ndarray:
    if len(values) == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(low, high):
        return np.full_like(values, 0.5 * (size_min + size_max), dtype=float)
    return size_min + (values - low) * (size_max - size_min) / (high - low)


def _short_config_text(row: dict) -> str:
    return f"{int(row['tile_width'])}x{int(row['tile_height'])},m{int(row['min_active_pixels'])}"


def _legend_handles_for_minpix(minpix_values: np.ndarray) -> list[Line2D]:
    handles = []
    for minpix in sorted({int(v) for v in minpix_values.tolist()}):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                label=str(minpix),
                markerfacecolor=MINPIX_PALETTE.get(minpix, "#666666"),
                markeredgecolor="white",
                markersize=8,
            )
        )
    return handles


def _legend_handles_for_tile_area(tile_area: np.ndarray, size_map: np.ndarray) -> list[Line2D]:
    unique_areas = np.unique(tile_area.astype(int))
    if len(unique_areas) > 3:
        unique_areas = np.array([unique_areas[0], unique_areas[len(unique_areas) // 2], unique_areas[-1]])
    handles = []
    for area in unique_areas:
        index = int(np.where(tile_area.astype(int) == int(area))[0][0])
        marker_size = max(6.5, np.sqrt(size_map[index]) * 0.8)
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                label=f"{int(area)}",
                markerfacecolor="#9aa3af",
                markeredgecolor="white",
                alpha=0.9,
                markersize=marker_size,
            )
        )
    return handles


def _annotate_selected_configs(
    ax: plt.Axes,
    summary_rows: list[dict],
    x: np.ndarray,
    y: np.ndarray,
    selected_indices: np.ndarray,
) -> None:
    x_mid = float(np.median(x))
    y_mid = float(np.median(y))
    for idx in selected_indices:
        idx = int(idx)
        on_left = float(x[idx]) <= x_mid
        on_bottom = float(y[idx]) <= y_mid
        ax.annotate(
            _short_config_text(summary_rows[idx]),
            xy=(x[idx], y[idx]),
            xytext=(7 if on_left else -7, 6 if on_bottom else -6),
            textcoords="offset points",
            fontsize=8.2,
            color="#111827",
            ha="left" if on_left else "right",
            va="bottom" if on_bottom else "top",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
        )


def load_or_compute_summary_rows(sweep_rows: list[dict]) -> tuple[list[dict], bool]:
    summary_path = INPUT_DIR / "config_tile_summary.csv"
    expected_keys = {"tile_width", "tile_height", "min_active_pixels"}
    if summary_path.exists():
        existing_rows = read_csv_rows(summary_path)
        if existing_rows and all(expected_keys.issubset(row.keys()) for row in existing_rows):
            expected_configs = {
                (int(row["tile_width"]), int(row["tile_height"]), int(row["min_active_pixels"])) for row in sweep_rows
            }
            existing_configs = {
                (int(row["tile_width"]), int(row["tile_height"]), int(row["min_active_pixels"]))
                for row in existing_rows
            }
            if expected_configs == existing_configs:
                print(f"Using existing summary CSV: {summary_path}")
                return existing_rows, False
            print(f"Summary CSV config mismatch, recomputing: {summary_path}")
        else:
            print(f"Summary CSV missing required columns, recomputing: {summary_path}")

    summary_rows = compute_mean_coverage_by_config(sweep_rows)
    write_csv_rows(summary_path, summary_rows)
    print(f"Saved recomputed summary CSV: {summary_path}")
    return summary_rows, True


def plot_coverage_efficiency(summary_rows: list[dict]) -> None:
    _apply_plot_style()
    x = np.array([row["mean_active_tiles"] for row in summary_rows], dtype=float)
    y = np.array([100.0 * row["mean_coverage_efficiency"] for row in summary_rows], dtype=float)
    minpix = np.array([int(row["min_active_pixels"]) for row in summary_rows], dtype=int)
    tile_area = np.array([int(row["tile_width"]) * int(row["tile_height"]) for row in summary_rows], dtype=float)
    size = _scale_marker_sizes(tile_area)
    colors = [MINPIX_PALETTE.get(int(value), "#666666") for value in minpix.tolist()]

    fig, ax = plt.subplots(figsize=(9.4, 6.6), constrained_layout=True)
    ax.scatter(
        x,
        y,
        c=colors,
        s=size,
        edgecolors="white",
        linewidths=0.55,
        alpha=0.93,
    )
    ax.set_title("Coverage Efficiency vs Mean Active Tiles")
    ax.set_ylabel("Coverage efficiency (%)")
    ax.set_xlabel("Mean active tiles")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(7))
    ax.grid(True, linestyle="-", linewidth=0.65, alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    top_indices = np.argsort(y)[-4:]
    _annotate_selected_configs(ax=ax, summary_rows=summary_rows, x=x, y=y, selected_indices=top_indices)

    minpix_legend = ax.legend(
        handles=_legend_handles_for_minpix(minpix),
        title="min active pixels",
        loc="lower right",
        frameon=True,
    )
    ax.add_artist(minpix_legend)
    ax.legend(
        handles=_legend_handles_for_tile_area(tile_area=tile_area, size_map=size),
        title="tile area (px)",
        loc="lower left",
        frameon=True,
    )

    fig.savefig(INPUT_DIR / "coverage_efficiency_by_config.png", dpi=220)
    plt.close(fig)


def plot_runtime_vs_active_tiles(summary_rows: list[dict]) -> None:
    _apply_plot_style()
    x = np.array([row["mean_active_tiles"] for row in summary_rows], dtype=float)
    y = np.array([row["mean_sparse_ms"] for row in summary_rows], dtype=float)
    minpix = np.array([int(row["min_active_pixels"]) for row in summary_rows], dtype=int)
    tile_area = np.array([int(row["tile_width"]) * int(row["tile_height"]) for row in summary_rows], dtype=float)
    size = _scale_marker_sizes(tile_area)
    colors = [MINPIX_PALETTE.get(int(value), "#666666") for value in minpix.tolist()]
    pearson = pearson_corr(x, y)
    spearman = spearman_corr(x, y)

    fig, ax = plt.subplots(figsize=(9.4, 6.6), constrained_layout=True)
    ax.scatter(
        x,
        y,
        c=colors,
        s=size,
        edgecolors="white",
        linewidths=0.55,
        alpha=0.93,
    )

    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    slope, intercept = np.polyfit(x, y, deg=1)
    ax.plot(x_line, slope * x_line + intercept, color="#1f2937", linewidth=1.25, linestyle="--", label="trend")

    ax.set_title("Mean Sparse Latency vs Mean Active Tiles")
    ax.set_xlabel("Mean active tiles")
    ax.set_ylabel("Mean sparse latency (ms)")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(7))
    ax.grid(True, linestyle="-", linewidth=0.65, alpha=0.24)
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
        bbox={"facecolor": "white", "edgecolor": "#6b7280", "boxstyle": "round,pad=0.35"},
    )

    fast_order = np.argsort(y)
    x_threshold = float(np.quantile(x, 0.15))
    selected_fast = [int(idx) for idx in fast_order if float(x[int(idx)]) >= x_threshold][:2]
    if len(selected_fast) < 2:
        selected_fast = [int(idx) for idx in fast_order[:2]]
    selected = np.unique(np.array(selected_fast + [int(np.argmax(y))], dtype=int))
    _annotate_selected_configs(ax=ax, summary_rows=summary_rows, x=x, y=y, selected_indices=selected)

    minpix_legend = ax.legend(
        handles=_legend_handles_for_minpix(minpix),
        title="min active pixels",
        loc="upper right",
        frameon=True,
    )
    ax.add_artist(minpix_legend)
    ax.legend(
        handles=_legend_handles_for_tile_area(tile_area=tile_area, size_map=size),
        title="tile area (px)",
        loc="lower left",
        frameon=True,
    )
    fig.savefig(INPUT_DIR / "runtime_vs_active_tiles.png", dpi=220)
    plt.close(fig)


def main() -> int:
    csv_path = INPUT_DIR / "grid_search.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing CSV file: {csv_path}")
    rows = read_csv_rows(csv_path)
    if not rows:
        raise SystemExit(f"CSV file is empty: {csv_path}")
    plot_grid_overview(rows=rows, output_path=INPUT_DIR / "grid_search_overview.png")

    sweep_rows = load_dataset_sweep_rows()
    summary_rows, recomputed = load_or_compute_summary_rows(sweep_rows)
    plot_coverage_efficiency(summary_rows)
    plot_runtime_vs_active_tiles(summary_rows)

    print(f"Replotted graphs from CSV files under: {INPUT_DIR}")
    if not recomputed:
        print(f"Reused config summary CSV: {INPUT_DIR / 'config_tile_summary.csv'}")
    print(f"Saved coverage plot to: {INPUT_DIR / 'coverage_efficiency_by_config.png'}")
    print(f"Saved runtime correlation plot to: {INPUT_DIR / 'runtime_vs_active_tiles.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
