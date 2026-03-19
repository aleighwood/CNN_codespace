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


def load_configurations() -> list[dict]:
    configs = []
    for csv_name, family in (
        ("threshold_sweep.csv", "threshold"),
        ("tile_width_sweep.csv", "tile_width"),
        ("tile_height_sweep.csv", "tile_height"),
    ):
        for row in read_csv_rows(INPUT_DIR / csv_name):
            config = {
                "family": family,
                "tile_width": int(row["tile_width"]),
                "tile_height": int(row["tile_height"]),
                "min_active_pixels": int(row["min_active_pixels"]),
            }
            if family == "threshold":
                config["label"] = f"minpix={config['min_active_pixels']}"
            elif family == "tile_width":
                config["label"] = f"tile_w={config['tile_width']}"
            else:
                config["label"] = f"tile_h={config['tile_height']}"
            configs.append(config)
    return configs


def list_roi_inputs() -> list[Path]:
    return sorted(ROI_DIR.rglob("roi_input.npz"))


def compute_tile_efficiency_rows(configs: list[dict]) -> list[dict]:
    roi_input_paths = list_roi_inputs()
    if not roi_input_paths:
        raise SystemExit(f"No roi_input.npz files found under {ROI_DIR}")

    rows = []
    total_images = len(roi_input_paths)
    for image_index, roi_input_path in enumerate(roi_input_paths, start=1):
        bundle = np.load(roi_input_path)
        roi_mask = bundle["roi_mask"]
        image_key = roi_input_path.parent.name
        roi_pixels = int(roi_mask.sum())

        for config in configs:
            tile_width = config["tile_width"]
            tile_height = config["tile_height"]
            min_active_pixels = config["min_active_pixels"]
            tile_pixel_counts, _ = calculate_tile_counts_direct(mask=roi_mask, tile_w=tile_width, tile_h=tile_height)
            active_mask = tile_pixel_counts >= max(1, min_active_pixels)
            active_tile_count = int(active_mask.sum())
            active_pixels = int(tile_pixel_counts[active_mask].sum()) if active_tile_count > 0 else 0
            tile_capacity = active_tile_count * tile_width * tile_height
            coverage = float(active_pixels / tile_capacity) if tile_capacity > 0 else 0.0
            rows.append(
                {
                    "image_index": image_index,
                    "image_key": image_key,
                    "family": config["family"],
                    "config_label": config["label"],
                    "tile_width": tile_width,
                    "tile_height": tile_height,
                    "min_active_pixels": min_active_pixels,
                    "roi_pixels": roi_pixels,
                    "active_tiles": active_tile_count,
                    "active_pixels_in_active_tiles": active_pixels,
                    "tile_capacity_pixels": tile_capacity,
                    "coverage": coverage,
                }
            )

        if image_index % 250 == 0 or image_index == total_images:
            print(f"tile efficiency stats {image_index}/{total_images} images")

    return rows


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.24,
            "lines.linewidth": 1.1,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_tile_efficiency(rows: list[dict], configs: list[dict]) -> None:
    _apply_plot_style()
    grouped = {}
    for row in rows:
        grouped.setdefault(row["config_label"], []).append(row)

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(configs), 1)))
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.6), sharex=True)
    active_ax, coverage_ax = axes

    for idx, config in enumerate(configs):
        label = config["label"]
        series = grouped.get(label, [])
        if not series:
            continue
        x = [row["image_index"] for row in series]
        active_y = [row["active_tiles"] for row in series]
        coverage_y = [100.0 * row["coverage"] for row in series]
        color = colors[idx % len(colors)]
        active_ax.plot(x, active_y, color=color, label=label)
        coverage_ax.plot(x, coverage_y, color=color, label=label)

    active_ax.set_title("Active Tiles Across Validation Images")
    active_ax.set_ylabel("Active tiles")
    active_ax.grid(True)

    coverage_ax.set_title("Average Active-Tile Coverage Across Validation Images")
    coverage_ax.set_xlabel("Image index")
    coverage_ax.set_ylabel("Coverage (%)")
    coverage_ax.grid(True)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = active_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=True, title="Configuration")
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(INPUT_DIR / "tile_efficiency_by_image.png", dpi=160)
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

    configs = load_configurations()
    tile_efficiency_rows = compute_tile_efficiency_rows(configs)
    write_csv_rows(INPUT_DIR / "image_tile_efficiency.csv", tile_efficiency_rows)
    plot_tile_efficiency(tile_efficiency_rows, configs)

    print(f"Replotted graphs from CSV files under: {INPUT_DIR}")
    print(f"Saved tile efficiency CSV to: {INPUT_DIR / 'image_tile_efficiency.csv'}")
    print(f"Saved tile efficiency plot to: {INPUT_DIR / 'tile_efficiency_by_image.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
