import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from roi_tiles import process_scene
from tiled_conv_demo import evaluate_tiled_conv


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(rows: list[dict], x_key: str, title: str, output_path: Path) -> None:
    x = [row[x_key] for row in rows]
    tiled_ms = [row["tiled_ms"] for row in rows]
    active_tiles = [row["active_tiles"] for row in rows]
    correctness = [row["mean_abs_diff"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x, tiled_ms, color="tab:blue", marker="o", label="tiled wall time (ms)")
    ax1.set_xlabel(x_key.replace("_", " "))
    ax1.set_ylabel("wall time (ms)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(x, active_tiles, color="tab:red", marker="s", label="active tiles")
    ax2.set_ylabel("active tiles", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    ax3.plot(x, correctness, color="tab:green", marker="^", label="mean abs diff")
    ax3.set_ylabel("mean abs diff", color="tab:green")
    ax3.tick_params(axis="y", labelcolor="tab:green")

    lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_config(
    scene_path: str,
    output_dir: Path,
    depth_threshold: float,
    tile_width: int,
    tile_height: int,
    min_active_pixels: int,
    device: str,
    out_channels: int,
    seed: int,
) -> dict:
    config_dir = output_dir / f"tw_{tile_width}_th_{tile_height}_minpix_{min_active_pixels}"
    process_scene(
        scene_path=scene_path,
        depth_threshold=depth_threshold,
        tile_width=tile_width,
        tile_height=tile_height,
        min_active_pixels=min_active_pixels,
        output_dir=str(config_dir),
    )
    metrics = evaluate_tiled_conv(
        roi_bundle_path=str(config_dir / "roi_tiles.npz"),
        device_name=device,
        out_channels=out_channels,
        seed=seed,
        output_dir=None,
    )
    metrics["tile_width"] = tile_width
    metrics["tile_height"] = tile_height
    metrics["min_active_pixels"] = min_active_pixels
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep tile activation threshold and tile dimensions for one synthetic image.")
    parser.add_argument("--scene", type=str, default="generated_demo/synthetic_scene.npz")
    parser.add_argument("--depth-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-channels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--default-tile-width", type=int, default=16)
    parser.add_argument("--default-tile-height", type=int, default=16)
    parser.add_argument("--min-active-pixels-start", type=int, default=1)
    parser.add_argument("--min-active-pixels-stop", type=int, default=32)
    parser.add_argument("--tile-width-start", type=int, default=8)
    parser.add_argument("--tile-width-stop", type=int, default=24)
    parser.add_argument("--tile-height-start", type=int, default=8)
    parser.add_argument("--tile-height-stop", type=int, default=24)
    parser.add_argument("--output-dir", type=str, default="active_tile_pixel_one_image_sweep")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_active_rows = []
    for min_active_pixels in range(args.min_active_pixels_start, args.min_active_pixels_stop + 1):
        min_active_rows.append(
            evaluate_config(
                scene_path=args.scene,
                output_dir=output_dir / "threshold_configs",
                depth_threshold=args.depth_threshold,
                tile_width=args.default_tile_width,
                tile_height=args.default_tile_height,
                min_active_pixels=min_active_pixels,
                device=args.device,
                out_channels=args.out_channels,
                seed=args.seed,
            )
        )

    tile_width_rows = []
    for tile_width in range(args.tile_width_start, args.tile_width_stop + 1):
        tile_width_rows.append(
            evaluate_config(
                scene_path=args.scene,
                output_dir=output_dir / "tile_width_configs",
                depth_threshold=args.depth_threshold,
                tile_width=tile_width,
                tile_height=args.default_tile_height,
                min_active_pixels=1,
                device=args.device,
                out_channels=args.out_channels,
                seed=args.seed,
            )
        )

    tile_height_rows = []
    for tile_height in range(args.tile_height_start, args.tile_height_stop + 1):
        tile_height_rows.append(
            evaluate_config(
                scene_path=args.scene,
                output_dir=output_dir / "tile_height_configs",
                depth_threshold=args.depth_threshold,
                tile_width=args.default_tile_width,
                tile_height=tile_height,
                min_active_pixels=1,
                device=args.device,
                out_channels=args.out_channels,
                seed=args.seed,
            )
        )

    write_csv(output_dir / "threshold_sweep.csv", min_active_rows)
    write_csv(output_dir / "tile_width_sweep.csv", tile_width_rows)
    write_csv(output_dir / "tile_height_sweep.csv", tile_height_rows)

    plot_sweep(
        min_active_rows,
        x_key="min_active_pixels",
        title="Min Active Pixels Sweep",
        output_path=output_dir / "threshold_sweep.png",
    )
    plot_sweep(
        tile_width_rows,
        x_key="tile_width",
        title="Tile Width Sweep",
        output_path=output_dir / "tile_width_sweep.png",
    )
    plot_sweep(
        tile_height_rows,
        x_key="tile_height",
        title="Tile Height Sweep",
        output_path=output_dir / "tile_height_sweep.png",
    )

    print(f"Saved sweep outputs to: {output_dir}")
    print(f"Threshold sweep rows: {len(min_active_rows)}")
    print(f"Tile width sweep rows: {len(tile_width_rows)}")
    print(f"Tile height sweep rows: {len(tile_height_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
