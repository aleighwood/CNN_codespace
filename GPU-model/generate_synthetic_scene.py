import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def build_rgb_image(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    rgb = np.zeros((size, size, 3), dtype=np.uint8)

    rgb[..., 0] = np.clip(40 + 150 * (xx / max(1, size - 1)), 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(30 + 180 * (yy / max(1, size - 1)), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(180 - 100 * (xx / max(1, size - 1)), 0, 255).astype(np.uint8)

    center = size // 2
    radius = size // 5
    circle = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    rgb[circle] = np.array([245, 200, 40], dtype=np.uint8)

    box_half = size // 12
    rgb[center - box_half : center + box_half, center + radius // 2 : center + radius // 2 + 2 * box_half] = np.array(
        [220, 60, 60], dtype=np.uint8
    )
    return rgb


def build_depth_map(size: int, near_depth: float, far_depth: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    center = size // 2
    radius = size // 5

    depth = np.full((size, size), far_depth, dtype=np.float32)
    circle = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    depth[circle] = near_depth

    inner_radius = max(4, radius // 3)
    inner_circle = (xx - center) ** 2 + (yy - center) ** 2 <= inner_radius ** 2
    depth[inner_circle] = max(0.05, near_depth * 0.7)
    return depth


def depth_to_preview(depth: np.ndarray) -> np.ndarray:
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max <= depth_min:
        return np.zeros(depth.shape, dtype=np.uint8)
    normalized = (depth - depth_min) / (depth_max - depth_min)
    return np.clip(255.0 * (1.0 - normalized), 0, 255).astype(np.uint8)


def generate_scene(size: int, near_depth: float, far_depth: float, output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rgb = build_rgb_image(size)
    depth = build_depth_map(size, near_depth=near_depth, far_depth=far_depth)

    np.savez(output_path / "synthetic_scene.npz", rgb=rgb, depth=depth)
    Image.fromarray(rgb, mode="RGB").save(output_path / "rgb.png")
    Image.fromarray(depth_to_preview(depth), mode="L").save(output_path / "depth_preview.png")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic RGB frame and pseudo depth frame.")
    parser.add_argument("--size", type=int, default=224, help="Width and height of the generated square frame.")
    parser.add_argument("--near-depth", type=float, default=0.35, help="Pseudo depth for the foreground object.")
    parser.add_argument("--far-depth", type=float, default=1.0, help="Pseudo depth for the background.")
    parser.add_argument("--output-dir", type=str, default="generated_demo", help="Where to save generated files.")
    args = parser.parse_args()

    output_dir = generate_scene(args.size, args.near_depth, args.far_depth, args.output_dir)

    print(f"Saved scene bundle: {output_dir / 'synthetic_scene.npz'}")
    print(f"Saved RGB preview: {output_dir / 'rgb.png'}")
    print(f"Saved depth preview: {output_dir / 'depth_preview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
