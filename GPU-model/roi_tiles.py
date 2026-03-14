import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def build_roi_mask(depth: np.ndarray, threshold: float) -> np.ndarray:
    return (depth <= threshold).astype(np.uint8)


def mask_rgb(rgb: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    masked = rgb.copy()
    masked[roi_mask == 0] = 0
    return masked


def get_scanline_stream(mask: np.ndarray) -> List[Tuple[int, int, int]]:
    stream: List[Tuple[int, int, int]] = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            stream.append((x, y, int(mask[y, x])))
    return stream


def calculate_tile_counts_direct(
    mask: np.ndarray,
    tile_w: int,
    tile_h: int,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, int]]]:
    image_h, image_w = mask.shape
    tile_rows = (image_h + tile_h - 1) // tile_h
    tile_cols = (image_w + tile_w - 1) // tile_w
    pad_h = tile_rows * tile_h - image_h
    pad_w = tile_cols * tile_w - image_w
    padded = np.pad(mask.astype(np.uint8), ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    reshaped = padded.reshape(tile_rows, tile_h, tile_cols, tile_w)
    tile_pixel_counts = reshaped.sum(axis=(1, 3)).astype(np.int32)

    windows_with_order: List[Tuple[int, int, int, int, int]] = []
    discovery_order = 0
    for tile_row in range(tile_rows):
        for tile_col in range(tile_cols):
            if tile_pixel_counts[tile_row, tile_col] > 0:
                windows_with_order.append((tile_col * tile_w, tile_row * tile_h, tile_w, tile_h, discovery_order))
                discovery_order += 1
    return tile_pixel_counts, windows_with_order


def calculate_windows_online(
    stream: List[Tuple[int, int, int]],
    image_h: int,
    image_w: int,
    tile_w: int,
    tile_h: int,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, int]], np.ndarray]:
    tile_rows = (image_h + tile_h - 1) // tile_h
    tile_cols = (image_w + tile_w - 1) // tile_w
    tile_mask = np.zeros((tile_rows, tile_cols), dtype=np.uint8)
    tile_pixel_counts = np.zeros((tile_rows, tile_cols), dtype=np.int32)
    windows_dict = {}
    discovery_order = 0

    for x, y, pixel_val in stream:
        if pixel_val != 1:
            continue

        window_x = (x // tile_w) * tile_w
        window_y = (y // tile_h) * tile_h
        key = (window_x, window_y)
        if key not in windows_dict:
            windows_dict[key] = discovery_order
            discovery_order += 1

        tile_row = window_y // tile_h
        tile_col = window_x // tile_w
        tile_pixel_counts[tile_row, tile_col] += 1

    windows_with_order: List[Tuple[int, int, int, int, int]] = []
    for (window_x, window_y), order in sorted(windows_dict.items(), key=lambda item: item[1]):
        windows_with_order.append((window_x, window_y, tile_w, tile_h, order))

    return tile_mask, windows_with_order, tile_pixel_counts


def activate_tiles_from_counts(tile_pixel_counts: np.ndarray, min_active_pixels: int) -> np.ndarray:
    min_count = max(1, int(min_active_pixels))
    return (tile_pixel_counts >= min_count).astype(np.uint8)


def active_tile_boxes(tile_mask: np.ndarray, tile_size: int, image_h: int, image_w: int) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    tile_rows, tile_cols = np.nonzero(tile_mask)
    for tile_row, tile_col in zip(tile_rows.tolist(), tile_cols.tolist()):
        y0 = tile_row * tile_size
        x0 = tile_col * tile_size
        y1 = min(image_h, y0 + tile_size)
        x1 = min(image_w, x0 + tile_size)
        boxes.append((y0, x0, y1, x1))
    return boxes


def draw_tile_overlay(
    rgb: np.ndarray,
    tile_mask: np.ndarray,
    tile_width: int,
    tile_height: int,
    windows_with_order: List[Tuple[int, int, int, int, int]],
) -> np.ndarray:
    overlay = rgb.copy()
    image_h, image_w = rgb.shape[:2]

    for row in range(0, image_h, tile_height):
        overlay[row : row + 1, :, :] = 255
    for col in range(0, image_w, tile_width):
        overlay[:, col : col + 1, :] = 255

    if tile_width == tile_height:
        boxes = active_tile_boxes(tile_mask, tile_width, image_h, image_w)
    else:
        boxes = []
        tile_rows, tile_cols = np.nonzero(tile_mask)
        for tile_row, tile_col in zip(tile_rows.tolist(), tile_cols.tolist()):
            y0 = tile_row * tile_height
            x0 = tile_col * tile_width
            y1 = min(image_h, y0 + tile_height)
            x1 = min(image_w, x0 + tile_width)
            boxes.append((y0, x0, y1, x1))

    for y0, x0, y1, x1 in boxes:
        overlay[y0:y1, x0:x0 + 2, :] = np.array([255, 0, 0], dtype=np.uint8)
        overlay[y0:y1, max(x0, x1 - 2) : x1, :] = np.array([255, 0, 0], dtype=np.uint8)
        overlay[y0:y0 + 2, x0:x1, :] = np.array([255, 0, 0], dtype=np.uint8)
        overlay[max(y0, y1 - 2) : y1, x0:x1, :] = np.array([255, 0, 0], dtype=np.uint8)

    for x, y, w, h, order in windows_with_order:
        label = np.array([(37 * (order + 1)) % 255, (91 * (order + 1)) % 255, (173 * (order + 1)) % 255], dtype=np.uint8)
        y0 = y + min(3, max(0, h - 1))
        x0 = x + min(3, max(0, w - 1))
        y1 = min(image_h, y0 + 4)
        x1 = min(image_w, x0 + 8)
        overlay[y0:y1, x0:x1, :] = label
    return overlay


def process_scene(
    scene_path: str,
    depth_threshold: float,
    tile_width: int,
    tile_height: int,
    min_active_pixels: int,
    output_dir: str,
    tile_count_method: str = "direct",
) -> Path:
    scene = np.load(scene_path)
    rgb = scene["rgb"]
    depth = scene["depth"]
    roi_mask = build_roi_mask(depth, depth_threshold)
    return process_rgb_mask(
        rgb=rgb,
        roi_mask=roi_mask,
        output_dir=output_dir,
        tile_width=tile_width,
        tile_height=tile_height,
        min_active_pixels=min_active_pixels,
        extra_arrays={"depth": depth},
        tile_count_method=tile_count_method,
    )


def process_rgb_mask(
    rgb: np.ndarray,
    roi_mask: np.ndarray,
    output_dir: str,
    tile_width: int,
    tile_height: int,
    min_active_pixels: int,
    extra_arrays: dict | None = None,
    tile_count_method: str = "direct",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tile_data = build_tile_data_from_rgb_mask(
        rgb=rgb,
        roi_mask=roi_mask,
        tile_width=tile_width,
        tile_height=tile_height,
        min_active_pixels=min_active_pixels,
        extra_arrays=extra_arrays,
        tile_count_method=tile_count_method,
    )

    np.savez(output_path / "roi_tiles.npz", **tile_data)
    Image.fromarray((roi_mask * 255).astype(np.uint8), mode="L").save(output_path / "roi_mask.png")
    Image.fromarray(tile_data["masked_rgb"], mode="RGB").save(output_path / "masked_rgb.png")
    windows_with_order = [tuple(int(v) for v in row) for row in tile_data["windows_with_order"].tolist()]
    Image.fromarray(
        draw_tile_overlay(rgb, tile_data["tile_mask"], tile_width, tile_height, windows_with_order), mode="RGB"
    ).save(output_path / "tile_overlay.png")

    print(f"Saved ROI bundle: {output_path / 'roi_tiles.npz'}")
    print(f"Pixels processed in scanline stream: {tile_data['stream_length']}")
    print(f"ROI pixels: {int(roi_mask.sum())}")
    print(f"Active tiles: {int(tile_data['tile_mask'].sum())}/{tile_data['tile_mask'].size}")
    print(f"Min active pixels per tile: {max(1, int(min_active_pixels))}")
    print(f"Tile size: {tile_width}x{tile_height}")
    print(f"Windows discovered online: {len(tile_data['windows_with_order'])}")
    return output_path


def build_tile_data_from_rgb_mask(
    rgb: np.ndarray,
    roi_mask: np.ndarray,
    tile_width: int,
    tile_height: int,
    min_active_pixels: int,
    extra_arrays: dict | None = None,
    tile_count_method: str = "direct",
) -> dict:
    masked_rgb = mask_rgb(rgb, roi_mask)
    if tile_count_method == "scanline":
        stream = get_scanline_stream(roi_mask)
        _, windows_with_order, tile_pixel_counts = calculate_windows_online(
            stream=stream,
            image_h=roi_mask.shape[0],
            image_w=roi_mask.shape[1],
            tile_w=tile_width,
            tile_h=tile_height,
        )
        stream_length = len(stream)
    elif tile_count_method == "direct":
        tile_pixel_counts, windows_with_order = calculate_tile_counts_direct(
            mask=roi_mask,
            tile_w=tile_width,
            tile_h=tile_height,
        )
        stream_length = int(roi_mask.shape[0] * roi_mask.shape[1])
    else:
        raise ValueError(f"Unknown tile_count_method: {tile_count_method}")

    tile_mask = activate_tiles_from_counts(tile_pixel_counts, min_active_pixels)
    discovery_order = np.full(tile_mask.shape, fill_value=-1, dtype=np.int32)
    for x, y, _, _, order in windows_with_order:
        discovery_order[y // tile_height, x // tile_width] = order

    save_dict = {
        "rgb": rgb,
        "roi_mask": roi_mask,
        "masked_rgb": masked_rgb,
        "tile_mask": tile_mask,
        "discovery_order": discovery_order,
        "tile_pixel_counts": tile_pixel_counts,
        "tile_width": np.int32(tile_width),
        "tile_height": np.int32(tile_height),
        "min_active_pixels": np.int32(max(1, min_active_pixels)),
    }
    if extra_arrays:
        save_dict.update(extra_arrays)
    save_dict["windows_with_order"] = np.array(windows_with_order, dtype=np.int32) if windows_with_order else np.zeros((0, 5), dtype=np.int32)
    save_dict["stream_length"] = np.int32(stream_length)
    save_dict["tile_count_method"] = np.array(tile_count_method)
    return save_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ROI and scanline-discovered active tiles from a synthetic RGB/depth scene.")
    parser.add_argument("--scene", type=str, default="generated_demo/synthetic_scene.npz")
    parser.add_argument("--depth-threshold", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=None, help="If set, applies the same size to width and height.")
    parser.add_argument("--tile-width", type=int, default=16)
    parser.add_argument("--tile-height", type=int, default=16)
    parser.add_argument("--min-active-pixels", type=int, default=1)
    parser.add_argument("--tile-count-method", type=str, choices=["direct", "scanline"], default="direct")
    parser.add_argument("--output-dir", type=str, default="generated_demo")
    args = parser.parse_args()

    tile_width = args.tile_size if args.tile_size is not None else args.tile_width
    tile_height = args.tile_size if args.tile_size is not None else args.tile_height

    process_scene(
        scene_path=args.scene,
        depth_threshold=args.depth_threshold,
        tile_width=tile_width,
        tile_height=tile_height,
        min_active_pixels=args.min_active_pixels,
        output_dir=args.output_dir,
        tile_count_method=args.tile_count_method,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
