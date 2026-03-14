import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return x.to(device)


def tensor_to_rgb_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().squeeze(0)
    x = x.permute(1, 2, 0).numpy()
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x


def run_tiled_conv(
    x: torch.Tensor,
    tile_mask: np.ndarray,
    tile_width: int,
    tile_height: int,
    conv: nn.Conv2d,
) -> torch.Tensor:
    _, _, height, width = x.shape
    output = torch.zeros((1, conv.out_channels, height, width), device=x.device, dtype=x.dtype)
    padded = F.pad(x, (1, 1, 1, 1))

    active_rows, active_cols = np.nonzero(tile_mask)
    for tile_row, tile_col in zip(active_rows.tolist(), active_cols.tolist()):
        y0 = tile_row * tile_height
        x0 = tile_col * tile_width
        valid_h = min(tile_height, height - y0)
        valid_w = min(tile_width, width - x0)

        patch = padded[:, :, y0 : y0 + valid_h + 2, x0 : x0 + valid_w + 2]
        patch_out = F.conv2d(patch, conv.weight, conv.bias, stride=1, padding=0)
        output[:, :, y0 : y0 + valid_h, x0 : x0 + valid_w] = patch_out[:, :, :valid_h, :valid_w]
    return output


def benchmark(fn, use_cuda_sync: bool) -> Tuple[torch.Tensor, float]:
    if use_cuda_sync:
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    if use_cuda_sync:
        torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1000.0


def tile_mask_to_pixel_mask(
    tile_mask: np.ndarray, tile_width: int, tile_height: int, height: int, width: int, device: torch.device
) -> torch.Tensor:
    pixel_mask = np.zeros((height, width), dtype=np.float32)
    active_rows, active_cols = np.nonzero(tile_mask)
    for tile_row, tile_col in zip(active_rows.tolist(), active_cols.tolist()):
        y0 = tile_row * tile_height
        x0 = tile_col * tile_width
        y1 = min(height, y0 + tile_height)
        x1 = min(width, x0 + tile_width)
        pixel_mask[y0:y1, x0:x1] = 1.0
    return torch.from_numpy(pixel_mask).to(device=device).unsqueeze(0).unsqueeze(0)


def evaluate_tiled_conv(
    roi_bundle_path: str,
    device_name: str,
    out_channels: int,
    seed: int,
    output_dir: str | None = None,
) -> dict:
    data = np.load(roi_bundle_path)
    masked_rgb = data["masked_rgb"]
    tile_mask = data["tile_mask"]
    tile_width = int(data["tile_width"]) if "tile_width" in data else int(data["tile_size"])
    tile_height = int(data["tile_height"]) if "tile_height" in data else int(data["tile_size"])
    min_active_pixels = int(data["min_active_pixels"]) if "min_active_pixels" in data else 1
    return evaluate_tiled_conv_arrays(
        masked_rgb=masked_rgb,
        tile_mask=tile_mask,
        tile_width=tile_width,
        tile_height=tile_height,
        min_active_pixels=min_active_pixels,
        device_name=device_name,
        out_channels=out_channels,
        seed=seed,
        output_dir=output_dir,
    )


def evaluate_tiled_conv_arrays(
    masked_rgb: np.ndarray,
    tile_mask: np.ndarray,
    tile_width: int,
    tile_height: int,
    min_active_pixels: int,
    device_name: str,
    out_channels: int,
    seed: int,
    output_dir: str | None = None,
) -> dict:
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_cuda_sync = device.type == "cuda"

    x = rgb_to_tensor(masked_rgb, device)
    torch.manual_seed(seed)
    conv = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True).to(device)
    conv.eval()

    dense_out, dense_ms = benchmark(lambda: conv(x), use_cuda_sync)
    tiled_out, tiled_ms = benchmark(lambda: run_tiled_conv(x, tile_mask, tile_width, tile_height, conv), use_cuda_sync)

    active_ratio = float(tile_mask.sum()) / max(1, tile_mask.size)
    max_abs_diff = torch.max(torch.abs(dense_out - tiled_out)).item()
    mean_abs_diff = torch.mean(torch.abs(dense_out - tiled_out)).item()
    pixel_mask = tile_mask_to_pixel_mask(tile_mask, tile_width, tile_height, x.shape[-2], x.shape[-1], device)
    active_max_abs_diff = torch.max(torch.abs((dense_out - tiled_out) * pixel_mask)).item()
    active_mean_abs_diff = torch.sum(torch.abs((dense_out - tiled_out) * pixel_mask)).item() / max(
        1.0, float(pixel_mask.sum().item())
    )

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        preview = tensor_to_rgb_image(torch.sigmoid(tiled_out[:, :3, :, :]))
        Image.fromarray(preview, mode="RGB").save(output_path / "tiled_conv_preview.png")
        preview_path = str(output_path / "tiled_conv_preview.png")
    else:
        preview_path = ""

    return {
        "device": str(device),
        "tile_width": tile_width,
        "tile_height": tile_height,
        "min_active_pixels": min_active_pixels,
        "active_tiles": int(tile_mask.sum()),
        "total_tiles": int(tile_mask.size),
        "active_ratio": active_ratio,
        "dense_ms": float(dense_ms),
        "tiled_ms": float(tiled_ms),
        "max_abs_diff": float(max_abs_diff),
        "mean_abs_diff": float(mean_abs_diff),
        "active_max_abs_diff": float(active_max_abs_diff),
        "active_mean_abs_diff": float(active_mean_abs_diff),
        "preview_path": preview_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dense and tiled convolution on the selected ROI tiles.")
    parser.add_argument("--roi-bundle", type=str, default="generated_demo/roi_tiles.npz")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-channels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="generated_demo")
    args = parser.parse_args()

    metrics = evaluate_tiled_conv(
        roi_bundle_path=args.roi_bundle,
        device_name=args.device,
        out_channels=args.out_channels,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"Device: {metrics['device']}")
    print(f"Tile size: {metrics['tile_width']}x{metrics['tile_height']}")
    print(f"Min active pixels per tile: {metrics['min_active_pixels']}")
    print(f"Active tiles: {metrics['active_tiles']}/{metrics['total_tiles']} ({100.0 * metrics['active_ratio']:.1f}%)")
    print(f"Dense conv time: {metrics['dense_ms']:.3f} ms")
    print(f"Tiled conv time: {metrics['tiled_ms']:.3f} ms")
    print(f"Max |dense - tiled| over full frame: {metrics['max_abs_diff']:.6f}")
    print(f"Max |dense - tiled| over active tiles: {metrics['active_max_abs_diff']:.6f}")
    if metrics["preview_path"]:
        print(f"Saved preview: {metrics['preview_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
