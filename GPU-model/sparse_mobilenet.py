from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from mobile_net import MobileNetV1
from roi_tiles import calculate_tile_counts_direct, calculate_windows_online, get_scanline_stream


@dataclass
class LayerSpec:
    name: str
    kind: str
    stride: int


LAYER_SPECS: List[LayerSpec] = [
    LayerSpec("conv1", "conv", 2),
    LayerSpec("conv_dw_1", "dws", 1),
    LayerSpec("conv_dw_2", "dws", 2),
    LayerSpec("conv_dw_3", "dws", 1),
    LayerSpec("conv_dw_4", "dws", 2),
    LayerSpec("conv_dw_5", "dws", 1),
    LayerSpec("conv_dw_6", "dws", 2),
    LayerSpec("conv_dw_7", "dws", 1),
    LayerSpec("conv_dw_8", "dws", 1),
    LayerSpec("conv_dw_9", "dws", 1),
    LayerSpec("conv_dw_10", "dws", 1),
    LayerSpec("conv_dw_11", "dws", 1),
    LayerSpec("conv_dw_12", "dws", 2),
    LayerSpec("conv_dw_13", "dws", 1),
]


def image_to_normalized_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = x.to(device)
    return (x - mean) / std


def label_from_roi_input_path(path: str) -> int:
    return int(path.split("/")[-2].split("__")[0])


def tile_counts_from_mask(mask: np.ndarray, tile_width: int, tile_height: int, method: str) -> np.ndarray:
    if method == "direct":
        tile_pixel_counts, _ = calculate_tile_counts_direct(mask=mask, tile_w=tile_width, tile_h=tile_height)
        return tile_pixel_counts
    if method == "scanline":
        stream = get_scanline_stream(mask)
        _, _, tile_pixel_counts = calculate_windows_online(
            stream=stream,
            image_h=mask.shape[0],
            image_w=mask.shape[1],
            tile_w=tile_width,
            tile_h=tile_height,
        )
        return tile_pixel_counts
    raise ValueError(f"Unknown tile-count method: {method}")


def mask_after_spatial_conv(mask: np.ndarray, stride: int) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    out = F.max_pool2d(tensor, kernel_size=3, stride=stride, padding=1)
    return (out.squeeze(0).squeeze(0).numpy() > 0).astype(np.uint8)


def build_layer_masks(
    roi_mask: np.ndarray,
    tile_width: int,
    tile_height: int,
    min_active_pixels: int,
    tile_count_method: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    pixel_masks: list[np.ndarray] = []
    tile_masks: list[np.ndarray] = []

    current_mask = roi_mask.astype(np.uint8)
    for spec in LAYER_SPECS:
        current_mask = mask_after_spatial_conv(current_mask, spec.stride)
        pixel_masks.append(current_mask.copy())
        tile_counts = tile_counts_from_mask(current_mask, tile_width=tile_width, tile_height=tile_height, method=tile_count_method)
        tile_masks.append((tile_counts >= max(1, min_active_pixels)).astype(np.uint8))
    return pixel_masks, tile_masks


def _apply_conv_bn_relu(x: torch.Tensor, layer) -> torch.Tensor:
    x = layer[0](x)
    x = layer[1](x)
    x = layer[2](x)
    return x


def _apply_dws_dense(x: torch.Tensor, layer) -> torch.Tensor:
    for module in layer:
        x = module(x)
    return x


def _apply_conv_block_patch(layer, patch: torch.Tensor) -> torch.Tensor:
    conv = layer[0]
    x = F.conv2d(patch, conv.weight, bias=conv.bias, stride=conv.stride, padding=0, groups=conv.groups)
    x = layer[1](x)
    x = layer[2](x)
    return x


def _apply_dws_block_patch(layer, patch: torch.Tensor) -> torch.Tensor:
    dw = layer[0]
    x = F.conv2d(patch, dw.weight, bias=dw.bias, stride=dw.stride, padding=0, groups=dw.groups)
    x = layer[1](x)
    x = layer[2](x)
    pw = layer[3]
    x = F.conv2d(x, pw.weight, bias=pw.bias, stride=pw.stride, padding=0, groups=pw.groups)
    x = layer[4](x)
    x = layer[5](x)
    return x


class SparseMobileNetRunner:
    def __init__(self, model: MobileNetV1, device: torch.device, chunk_tiles: int = 32):
        self.model = model.to(device).eval()
        self.device = device
        self.chunk_tiles = chunk_tiles
        self.layers = [(spec, getattr(self.model, spec.name)) for spec in LAYER_SPECS]

    def dense_semantic_forward(self, image_tensor: torch.Tensor, pixel_masks: list[np.ndarray]) -> torch.Tensor:
        x = image_tensor
        for (spec, layer), pixel_mask in zip(self.layers, pixel_masks):
            if spec.kind == "conv":
                x = _apply_conv_bn_relu(x, layer)
            else:
                x = _apply_dws_dense(x, layer)
            mask_tensor = torch.from_numpy(pixel_mask.astype(np.float32)).to(self.device).unsqueeze(0).unsqueeze(0)
            x = x * mask_tensor
        x = self.model.avg_pool(x)
        x = self.model.dropout(x)
        x = self.model.conv_preds(x)
        x = x.view(x.size(0), -1)
        x = torch.softmax(x, dim=1)
        return x

    def dense_unmasked_forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(image_tensor)

    def sparse_forward(
        self,
        image_tensor: torch.Tensor,
        pixel_masks: list[np.ndarray],
        tile_masks: list[np.ndarray],
        tile_width: int,
        tile_height: int,
    ) -> tuple[torch.Tensor, int, int]:
        x = image_tensor
        total_active_tiles = 0
        total_tiles = 0
        for (spec, layer), pixel_mask, tile_mask in zip(self.layers, pixel_masks, tile_masks):
            x = self._run_sparse_layer(
                x=x,
                layer=layer,
                spec=spec,
                pixel_mask=pixel_mask,
                tile_mask=tile_mask,
                tile_width=tile_width,
                tile_height=tile_height,
            )
            total_active_tiles += int(tile_mask.sum())
            total_tiles += int(tile_mask.size)
        x = self.model.avg_pool(x)
        x = self.model.dropout(x)
        x = self.model.conv_preds(x)
        x = x.view(x.size(0), -1)
        x = torch.softmax(x, dim=1)
        return x, total_active_tiles, total_tiles

    def _extract_patch(self, x: torch.Tensor, out_row: int, out_col: int, valid_h: int, valid_w: int, stride: int) -> torch.Tensor:
        x_padded = F.pad(x, (1, 1, 1, 1))
        patch_h = (valid_h - 1) * stride + 3
        patch_w = (valid_w - 1) * stride + 3
        start_row = out_row * stride
        start_col = out_col * stride
        return x_padded[:, :, start_row : start_row + patch_h, start_col : start_col + patch_w]

    def _run_sparse_layer(
        self,
        x: torch.Tensor,
        layer,
        spec: LayerSpec,
        pixel_mask: np.ndarray,
        tile_mask: np.ndarray,
        tile_width: int,
        tile_height: int,
    ) -> torch.Tensor:
        _, _, out_h, out_w = self._dense_output_shape(x, spec.stride)
        out_channels = layer[0].out_channels if spec.kind == "conv" else layer[3].out_channels
        out = torch.zeros((1, out_channels, out_h, out_w), device=self.device, dtype=x.dtype)

        active_rows, active_cols = np.nonzero(tile_mask)
        jobs = []
        for tile_row, tile_col in zip(active_rows.tolist(), active_cols.tolist()):
            out_row = tile_row * tile_height
            out_col = tile_col * tile_width
            valid_h = min(tile_height, out_h - out_row)
            valid_w = min(tile_width, out_w - out_col)
            jobs.append((out_row, out_col, valid_h, valid_w))

        jobs_by_shape: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        for job in jobs:
            jobs_by_shape.setdefault((job[2], job[3]), []).append(job)

        for (_, _), shaped_jobs in jobs_by_shape.items():
            for start in range(0, len(shaped_jobs), self.chunk_tiles):
                batch_jobs = shaped_jobs[start : start + self.chunk_tiles]
                patches = [self._extract_patch(x, row, col, valid_h, valid_w, spec.stride) for row, col, valid_h, valid_w in batch_jobs]
                patch_batch = torch.cat(patches, dim=0)
                if spec.kind == "conv":
                    patch_out = _apply_conv_block_patch(layer, patch_batch)
                else:
                    patch_out = _apply_dws_block_patch(layer, patch_batch)

                for idx, (out_row, out_col, valid_h, valid_w) in enumerate(batch_jobs):
                    out[:, :, out_row : out_row + valid_h, out_col : out_col + valid_w] = patch_out[
                        idx : idx + 1, :, :valid_h, :valid_w
                    ]

        mask_tensor = torch.from_numpy(pixel_mask.astype(np.float32)).to(self.device).unsqueeze(0).unsqueeze(0)
        return out * mask_tensor

    @staticmethod
    def _dense_output_shape(x: torch.Tensor, stride: int) -> tuple[int, int, int, int]:
        _, _, h, w = x.shape
        out_h = (h + stride - 1) // stride
        out_w = (w + stride - 1) // stride
        return (1, 1, out_h, out_w)
