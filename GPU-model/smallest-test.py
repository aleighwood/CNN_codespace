#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from mobile_net import MobileNetV1
from sparse_mobilenet import SparseMobileNetRunner, build_layer_masks, image_to_normalized_tensor


def parse_tile_configs(raw: str) -> list[tuple[int, int, int]]:
    configs: list[tuple[int, int, int]] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        parts = token.split("x")
        if len(parts) != 3:
            raise ValueError(f"Invalid config token: {token}. Expected format like 14x14x4")
        w, h, m = (int(parts[0]), int(parts[1]), int(parts[2]))
        configs.append((w, h, m))
    if not configs:
        raise ValueError("No tile configs provided")
    return configs


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Invalid non-positive value: {value}")
        values.append(value)
    if not values:
        raise ValueError("No integer values provided")

    dedup: list[int] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        dedup.append(value)
        seen.add(value)
    return dedup


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_ms(fn, device: torch.device) -> float:
    sync_if_cuda(device)
    start = time.perf_counter()
    fn()
    sync_if_cuda(device)
    return (time.perf_counter() - start) * 1000.0


def stats(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(np.median(arr)), float(np.percentile(arr, 95)), float(arr.std())


def load_runner(weights_path: str, device_name: str, chunk_tiles: int) -> tuple[SparseMobileNetRunner, torch.device]:
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MobileNetV1()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    runner = SparseMobileNetRunner(model=model, device=device, chunk_tiles=chunk_tiles)
    return runner, device


def print_env(device: torch.device) -> None:
    print("=== Environment ===")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"selected device: {device}")
    if torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        print(f"gpu name: {torch.cuda.get_device_name(idx)}")
        print(f"torch cuda version: {torch.version.cuda}")
        print(f"cudnn enabled: {torch.backends.cudnn.enabled}")
        print(f"cudnn version: {torch.backends.cudnn.version()}")
    print()


def make_dense_tiled_masks(pixel_masks: list[np.ndarray], tile_masks: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    dense_pixel_masks = [np.ones_like(pixel_mask, dtype=np.uint8) for pixel_mask in pixel_masks]
    dense_tile_masks = [np.ones_like(tile_mask, dtype=np.uint8) for tile_mask in tile_masks]
    return dense_pixel_masks, dense_tile_masks


def estimate_sparse_launch_counts(
    runner: SparseMobileNetRunner,
    tile_masks: list[np.ndarray],
    tile_width: int,
    tile_height: int,
    input_h: int,
    input_w: int,
) -> tuple[int, int, int]:
    """
    Returns estimated launch counts for one image:
    1) logical patch-batch launches
    2) conv-kernel lower bound launches
    3) rough op-launch estimate (conv/bn/relu/mul style accounting)

    These are algorithmic estimates, not profiler-true CUDA launch counts.
    """
    current_h = int(input_h)
    current_w = int(input_w)
    patch_batches = 0
    conv_kernel_lb = 0
    rough_op_launches = 0

    for (spec, _), tile_mask in zip(runner.layers, tile_masks):
        out_h = (current_h + spec.stride - 1) // spec.stride
        out_w = (current_w + spec.stride - 1) // spec.stride

        active_rows, active_cols = np.nonzero(tile_mask)
        jobs_by_shape: dict[tuple[int, int], int] = {}
        for tile_row, tile_col in zip(active_rows.tolist(), active_cols.tolist()):
            out_row = tile_row * tile_height
            out_col = tile_col * tile_width
            valid_h = min(tile_height, out_h - out_row)
            valid_w = min(tile_width, out_w - out_col)
            key = (int(valid_h), int(valid_w))
            jobs_by_shape[key] = jobs_by_shape.get(key, 0) + 1

        layer_batches = 0
        for job_count in jobs_by_shape.values():
            layer_batches += (job_count + runner.chunk_tiles - 1) // runner.chunk_tiles

        patch_batches += layer_batches

        if spec.kind == "conv":
            conv_kernel_lb += layer_batches
            rough_op_launches += 4 * layer_batches
        else:
            conv_kernel_lb += 2 * layer_batches
            rough_op_launches += 7 * layer_batches

        current_h = out_h
        current_w = out_w

    return patch_batches, conv_kernel_lb, rough_op_launches


def profile_callable(fn, device: torch.device, iters: int, topk: int) -> tuple[int, list[tuple[str, int, float]], str]:
    if device.type != "cuda":
        return 0, [], ""

    sync_if_cuda(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(iters):
            fn()
    sync_if_cuda(device)

    cuda_events = []
    for event in prof.events():
        device_type = str(getattr(event, "device_type", ""))
        if "CUDA" in device_type:
            cuda_events.append(event)

    aggregated: dict[str, dict[str, float]] = {}
    for event in cuda_events:
        name = str(getattr(event, "name", "unknown"))
        time_us = float(getattr(event, "self_cuda_time_total", 0.0) or getattr(event, "cuda_time_total", 0.0) or 0.0)
        if name not in aggregated:
            aggregated[name] = {"count": 0.0, "time_us": 0.0}
        aggregated[name]["count"] += 1.0
        aggregated[name]["time_us"] += time_us

    top_kernel_rows = sorted(
        ((name, int(info["count"]), float(info["time_us"])) for name, info in aggregated.items()),
        key=lambda row: (row[2], row[1]),
        reverse=True,
    )[:topk]

    op_table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=topk)
    return len(cuda_events), top_kernel_rows, op_table


def print_profile_summary(label: str, total_cuda_events: int, rows: list[tuple[str, int, float]], op_table: str) -> None:
    print(f"[profile] {label}")
    print(f"  captured CUDA events (kernel launches): {total_cuda_events}")
    if not rows:
        print("  no CUDA kernel events captured")
        return

    print("  top CUDA kernel names by self CUDA time:")
    for name, count, time_us in rows:
        print(f"    count={count:4d}, self_cuda_ms={time_us / 1000.0:9.3f}, name={name}")

    print("  top operator table (PyTorch profiler):")
    print(op_table)


def main() -> int:
    parser = argparse.ArgumentParser(description="Small benchmark for sparse vs dense MobileNet paths.")
    parser.add_argument("--roi-dataset-dir", type=str, default="dataset_roi_frames")
    parser.add_argument("--weights", type=str, default="my_mobilenet_with_weights.pth")
    parser.add_argument("--device", type=str, default="cuda", help="cuda | cpu | auto")
    parser.add_argument("--chunk-tiles", type=int, default=None, help="single chunk_tiles value (overrides list)")
    parser.add_argument(
        "--chunk-tiles-list",
        type=str,
        default="32,64,128,256,512,1024,2048",
        help="comma-separated chunk sizes",
    )
    parser.add_argument("--tile-count-method", type=str, choices=["direct", "scanline"], default="direct")
    parser.add_argument("--num-images", type=int, default=80)
    parser.add_argument("--warmup-images", type=int, default=8)
    parser.add_argument("--profile-kernels", action="store_true", help="run torch profiler to print real CUDA event names/counts")
    parser.add_argument("--profile-iters", type=int, default=5, help="iterations per profiled forward")
    parser.add_argument("--profile-topk", type=int, default=20, help="top rows to print from profiler summaries")
    parser.add_argument(
        "--tile-configs",
        type=str,
        default="14x14x4,14x14x5,16x16x1,15x18x3",
        help="Comma-separated list, e.g. 14x14x4,16x16x1",
    )
    args = parser.parse_args()

    configs = parse_tile_configs(args.tile_configs)
    chunk_values = [int(args.chunk_tiles)] if args.chunk_tiles is not None else parse_int_list(args.chunk_tiles_list)

    roi_paths = sorted(Path(args.roi_dataset_dir).rglob("roi_input.npz"))
    if not roi_paths:
        raise SystemExit(f"No roi_input.npz found under: {args.roi_dataset_dir}")

    if args.num_images <= 0:
        raise SystemExit("--num-images must be > 0")

    roi_paths = roi_paths[: min(args.num_images, len(roi_paths))]

    first_runner, device = load_runner(args.weights, args.device, chunk_values[0])
    del first_runner

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print_env(device)
    print("=== Benchmark Setup ===")
    print(f"images: {len(roi_paths)}")
    print(f"warmup images: {min(args.warmup_images, len(roi_paths))}")
    print(f"chunk_tiles list: {chunk_values}")
    print(f"tile_count_method: {args.tile_count_method}")
    print(f"configs: {configs}")
    print("kernel launch estimate lines are algorithmic; profiler lines are real captured CUDA events")
    print()

    sample_bundle = np.load(roi_paths[0])

    with torch.inference_mode():
        for chunk_tiles in chunk_values:
            runner, _ = load_runner(args.weights, args.device, chunk_tiles)
            print(f"===== chunk_tiles={chunk_tiles} =====")

            for (tile_w, tile_h, minpix) in configs:
                print(f"--- Config w={tile_w}, h={tile_h}, minpix={minpix} ---")

                warmup_n = min(args.warmup_images, len(roi_paths))
                for path in roi_paths[:warmup_n]:
                    bundle = np.load(path)
                    pixel_masks, tile_masks = build_layer_masks(
                        roi_mask=bundle["roi_mask"],
                        tile_width=tile_w,
                        tile_height=tile_h,
                        min_active_pixels=minpix,
                        tile_count_method=args.tile_count_method,
                    )
                    dense_pixel_masks, dense_tile_masks = make_dense_tiled_masks(pixel_masks, tile_masks)
                    x_masked = image_to_normalized_tensor(bundle["masked_rgb"], device)
                    x_unmasked = image_to_normalized_tensor(bundle["rgb"], device)
                    runner.sparse_forward(x_masked, pixel_masks, tile_masks, tile_w, tile_h)
                    runner.sparse_forward(x_unmasked, dense_pixel_masks, dense_tile_masks, tile_w, tile_h)
                    runner.dense_semantic_forward(x_masked, pixel_masks)
                    runner.dense_unmasked_forward(x_unmasked)

                if args.profile_kernels and device.type == "cuda":
                    pixel_masks_sample, tile_masks_sample = build_layer_masks(
                        roi_mask=sample_bundle["roi_mask"],
                        tile_width=tile_w,
                        tile_height=tile_h,
                        min_active_pixels=minpix,
                        tile_count_method=args.tile_count_method,
                    )
                    dense_pixel_masks_sample, dense_tile_masks_sample = make_dense_tiled_masks(
                        pixel_masks_sample, tile_masks_sample
                    )
                    x_masked_sample = image_to_normalized_tensor(sample_bundle["masked_rgb"], device)
                    x_unmasked_sample = image_to_normalized_tensor(sample_bundle["rgb"], device)

                    total, rows, table = profile_callable(
                        lambda: runner.sparse_forward(
                            x_masked_sample, pixel_masks_sample, tile_masks_sample, tile_w, tile_h
                        ),
                        device,
                        args.profile_iters,
                        args.profile_topk,
                    )
                    print_profile_summary("sparse_forward", total, rows, table)

                    total, rows, table = profile_callable(
                        lambda: runner.sparse_forward(
                            x_unmasked_sample,
                            dense_pixel_masks_sample,
                            dense_tile_masks_sample,
                            tile_w,
                            tile_h,
                        ),
                        device,
                        args.profile_iters,
                        args.profile_topk,
                    )
                    print_profile_summary("dense_tiled_via_sparse_forward", total, rows, table)

                    total, rows, table = profile_callable(
                        lambda: runner.dense_unmasked_forward(x_unmasked_sample),
                        device,
                        args.profile_iters,
                        args.profile_topk,
                    )
                    print_profile_summary("dense_unmasked_forward", total, rows, table)

                io_ms: list[float] = []
                mask_build_ms: list[float] = []
                preprocess_ms: list[float] = []
                sparse_ms: list[float] = []
                dense_tiled_ms: list[float] = []
                dense_masked_ms: list[float] = []
                dense_unmasked_ms: list[float] = []
                active_ratio: list[float] = []
                launch_patch_batches_sparse: list[float] = []
                launch_conv_lb_sparse: list[float] = []
                launch_rough_ops_sparse: list[float] = []
                launch_patch_batches_dense_tiled: list[float] = []
                launch_conv_lb_dense_tiled: list[float] = []
                launch_rough_ops_dense_tiled: list[float] = []

                for path in roi_paths:
                    t0 = time.perf_counter()
                    bundle = np.load(path)
                    t1 = time.perf_counter()
                    io_ms.append((t1 - t0) * 1000.0)

                    mb_ms = timed_ms(
                        lambda: build_layer_masks(
                            roi_mask=bundle["roi_mask"],
                            tile_width=tile_w,
                            tile_height=tile_h,
                            min_active_pixels=minpix,
                            tile_count_method=args.tile_count_method,
                        ),
                        device,
                    )
                    mask_build_ms.append(mb_ms)

                    pixel_masks, tile_masks = build_layer_masks(
                        roi_mask=bundle["roi_mask"],
                        tile_width=tile_w,
                        tile_height=tile_h,
                        min_active_pixels=minpix,
                        tile_count_method=args.tile_count_method,
                    )
                    dense_pixel_masks, dense_tile_masks = make_dense_tiled_masks(pixel_masks, tile_masks)

                    pp_ms = timed_ms(
                        lambda: (
                            image_to_normalized_tensor(bundle["masked_rgb"], device),
                            image_to_normalized_tensor(bundle["rgb"], device),
                        ),
                        device,
                    )
                    preprocess_ms.append(pp_ms)

                    x_masked = image_to_normalized_tensor(bundle["masked_rgb"], device)
                    x_unmasked = image_to_normalized_tensor(bundle["rgb"], device)

                    pb, clb, rop = estimate_sparse_launch_counts(
                        runner=runner,
                        tile_masks=tile_masks,
                        tile_width=tile_w,
                        tile_height=tile_h,
                        input_h=int(x_masked.shape[-2]),
                        input_w=int(x_masked.shape[-1]),
                    )
                    launch_patch_batches_sparse.append(float(pb))
                    launch_conv_lb_sparse.append(float(clb))
                    launch_rough_ops_sparse.append(float(rop))

                    pb, clb, rop = estimate_sparse_launch_counts(
                        runner=runner,
                        tile_masks=dense_tile_masks,
                        tile_width=tile_w,
                        tile_height=tile_h,
                        input_h=int(x_unmasked.shape[-2]),
                        input_w=int(x_unmasked.shape[-1]),
                    )
                    launch_patch_batches_dense_tiled.append(float(pb))
                    launch_conv_lb_dense_tiled.append(float(clb))
                    launch_rough_ops_dense_tiled.append(float(rop))

                    sparse_holder: dict[str, float] = {}

                    def run_sparse() -> None:
                        _, active_tiles, total_tiles = runner.sparse_forward(
                            x_masked, pixel_masks, tile_masks, tile_w, tile_h
                        )
                        sparse_holder["ratio"] = float(active_tiles / max(1, total_tiles))

                    sparse_ms.append(timed_ms(run_sparse, device))
                    active_ratio.append(sparse_holder["ratio"])

                    dense_tiled_ms.append(
                        timed_ms(
                            lambda: runner.sparse_forward(
                                x_unmasked,
                                dense_pixel_masks,
                                dense_tile_masks,
                                tile_w,
                                tile_h,
                            ),
                            device,
                        )
                    )

                    dense_masked_ms.append(
                        timed_ms(lambda: runner.dense_semantic_forward(x_masked, pixel_masks), device)
                    )
                    dense_unmasked_ms.append(timed_ms(lambda: runner.dense_unmasked_forward(x_unmasked), device))

                io_mean, io_p50, io_p95, _ = stats(io_ms)
                mb_mean, mb_p50, mb_p95, _ = stats(mask_build_ms)
                pp_mean, pp_p50, pp_p95, _ = stats(preprocess_ms)
                s_mean, s_p50, s_p95, _ = stats(sparse_ms)
                dt_mean, dt_p50, dt_p95, _ = stats(dense_tiled_ms)
                dm_mean, dm_p50, dm_p95, _ = stats(dense_masked_ms)
                du_mean, du_p50, du_p95, _ = stats(dense_unmasked_ms)
                ar_mean, ar_p50, ar_p95, _ = stats(active_ratio)

                pbs_mean, pbs_p50, pbs_p95, _ = stats(launch_patch_batches_sparse)
                cls_mean, cls_p50, cls_p95, _ = stats(launch_conv_lb_sparse)
                ros_mean, ros_p50, ros_p95, _ = stats(launch_rough_ops_sparse)
                pbd_mean, pbd_p50, pbd_p95, _ = stats(launch_patch_batches_dense_tiled)
                cld_mean, cld_p50, cld_p95, _ = stats(launch_conv_lb_dense_tiled)
                rod_mean, rod_p50, rod_p95, _ = stats(launch_rough_ops_dense_tiled)

                end2end_sparse_mean = io_mean + mb_mean + pp_mean + s_mean
                end2end_dense_tiled_mean = io_mean + mb_mean + pp_mean + dt_mean
                end2end_dense_masked_mean = io_mean + mb_mean + pp_mean + dm_mean
                end2end_dense_unmasked_mean = io_mean + pp_mean + du_mean

                print(f"active ratio: mean={ar_mean:.4f}, p50={ar_p50:.4f}, p95={ar_p95:.4f}")
                print(f"io load ms: mean={io_mean:.3f}, p50={io_p50:.3f}, p95={io_p95:.3f}")
                print(f"mask build ms: mean={mb_mean:.3f}, p50={mb_p50:.3f}, p95={mb_p95:.3f}")
                print(f"preprocess ms: mean={pp_mean:.3f}, p50={pp_p50:.3f}, p95={pp_p95:.3f}")
                print(f"sparse fwd ms: mean={s_mean:.3f}, p50={s_p50:.3f}, p95={s_p95:.3f}")
                print(f"dense_tiled(via sparse path) fwd ms: mean={dt_mean:.3f}, p50={dt_p50:.3f}, p95={dt_p95:.3f}")
                print(f"dense_masked fwd ms: mean={dm_mean:.3f}, p50={dm_p50:.3f}, p95={dm_p95:.3f}")
                print(f"dense_unmasked fwd ms: mean={du_mean:.3f}, p50={du_p50:.3f}, p95={du_p95:.3f}")
                print(f"fwd speed ratio (sparse/dense_tiled): {s_mean / max(1e-9, dt_mean):.3f}x")
                print(f"fwd speed ratio (dense_tiled/dense_unmasked): {dt_mean / max(1e-9, du_mean):.3f}x")
                print(f"fwd speed ratio (sparse/dense_masked): {s_mean / max(1e-9, dm_mean):.3f}x")
                print(f"fwd speed ratio (sparse/dense_unmasked): {s_mean / max(1e-9, du_mean):.3f}x")

                print(
                    f"launch estimate sparse - patch batches/image: mean={pbs_mean:.1f}, p50={pbs_p50:.1f}, p95={pbs_p95:.1f}"
                )
                print(
                    f"launch estimate sparse - conv-kernel lower bound/image: mean={cls_mean:.1f}, p50={cls_p50:.1f}, p95={cls_p95:.1f}"
                )
                print(
                    f"launch estimate sparse - rough op launches/image: mean={ros_mean:.1f}, p50={ros_p50:.1f}, p95={ros_p95:.1f}"
                )

                print(
                    f"launch estimate dense_tiled - patch batches/image: mean={pbd_mean:.1f}, p50={pbd_p50:.1f}, p95={pbd_p95:.1f}"
                )
                print(
                    f"launch estimate dense_tiled - conv-kernel lower bound/image: mean={cld_mean:.1f}, p50={cld_p50:.1f}, p95={cld_p95:.1f}"
                )
                print(
                    f"launch estimate dense_tiled - rough op launches/image: mean={rod_mean:.1f}, p50={rod_p50:.1f}, p95={rod_p95:.1f}"
                )

                print(f"e2e mean sparse ms: {end2end_sparse_mean:.3f}")
                print(f"e2e mean dense_tiled ms: {end2end_dense_tiled_mean:.3f}")
                print(f"e2e mean dense_masked ms: {end2end_dense_masked_mean:.3f}")
                print(f"e2e mean dense_unmasked ms: {end2end_dense_unmasked_mean:.3f}")
                print()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
