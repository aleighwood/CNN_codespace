#!/usr/bin/env python3

import argparse
import glob
import json
import platform
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple, NamedTuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobile_net import MobileNetV1
from sparse_mobilenet import SparseMobileNetRunner, build_layer_masks, image_to_normalized_tensor


class PreparedSample(NamedTuple):
    key: str
    x_masked: torch.Tensor
    pixel_masks: List[np.ndarray]
    tile_masks: List[np.ndarray]
    dense_pixel_masks: List[np.ndarray]
    dense_tile_masks: List[np.ndarray]
    active_ratio: float


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def inference_context():
    context_factory = getattr(torch, "inference_mode", torch.no_grad)
    return context_factory()


def detect_jetson_platform() -> bool:
    if Path("/etc/nv_tegra_release").exists():
        return True
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        try:
            text = model_path.read_text(encoding="utf-8", errors="ignore").lower()
            return ("jetson" in text) or ("tegra" in text)
        except OSError:
            return False
    return False


class PowerBackend:
    name = "unavailable"
    details = "power telemetry not available"

    def read_power_w(self) -> Optional[float]:
        return None

    def close(self) -> None:
        return


class NullPowerBackend(PowerBackend):
    def __init__(self, details: str):
        self.name = "unavailable"
        self.details = details


class NVMLPowerBackend(PowerBackend):
    def __init__(self, device_index: int):
        import pynvml

        self._pynvml = pynvml
        self._pynvml.nvmlInit()
        self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.name = "nvml"
        self.details = f"NVML device index {device_index}"

    def read_power_w(self) -> Optional[float]:
        try:
            mw = float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle))
            return mw / 1000.0
        except Exception:
            return None

    def close(self) -> None:
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass


class JetsonSysfsPowerBackend(PowerBackend):
    _PREFERRED_LABEL_TOKENS = ("POM_5V_IN", "VDD_IN", "VIN_SYS_5V0", "TOTAL")
    _POWER_PATTERNS = (
        # Keep these patterns bounded. Recursive globbing through /sys/devices can
        # follow Jetson subsystem symlink loops and hang before inference starts.
        "/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power*_input",
        "/sys/bus/i2c/drivers/ina3221/*/iio:device*/in_power*_input",
        "/sys/bus/i2c/devices/*/iio:device*/in_power*_input",
        "/sys/class/hwmon/hwmon*/power*_input",
        "/sys/class/hwmon/hwmon*/in_power*_input",
    )

    def __init__(self, selected_paths: List[Path], details: str):
        self.selected_paths = selected_paths
        self.name = "jetson_sysfs"
        self.details = details

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            return ""

    @classmethod
    def _label_path_for(cls, power_path: Path) -> Path:
        filename = power_path.name
        if filename.endswith("_input"):
            return power_path.with_name(filename.replace("_input", "_label"))
        return power_path

    @classmethod
    def _read_label(cls, power_path: Path) -> str:
        label_path = cls._label_path_for(power_path)
        if label_path.exists() and label_path != power_path:
            return cls._read_text(label_path)
        return ""

    @staticmethod
    def _raw_to_watts(raw: float) -> float:
        # Heuristic for common sysfs units across Jetson boards.
        if raw > 100000.0:
            return raw / 1_000_000.0
        if raw > 1000.0:
            return raw / 1000.0
        return raw

    @classmethod
    def try_create(cls) -> Optional["JetsonSysfsPowerBackend"]:
        files: List[Path] = []
        for pattern in cls._POWER_PATTERNS:
            try:
                files.extend(Path(p) for p in glob.glob(pattern))
            except OSError:
                continue
        files = sorted({p for p in files if p.is_file()})
        if not files:
            return None

        scored: List[Tuple[int, Path, str]] = []
        for path in files:
            label = cls._read_label(path).upper()
            score = 0
            for idx, token in enumerate(cls._PREFERRED_LABEL_TOKENS):
                if token in label:
                    score = 100 - idx
                    break
            scored.append((score, path, label))

        scored.sort(key=lambda item: item[0], reverse=True)
        if scored[0][0] > 0:
            top = scored[0]
            details = f"preferred rail label={top[2] or 'unknown'} file={top[1]}"
            return cls([top[1]], details)

        details = f"sum of {len(files)} rail files (no total-rail label found)"
        return cls(files, details)

    def read_power_w(self) -> Optional[float]:
        total_w = 0.0
        valid = 0
        for path in self.selected_paths:
            try:
                raw_text = path.read_text(encoding="utf-8", errors="ignore").strip()
                raw = float(raw_text)
            except Exception:
                continue
            total_w += self._raw_to_watts(raw)
            valid += 1
        if valid == 0:
            return None
        return total_w


class PowerSampler:
    def __init__(self, backend: PowerBackend, interval_s: float):
        self.backend = backend
        self.interval_s = interval_s
        self.samples_w: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self.backend.read_power_w()
            if value is not None and value > 0.0:
                self.samples_w.append(float(value))
            time.sleep(self.interval_s)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, 2.0 * self.interval_s))
        self.backend.close()

    @property
    def average_power_w(self) -> Optional[float]:
        if not self.samples_w:
            return None
        return float(np.mean(np.array(self.samples_w, dtype=np.float64)))


def create_power_backend(device: torch.device, is_jetson: bool) -> PowerBackend:
    if is_jetson:
        jetson_backend = JetsonSysfsPowerBackend.try_create()
        if jetson_backend is not None:
            return jetson_backend
        return NullPowerBackend("Jetson platform detected but no readable sysfs power rails found")

    if device.type == "cuda":
        try:
            idx = device.index if device.index is not None else 0
            return NVMLPowerBackend(device_index=idx)
        except Exception as exc:
            return NullPowerBackend(f"NVML unavailable on this CUDA host: {exc}")

    return NullPowerBackend("CPU run (no GPU power telemetry)")


def dense_masks_from(pixel_masks: List[np.ndarray], tile_masks: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    dense_pixel_masks = [np.ones_like(mask, dtype=np.uint8) for mask in pixel_masks]
    dense_tile_masks = [np.ones_like(mask, dtype=np.uint8) for mask in tile_masks]
    return dense_pixel_masks, dense_tile_masks


def list_roi_inputs(roi_dir: Path) -> List[Path]:
    return sorted(roi_dir.rglob("roi_input.npz"))


def load_weights(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_runner(weights_path: Path, device: torch.device, chunk_tiles: int) -> SparseMobileNetRunner:
    model = MobileNetV1()
    state_dict = load_weights(weights_path, device)
    model.load_state_dict(state_dict)
    return SparseMobileNetRunner(model=model, device=device, chunk_tiles=chunk_tiles)


def prepare_samples(
    roi_paths: List[Path],
    device: torch.device,
    tile_width: int,
    tile_height: int,
    min_active_pixels: int,
    tile_count_method: str,
) -> List[PreparedSample]:
    prepared: List[PreparedSample] = []
    for roi_path in roi_paths:
        bundle = np.load(roi_path)
        pixel_masks, tile_masks = build_layer_masks(
            roi_mask=bundle["roi_mask"],
            tile_width=tile_width,
            tile_height=tile_height,
            min_active_pixels=min_active_pixels,
            tile_count_method=tile_count_method,
        )
        dense_pixel_masks, dense_tile_masks = dense_masks_from(pixel_masks, tile_masks)
        x_masked = image_to_normalized_tensor(bundle["masked_rgb"], device)
        active_tiles = int(sum(mask.sum() for mask in tile_masks))
        total_tiles = int(sum(mask.size for mask in tile_masks))
        prepared.append(
            PreparedSample(
                key=roi_path.parent.name,
                x_masked=x_masked,
                pixel_masks=pixel_masks,
                tile_masks=tile_masks,
                dense_pixel_masks=dense_pixel_masks,
                dense_tile_masks=dense_tile_masks,
                active_ratio=(active_tiles / max(1, total_tiles)),
            )
        )
    return prepared


def run_mode(
    mode_name: str,
    prepared: List[PreparedSample],
    runner: SparseMobileNetRunner,
    tile_width: int,
    tile_height: int,
    iters: int,
    power_interval_ms: int,
    is_jetson: bool,
) -> dict:
    backend = create_power_backend(runner.device, is_jetson=is_jetson)
    sampler = PowerSampler(backend=backend, interval_s=max(0.01, power_interval_ms / 1000.0))

    def _forward(sample: PreparedSample) -> None:
        if mode_name == "sparse":
            runner.sparse_forward(
                image_tensor=sample.x_masked,
                pixel_masks=sample.pixel_masks,
                tile_masks=sample.tile_masks,
                tile_width=tile_width,
                tile_height=tile_height,
            )
            return
        if mode_name == "dense_tiled":
            runner.sparse_forward(
                image_tensor=sample.x_masked,
                pixel_masks=sample.dense_pixel_masks,
                tile_masks=sample.dense_tile_masks,
                tile_width=tile_width,
                tile_height=tile_height,
            )
            return
        raise ValueError(f"Unsupported mode: {mode_name}")

    sync_if_cuda(runner.device)
    sampler.start()
    start = time.perf_counter()
    with inference_context():
        for _ in range(iters):
            for sample in prepared:
                _forward(sample)
    sync_if_cuda(runner.device)
    elapsed_s = time.perf_counter() - start
    sampler.stop()

    num_inferences = int(iters * len(prepared))
    avg_latency_ms = (elapsed_s * 1000.0) / max(1, num_inferences)
    throughput_ips = num_inferences / max(1e-12, elapsed_s)

    avg_power_w = sampler.average_power_w
    if avg_power_w is None:
        energy_j = None
        energy_per_inf_mj = None
    else:
        energy_j = float(avg_power_w * elapsed_s)
        energy_per_inf_mj = float((energy_j * 1000.0) / max(1, num_inferences))

    return {
        "mode": mode_name,
        "num_inferences": num_inferences,
        "total_time_s": float(elapsed_s),
        "avg_latency_ms": float(avg_latency_ms),
        "throughput_inf_per_s": float(throughput_ips),
        "power_backend": backend.name,
        "power_backend_details": backend.details,
        "power_sample_count": len(sampler.samples_w),
        "avg_power_w": avg_power_w,
        "energy_j": energy_j,
        "energy_per_inf_mj": energy_per_inf_mj,
    }


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested CUDA but torch.cuda.is_available() is False")
    return device


def run_warmup(
    prepared: List[PreparedSample],
    runner: SparseMobileNetRunner,
    tile_width: int,
    tile_height: int,
    warmup_iters: int,
) -> None:
    if warmup_iters <= 0:
        return
    with inference_context():
        for _ in range(warmup_iters):
            for sample in prepared:
                runner.sparse_forward(
                    image_tensor=sample.x_masked,
                    pixel_masks=sample.pixel_masks,
                    tile_masks=sample.tile_masks,
                    tile_width=tile_width,
                    tile_height=tile_height,
                )
                runner.sparse_forward(
                    image_tensor=sample.x_masked,
                    pixel_masks=sample.dense_pixel_masks,
                    tile_masks=sample.dense_tile_masks,
                    tile_width=tile_width,
                    tile_height=tile_height,
                )
    sync_if_cuda(runner.device)


def main() -> int:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Jetson-oriented sparse vs dense-tiled inference runner (no sweep).")
    parser.add_argument("--roi-dir", type=Path, default=script_dir / "samples", help="Directory containing copied roi_input.npz files")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "my_mobilenet_with_weights.pth")
    parser.add_argument("--device", type=str, default="auto", help="cuda | cpu | auto")
    parser.add_argument("--tile-width", type=int, default=14)
    parser.add_argument("--tile-height", type=int, default=14)
    parser.add_argument("--min-active-pixels", type=int, default=4)
    parser.add_argument("--chunk-tiles", type=int, default=64)
    parser.add_argument("--tile-count-method", choices=["direct", "scanline"], default="direct")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--power-sample-interval-ms", type=int, default=50)
    parser.add_argument("--report-json", type=Path, default=script_dir / "inference_report.json")
    args = parser.parse_args()

    if args.tile_width <= 0 or args.tile_height <= 0:
        raise SystemExit("tile width/height must be > 0")
    if args.min_active_pixels <= 0:
        raise SystemExit("min-active-pixels must be > 0")
    if args.chunk_tiles <= 0:
        raise SystemExit("chunk-tiles must be > 0")
    if args.iters <= 0:
        raise SystemExit("iters must be > 0")

    device = select_device(args.device)
    is_jetson = detect_jetson_platform()

    roi_paths = list_roi_inputs(args.roi_dir)
    if args.max_images is not None:
        roi_paths = roi_paths[: max(0, args.max_images)]
    if not roi_paths:
        raise SystemExit(f"No roi_input.npz found under: {args.roi_dir}")

    runner = load_runner(weights_path=args.weights, device=device, chunk_tiles=args.chunk_tiles)

    prepared = prepare_samples(
        roi_paths=roi_paths,
        device=device,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        min_active_pixels=args.min_active_pixels,
        tile_count_method=args.tile_count_method,
    )

    mean_active_ratio = float(np.mean(np.array([sample.active_ratio for sample in prepared], dtype=np.float64)))

    run_warmup(
        prepared=prepared,
        runner=runner,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        warmup_iters=args.warmup_iters,
    )

    sparse_result = run_mode(
        mode_name="sparse",
        prepared=prepared,
        runner=runner,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        iters=args.iters,
        power_interval_ms=args.power_sample_interval_ms,
        is_jetson=is_jetson,
    )
    dense_tiled_result = run_mode(
        mode_name="dense_tiled",
        prepared=prepared,
        runner=runner,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        iters=args.iters,
        power_interval_ms=args.power_sample_interval_ms,
        is_jetson=is_jetson,
    )

    report = {
        "platform": {
            "is_jetson": is_jetson,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": (torch.cuda.get_device_name(device.index if device.index is not None else 0) if device.type == "cuda" else None),
        },
        "config": {
            "roi_dir": str(args.roi_dir),
            "num_images": len(prepared),
            "tile_width": args.tile_width,
            "tile_height": args.tile_height,
            "min_active_pixels": args.min_active_pixels,
            "chunk_tiles": args.chunk_tiles,
            "tile_count_method": args.tile_count_method,
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
            "mean_active_ratio": mean_active_ratio,
        },
        "results": {
            "sparse": sparse_result,
            "dense_tiled": dense_tiled_result,
        },
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    print("=== Jetson Inference Summary ===")
    print(f"ROI dir: {args.roi_dir}")
    print(f"Images: {len(prepared)}")
    print(
        f"Tile config: {args.tile_width}x{args.tile_height}, min_active_pixels={args.min_active_pixels}, chunk_tiles={args.chunk_tiles}"
    )
    print(f"Device: {device}")
    print(f"Jetson detected: {is_jetson}")
    print(f"Mean active ratio: {100.0 * mean_active_ratio:.2f}%")

    for mode_key in ("sparse", "dense_tiled"):
        row = report["results"][mode_key]
        print(f"[{mode_key}] avg_latency_ms={row['avg_latency_ms']:.3f} throughput={row['throughput_inf_per_s']:.3f} inf/s")
        if row["avg_power_w"] is None:
            print(f"[{mode_key}] power/energy: unavailable ({row['power_backend_details']})")
        else:
            print(
                f"[{mode_key}] avg_power_w={row['avg_power_w']:.3f} energy_j={row['energy_j']:.3f} energy_per_inf_mj={row['energy_per_inf_mj']:.3f}"
            )

    print(f"Saved report: {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
