from pathlib import Path

from dataset_roi_generator import process_dataset
from active_tile_pixel_dataset_sweep import main as dataset_sweep_main


def main() -> int:
    roi_dir = Path("dataset_roi_frames")
    manifest_path = roi_dir / "manifest.csv"

    if manifest_path.exists():
        print(f"Reusing existing ROI dataset: {roi_dir}")
    else:
        process_dataset(
            dataset_dir="imagenet_val",
            output_dir=str(roi_dir),
            image_size=224,
            padding=15,
            device="cuda",
            max_images=None,
        )
    return dataset_sweep_main()


if __name__ == "__main__":
    raise SystemExit(main())
