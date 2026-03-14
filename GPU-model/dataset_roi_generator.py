import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from transparent_background import Remover


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}


def list_images(root_dir: str) -> list[Path]:
    root = Path(root_dir)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in VALID_SUFFIXES)


def resize_rgb(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), resample=Image.BILINEAR)


def alpha_bbox(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h


def alpha_to_mask(alpha: np.ndarray, threshold: int = 1) -> np.ndarray:
    return (alpha >= threshold).astype(np.uint8)


def bbox_to_mask(height: int, width: int, bbox: tuple[int, int, int, int] | None, padding: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if bbox is None:
        return mask
    x, y, w, h = bbox
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


def save_roi_bundle(
    output_dir: Path,
    image_key: str,
    rgb: np.ndarray,
    alpha: np.ndarray,
    roi_mask: np.ndarray,
    bbox_mask: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> None:
    item_dir = output_dir / image_key
    item_dir.mkdir(parents=True, exist_ok=True)

    masked_rgb = rgb.copy()
    masked_rgb[roi_mask == 0] = 0

    np.savez(
        item_dir / "roi_input.npz",
        rgb=rgb,
        alpha=alpha,
        roi_mask=roi_mask,
        bbox_mask=bbox_mask,
        bbox=np.array(bbox if bbox is not None else (-1, -1, -1, -1), dtype=np.int32),
        masked_rgb=masked_rgb,
    )

    Image.fromarray(rgb, mode="RGB").save(item_dir / "rgb.png")
    Image.fromarray(alpha, mode="L").save(item_dir / "alpha.png")
    Image.fromarray((roi_mask * 255).astype(np.uint8), mode="L").save(item_dir / "roi_mask.png")
    Image.fromarray((bbox_mask * 255).astype(np.uint8), mode="L").save(item_dir / "bbox_mask.png")
    Image.fromarray(masked_rgb, mode="RGB").save(item_dir / "masked_rgb.png")

    if bbox is not None:
        x, y, w, h = bbox
        cropped_rgb = rgb[y : y + h, x : x + w]
        if cropped_rgb.size > 0:
            Image.fromarray(cropped_rgb, mode="RGB").save(item_dir / "cropped_rgb.png")


def process_dataset(
    dataset_dir: str,
    output_dir: str,
    image_size: int,
    padding: int,
    device: str,
    max_images: int | None = None,
) -> Path:
    dataset_root = Path(dataset_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    images = list_images(dataset_dir)
    if max_images is not None:
        images = images[: max(0, max_images)]

    print("Initializing background remover...")
    remover = Remover(device=device)

    manifest_rows = []
    for index, image_path in enumerate(images, start=1):
        rel = image_path.relative_to(dataset_root)
        image_key = str(rel.with_suffix("")).replace("/", "__")

        rgb_pil = resize_rgb(Image.open(image_path), image_size)
        out = remover.process(rgb_pil, type="rgba")
        out_np = np.array(out)
        rgb = np.array(rgb_pil)
        alpha = out_np[:, :, 3]
        roi_mask = alpha_to_mask(alpha)
        bbox = alpha_bbox((roi_mask * 255).astype(np.uint8))
        bbox_mask = bbox_to_mask(alpha.shape[0], alpha.shape[1], bbox, padding=padding)

        save_roi_bundle(out_root, image_key, rgb, alpha, roi_mask, bbox_mask, bbox)
        manifest_rows.append(
            {
                "image_key": image_key,
                "source_path": str(image_path),
                "roi_pixels": int(roi_mask.sum()),
                "bbox_found": int(bbox is not None),
            }
        )
        if index % 50 == 0 or index == len(images):
            print(f"Processed {index}/{len(images)} images")

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write("image_key,source_path,roi_pixels,bbox_found\n")
        for row in manifest_rows:
            handle.write(f"{row['image_key']},{row['source_path']},{row['roi_pixels']},{row['bbox_found']}\n")

    print(f"Saved ROI dataset to: {out_root}")
    print(f"Saved manifest: {manifest_path}")
    return out_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ROI bundles for an image dataset using background removal.")
    parser.add_argument("--dataset-dir", type=str, default="imagenet_val")
    parser.add_argument("--output-dir", type=str, default="dataset_roi_frames")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--padding", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    process_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        padding=args.padding,
        device=args.device,
        max_images=args.max_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
