import os
import cv2
import numpy as np
from transparent_background import Remover
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _save_image(image, path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg") and image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        background.save(path)
        return
    image.save(path)


def _save_mask(mask, path, size=None):
    out = mask
    if size is not None:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_NEAREST)
    if path.endswith(".npy"):
        np.save(path, out.astype(np.uint8))
        return
    cv2.imwrite(path, (out * 255).astype(np.uint8))


def process_frame(
    frame_bgr,
    raw_path,
    processed_path,
    remover,
    padding=15,
    mask_png_path="",
    mask_npy_path="",
    mask_size=None,
    mask_thresh=10,
    keep_size=False,
    bg_color=0,
):
    cv2.imwrite(raw_path, frame_bgr)

    img_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    out = remover.process(img_rgb, type="rgba")
    out_np = np.array(out)
    alpha = out_np[:, :, 3]
    mask = (alpha > mask_thresh).astype(np.uint8)

    if mask_png_path:
        _save_mask(mask, mask_png_path, size=mask_size)
    if mask_npy_path:
        _save_mask(mask, mask_npy_path, size=mask_size)

    coords = cv2.findNonZero(alpha)
    if keep_size:
        rgb = out_np[:, :, :3].astype(np.float32)
        alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
        bg = np.full_like(rgb, float(bg_color))
        comp = rgb * alpha_f + bg * (1.0 - alpha_f)
        out_img = Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8), mode="RGB")
        _save_image(out_img, processed_path)
        return

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        y1, y2 = max(0, y - padding), min(out_np.shape[0], y + h + padding)
        x1, x2 = max(0, x - padding), min(out_np.shape[1], x + w + padding)
        cropped_img = out.crop((x1, y1, x2, y2))
        _save_image(cropped_img, processed_path)
    else:
        _save_image(out, processed_path)


def process_video_pipeline(
    video_path,
    frame_interval=5,
    device="cuda",
    mask_size=None,
    mask_thresh=10,
    keep_size=False,
    bg_color=0,
):
    # 1. Extract video filename and create output paths
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]

    # Define two main output folders
    raw_frames_dir = os.path.join("selected_frames", video_name)
    processed_frames_dir = os.path.join("processed_frames", video_name)

    os.makedirs(raw_frames_dir, exist_ok=True)
    os.makedirs(processed_frames_dir, exist_ok=True)

    # 2. Initialize background remover (only once to save GPU memory and time)
    print("Initializing AI model...")
    remover = Remover(device=device)

    # 3. Read the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video: {video_path}")
        return

    frame_count = 0
    saved_count = 0

    print(f"Start processing video: {video_filename} ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract one frame every frame_interval frames
        if frame_count % frame_interval == 0:
            saved_count += 1
            img_name = f"{video_name}_{saved_count}.png"
            raw_path = os.path.join(raw_frames_dir, img_name)
            processed_path = os.path.join(processed_frames_dir, img_name)
            mask_png = os.path.join(processed_frames_dir, f"{video_name}_{saved_count}_mask.png")
            mask_npy = os.path.join(processed_frames_dir, f"{video_name}_{saved_count}_mask.npy")
            process_frame(
                frame,
                raw_path,
                processed_path,
                remover,
                mask_png_path=mask_png,
                mask_npy_path=mask_npy,
                mask_size=mask_size,
                mask_thresh=mask_thresh,
                keep_size=keep_size,
                bg_color=bg_color,
            )
            print(f"Processed image #{saved_count}: {img_name}")

        frame_count += 1

    cap.release()
    print("--- Processing complete ---")
    print(f"Raw frames saved in: {raw_frames_dir}")
    print(f"Cropped results saved in: {processed_frames_dir}")


def process_image_file(
    image_path,
    device="cuda",
    mask_size=None,
    mask_thresh=10,
    keep_size=False,
    bg_color=0,
):
    image_filename = os.path.basename(image_path)
    image_name = os.path.splitext(image_filename)[0]

    raw_frames_dir = os.path.join("selected_frames", image_name)
    processed_frames_dir = os.path.join("processed_frames", image_name)

    os.makedirs(raw_frames_dir, exist_ok=True)
    os.makedirs(processed_frames_dir, exist_ok=True)

    print("Initializing AI model...")
    remover = Remover(device=device)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Unable to read image: {image_path}")
        return

    raw_path = os.path.join(raw_frames_dir, image_filename)
    processed_path = os.path.join(processed_frames_dir, image_filename)
    mask_png = os.path.join(processed_frames_dir, f"{image_name}_mask.png")
    mask_npy = os.path.join(processed_frames_dir, f"{image_name}_mask.npy")
    process_frame(
        frame,
        raw_path,
        processed_path,
        remover,
        mask_png_path=mask_png,
        mask_npy_path=mask_npy,
        mask_size=mask_size,
        mask_thresh=mask_thresh,
        keep_size=keep_size,
        bg_color=bg_color,
    )
    print("--- Processing complete ---")
    print(f"Raw image saved in: {raw_frames_dir}")
    print(f"Cropped result saved in: {processed_frames_dir}")


def process_image_folder(
    folder_path,
    device="cuda",
    mask_size=None,
    mask_thresh=10,
    keep_size=False,
    bg_color=0,
):
    folder_name = os.path.basename(folder_path.rstrip(os.sep))
    raw_frames_dir = os.path.join("selected_frames", folder_name)
    processed_frames_dir = os.path.join("processed_frames", folder_name)

    os.makedirs(raw_frames_dir, exist_ok=True)
    os.makedirs(processed_frames_dir, exist_ok=True)

    print("Initializing AI model...")
    remover = Remover(device=device)

    files = sorted(
        f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not files:
        print(f"No images found in: {folder_path}")
        return

    for idx, fname in enumerate(files, start=1):
        img_path = os.path.join(folder_path, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Skipping unreadable image: {fname}")
            continue
        raw_path = os.path.join(raw_frames_dir, fname)
        processed_path = os.path.join(processed_frames_dir, fname)
        base = os.path.splitext(fname)[0]
        mask_png = os.path.join(processed_frames_dir, f"{base}_mask.png")
        mask_npy = os.path.join(processed_frames_dir, f"{base}_mask.npy")
        process_frame(
            frame,
            raw_path,
            processed_path,
            remover,
            mask_png_path=mask_png,
            mask_npy_path=mask_npy,
            mask_size=mask_size,
            mask_thresh=mask_thresh,
            keep_size=keep_size,
            bg_color=bg_color,
        )
        print(f"Processed image #{idx}: {fname}")

    print("--- Processing complete ---")
    print(f"Raw images saved in: {raw_frames_dir}")
    print(f"Cropped results saved in: {processed_frames_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Background removal and crop for video or images.")
    parser.add_argument("input_path", nargs="?", default="apple_tesco.mp4")
    parser.add_argument("--frame-interval", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mask-size", type=int, default=None, help="Optional square size to save mask (e.g., 224).")
    parser.add_argument("--mask-thresh", type=int, default=10, help="Alpha threshold for mask (0-255).")
    parser.add_argument("--keep-size", action="store_true", help="Keep full frame size, remove background only.")
    parser.add_argument("--bg-color", type=int, default=0, help="Background color (0-255) when keep-size is set.")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Input not found: {args.input_path}")
        raise SystemExit(1)

    if os.path.isdir(args.input_path):
        process_image_folder(
            args.input_path,
            device=args.device,
            mask_size=args.mask_size,
            mask_thresh=args.mask_thresh,
            keep_size=args.keep_size,
            bg_color=args.bg_color,
        )
    else:
        ext = os.path.splitext(args.input_path)[1].lower()
        if ext in IMAGE_EXTS:
            process_image_file(
                args.input_path,
                device=args.device,
                mask_size=args.mask_size,
                mask_thresh=args.mask_thresh,
                keep_size=args.keep_size,
                bg_color=args.bg_color,
            )
        else:
            process_video_pipeline(
                args.input_path,
                frame_interval=args.frame_interval,
                device=args.device,
                mask_size=args.mask_size,
                mask_thresh=args.mask_thresh,
                keep_size=args.keep_size,
                bg_color=args.bg_color,
            )
