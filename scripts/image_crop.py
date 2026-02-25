import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


DEFAULT_IMAGE_WIDTH = 224
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_WINDOW_WIDTH = 16
DEFAULT_WINDOW_HEIGHT = 16


def generate_blob_pixels(image_w, image_h):
    """
    Generate a blob of pixels with some stems/branches growing from it.
    Returns a binary mask where 1 = object pixel, 0 = background
    """
    # Start with empty image
    pixels = np.zeros((image_h, image_w), dtype=np.uint8)
    
    # Create main blob in the center
    center_x, center_y = image_w // 2, image_h // 2
    sx = image_w / 80.0
    sy = image_h / 60.0
    s = min(sx, sy)
    
    # Main blob (roughly circular)
    radius = max(2, int(round(12 * s)))
    for y in range(image_h):
        for x in range(image_w):
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if dist < radius:
                pixels[y, x] = 1
    
    # Add some stems/branches growing from the main blob
    # Branch 1: extending right
    start_x = center_x + int(round(10 * sx))
    end_x = center_x + int(round(25 * sx))
    thickness = max(1, int(round(2 * sy)))
    for x in range(start_x, end_x):
        y = center_y - int(round(5 * sy)) + int(np.sin((x - center_x) * 0.3) * (3 * sy))
        if 0 <= y < image_h:
            pixels[y:y+thickness, x] = 1
    
    # Branch 2: extending up-left
    for i in range(max(1, int(round(15 * s)))):
        x = center_x - int(round(8 * sx)) - i
        y = center_y - int(round(10 * sy)) - i
        if 0 <= x < image_w and 0 <= y < image_h:
            pixels[y:y+thickness, x:x+thickness] = 1
    
    # Branch 3: extending down
    start_y = center_y + int(round(10 * sy))
    end_y = center_y + int(round(20 * sy))
    for y in range(start_y, end_y):
        x = center_x + int(round(5 * sx)) + int(np.cos((y - center_y) * 0.4) * (2 * sx))
        if 0 <= x < image_w:
            pixels[y, x:x+thickness] = 1
    
    # Small separate blob (simulating noise or another object)
    n_y0 = int(round(10 * sy))
    n_y1 = int(round(15 * sy))
    n_x0 = int(round(65 * sx))
    n_x1 = int(round(70 * sx))
    pixels[n_y0:n_y1, n_x0:n_x1] = 1
    
    return pixels


def otsu_threshold(gray):
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0.0, 1.0))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 0.5
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = i / 255.0
    return threshold


def binary_dilate(mask, k=3):
    if k <= 1:
        return mask.copy()
    pad = k // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=0)
    out = np.zeros_like(mask)
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if np.any(padded[y : y + k, x : x + k]):
                out[y, x] = 1
    return out


def binary_erode(mask, k=3):
    if k <= 1:
        return mask.copy()
    pad = k // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=0)
    out = np.zeros_like(mask)
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if np.all(padded[y : y + k, x : x + k]):
                out[y, x] = 1
    return out


def binary_open(mask, k=3, iters=1):
    out = mask.copy()
    for _ in range(iters):
        out = binary_erode(out, k)
        out = binary_dilate(out, k)
    return out


def binary_close(mask, k=3, iters=1):
    out = mask.copy()
    for _ in range(iters):
        out = binary_dilate(out, k)
        out = binary_erode(out, k)
    return out


def keep_largest_component(mask, min_area=0):
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 0
    areas = {}
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 1 and labels[y, x] == 0:
                label += 1
                stack = [(y, x)]
                labels[y, x] = label
                area = 0
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] == 1 and labels[ny, nx] == 0:
                                labels[ny, nx] = label
                                stack.append((ny, nx))
                areas[label] = area

    if not areas:
        return mask

    if min_area > 0:
        keep = {lab for lab, area in areas.items() if area >= min_area}
    else:
        max_label = max(areas, key=areas.get)
        keep = {max_label}

    out = np.zeros_like(mask)
    for y in range(h):
        for x in range(w):
            if labels[y, x] in keep:
                out[y, x] = 1
    return out


def load_image_mask(image_path, image_w, image_h, threshold, auto_threshold, invert):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_w, image_h), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    if auto_threshold:
        threshold = otsu_threshold(gray)
    mask = (gray >= threshold).astype(np.uint8)
    if invert:
        mask = 1 - mask
    return mask, threshold, arr

def get_scanline_stream(pixels):
    """
    Convert the 2D pixel array into a full scanline stream (x-major, row by row).
    This is what the FPGA actually receives: ALL pixels in order.
    Returns list of tuples: (x, y, pixel_value) where pixel_value is 0 or 1
    """
    stream = []
    for y in range(pixels.shape[0]):
        for x in range(pixels.shape[1]):
            stream.append((x, y, pixels[y, x]))
    return stream

def calculate_windows_online(stream, window_w, window_h, verbose=False):
    """
    Calculate tiled windows ON-THE-FLY as the pixel stream comes in.
    This simulates what the FPGA would do in real-time.
    
    Algorithm:
    1. As each pixel arrives, check if it's an object pixel
    2. If yes, determine which window it belongs to
    3. Mark that window as needed
    
    Returns: list of windows and the order they were discovered
    """
    windows_dict = {}  # Key: (window_x, window_y), Value: discovery_order
    discovery_order = 0
    
    print("\n" + "="*60)
    print("SIMULATING ON-THE-FLY WINDOW DETECTION")
    print("="*60)
    print("Processing pixel stream in scanline order (x-major)...")
    
    object_count = 0
    for idx, (x, y, pixel_val) in enumerate(stream):
        if pixel_val == 1:  # Object pixel detected
            object_count += 1
            
            # Calculate which window this pixel belongs to
            window_x = (x // window_w) * window_w
            window_y = (y // window_h) * window_h
            window_key = (window_x, window_y)
            
            # If this window hasn't been discovered yet, add it
            if window_key not in windows_dict:
                windows_dict[window_key] = discovery_order
                if verbose and discovery_order < 10:  # Only print first 10 to avoid clutter
                    print(f"  Pixel #{idx}: ({x},{y}) → NEW Window at ({window_x},{window_y}) [Window #{discovery_order}]")
                discovery_order += 1
    
    print(f"\nStream Statistics:")
    print(f"  Total pixels processed: {len(stream)}")
    print(f"  Object pixels found: {object_count}")
    print(f"  Empty pixels: {len(stream) - object_count}")
    print(f"  Windows needed: {len(windows_dict)}")
    print("="*60)
    
    # Convert to list format with discovery order
    windows_with_order = []
    for (wx, wy), order in sorted(windows_dict.items(), key=lambda x: x[1]):
        windows_with_order.append((wx, wy, window_w, window_h, order))
    
    return windows_with_order

def visualize_tiled_convolution(pixels, windows, stream, window_w, window_h, output_image, rgb_image=None):
    """
    Create visualization showing:
    1. The full scanline stream concept
    2. The tiled windows overlaid with discovery order
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Show scanline concept
    ax1 = axes[0]
    ax1.imshow(pixels, cmap='gray', interpolation='nearest')
    ax1.set_title('Pixel Stream (Scanline Order: X-major)\nWhite=Object(1), Black=Empty(0)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X → (scanning direction)', fontsize=12)
    ax1.set_ylabel('Y (row number)', fontsize=12)
    
    # Draw scanline direction arrows
    image_h, image_w = pixels.shape
    arrow_y_positions = [5, 20, 35, 50]
    for y_pos in arrow_y_positions:
        if y_pos < image_h:
            ax1.annotate('', xy=(image_w-5, y_pos), xytext=(5, y_pos),
                        arrowprops=dict(arrowstyle='->', color='red', 
                                      lw=2, alpha=0.7))
            ax1.text(2, y_pos, f'Row {y_pos}', color='red', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    
    # Plot 2: Pixels with tiled windows showing discovery order
    ax2 = axes[1]
    if rgb_image is not None:
        ax2.imshow(rgb_image, interpolation='nearest')
    else:
        ax2.imshow(pixels, cmap='gray', interpolation='nearest', alpha=0.7)
    ax2.set_title(f'Windows Drawn On-The-Fly (Size: {window_w}×{window_h})\nNumbers = Discovery Order',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    
    # Draw windows with discovery order
    for x, y, w, h, order in windows:
        rect = patches.Rectangle((x - 0.5, y - 0.5), w, h,
                                linewidth=2, edgecolor='red',
                                facecolor='none', linestyle='-')
        ax2.add_patch(rect)
        
        # Add discovery order number at center
        center_x = x + w / 2
        center_y = y + h / 2
        ax2.text(center_x, center_y, str(order), 
                color='white', fontsize=11, weight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7,
                         edgecolor='cyan', linewidth=2))
    
    plt.tight_layout()
    if output_image:
        fig.savefig(output_image, dpi=150)
        print(f"Saved visualization to {output_image}")
    else:
        plt.show()
    print(f"✓ Total windows needed: {len(windows)}")
    print(f"✓ Window size: {window_w} × {window_h}")
    
    return fig

def print_stream_sample(stream, num_samples=20):
    """Print a sample of the pixel stream to show format"""
    print("\n" + "="*60)
    print("SAMPLE OF PIXEL STREAM (first 20 pixels)")
    print("="*60)
    print("Format: (x, y, value) where value: 0=empty, 1=object")
    print("-"*60)
    for i, (x, y, val) in enumerate(stream[:num_samples]):
        pixel_type = "OBJECT" if val == 1 else "empty"
        print(f"  Pixel #{i:3d}: ({x:2d}, {y:2d}) = {val} [{pixel_type}]")
    print("  ... (continues for all pixels)")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Tiled window discovery demo (scanline stream).")
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--image-size", type=int, default=None, help="Square size override.")
    parser.add_argument("--window-width", type=int, default=DEFAULT_WINDOW_WIDTH)
    parser.add_argument("--window-height", type=int, default=DEFAULT_WINDOW_HEIGHT)
    parser.add_argument("--image", type=str, default="", help="Optional real image path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Grayscale threshold (0..1).")
    parser.add_argument("--auto-threshold", action="store_true", help="Use Otsu threshold.")
    parser.add_argument("--invert", action="store_true", help="Invert mask after threshold.")
    parser.add_argument("--close-iter", type=int, default=0, help="Apply binary close this many times.")
    parser.add_argument("--open-iter", type=int, default=0, help="Apply binary open this many times.")
    parser.add_argument("--kernel", type=int, default=3, help="Kernel size for morphology (odd).")
    parser.add_argument("--keep-largest", action="store_true", help="Keep only the largest connected component.")
    parser.add_argument("--min-area", type=int, default=0, help="Keep components with area >= this.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-image", type=str, default="", help="Save visualization to this path instead of showing.")
    parser.add_argument(
        "--overlay-rgb",
        action="store_true",
        help="Overlay tiles on RGB image instead of mask (requires --image).",
    )
    args = parser.parse_args()

    if args.image_size is not None:
        image_w = args.image_size
        image_h = args.image_size
    else:
        image_w = args.image_width
        image_h = args.image_height

    window_w = args.window_width
    window_h = args.window_height

    print("="*60)
    print("FPGA OBJECT DETECTION - TILED CONVOLUTION SIMULATION")
    print("="*60)
    
    rgb_image = None
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            raise SystemExit(f"Image not found: {img_path}")
        print("\n1. Loading real image and thresholding...")
        pixels, used_thresh, rgb_image = load_image_mask(
            img_path, image_w, image_h, args.threshold, args.auto_threshold, args.invert
        )
        print(f"   Using threshold: {used_thresh:.3f}")
        if args.close_iter > 0:
            pixels = binary_close(pixels, k=args.kernel, iters=args.close_iter)
        if args.open_iter > 0:
            pixels = binary_open(pixels, k=args.kernel, iters=args.open_iter)
        if args.keep_largest or args.min_area > 0:
            pixels = keep_largest_component(pixels, min_area=args.min_area)
    else:
        print("\n1. Generating blob pixels with stems/branches...")
        pixels = generate_blob_pixels(image_w, image_h)
    
    print("2. Converting to scanline stream (x-major order)...")
    stream = get_scanline_stream(pixels)
    print(f"   Generated stream with {len(stream)} pixels")
    
    print_stream_sample(stream, num_samples=20)
    
    print(f"\n3. Processing stream to detect windows on-the-fly...")
    windows = calculate_windows_online(stream, window_w, window_h, verbose=args.verbose)
    
    print("\n4. Creating visualization...")
    output_image = args.output_image.strip() or None
    if args.overlay_rgb and rgb_image is None:
        raise SystemExit("--overlay-rgb requires --image")
    overlay = rgb_image if args.overlay_rgb else None
    visualize_tiled_convolution(pixels, windows, stream, window_w, window_h, output_image, overlay)
    
    print("\n" + "="*60)
    print("KEY INSIGHT FOR FPGA:")
    print("="*60)
    print("• Camera sends pixels in scanline order (row by row, left to right)")
    print("• FPGA receives: x=0,y=0, x=1,y=0, x=2,y=0, ... x=79,y=0, x=0,y=1, ...")
    print("• When an object pixel arrives, calculate which window it belongs to:")
    print(f"    window_x = (pixel_x / {window_w}) * {window_w}")
    print(f"    window_y = (pixel_y / {window_h}) * {window_h}")
    print("• Track unique windows in a lookup table/hash")
    print("• Each new window triggers convolution processing")
    print("="*60)
    
    print("\n✓ Complete! Check the visualization.")

if __name__ == "__main__":
    main()
