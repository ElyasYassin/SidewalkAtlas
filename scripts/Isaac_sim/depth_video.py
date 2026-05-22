import os
import glob
import time
import numpy as np
import cv2


# =====================================================================
# CONFIG
# =====================================================================

# Change this to your actual capture run folder
RUN_DIR = r"C:\Users\Elyas\sidewalk_dataset\walk_captures\run_20260522_155926"

DEPTH_NPY_DIR = os.path.join(RUN_DIR, "depth_npy")

FPS = 10

# Same style as your bottom-right plot
GRADIENT_MAX_MM_PER_PX = 20.0

SAVE_VIDEO = True
OUT_VIDEO_PATH = os.path.join(RUN_DIR, "depth_gradient_video.mp4")


# =====================================================================
# HELPERS
# =====================================================================

def depth_to_gradient_view(depth_m, vmax=20.0):
    """
    Create a depth-gradient visualization like the bottom-right image:
    'Depth gradient — slab edges / cracks'

    Bright/hot = sharp depth edge
    Dark = smooth/flat region
    """
    depth = depth_m.astype(np.float32)

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    valid = np.isfinite(depth) & (depth > 0)

    # Replace invalid depth with 0 for gradient computation
    depth_safe = np.where(valid, depth, 0.0)

    gy, gx = np.gradient(depth_safe)

    # Convert meters/pixel to millimeters/pixel
    grad_mm = np.sqrt(gx ** 2 + gy ** 2) * 1000.0

    # Clamp to match the visual scale from your plot
    grad_clipped = np.clip(grad_mm, 0.0, vmax)

    # Normalize 0..vmax to 0..255
    gray = (grad_clipped / vmax * 255.0).astype(np.uint8)

    # Use HOT colormap to match the bottom-right figure
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

    # Invalid pixels black
    colored[~valid] = (0, 0, 0)

    return colored


def add_text_overlay(frame, text):
    out = frame.copy()

    cv2.putText(
        out,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


# =====================================================================
# MAIN
# =====================================================================

def main():
    depth_files = sorted(glob.glob(os.path.join(DEPTH_NPY_DIR, "*.npy")))

    if not depth_files:
        raise FileNotFoundError(f"No depth .npy files found in: {DEPTH_NPY_DIR}")

    print(f"Found {len(depth_files)} depth frames.")
    print(f"Playing depth-gradient video from: {DEPTH_NPY_DIR}")

    first_depth = np.load(depth_files[0])

    if first_depth.ndim == 3 and first_depth.shape[-1] == 1:
        first_depth = first_depth[..., 0]

    h, w = first_depth.shape[:2]

    writer = None

    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, FPS, (w, h))
        print(f"Saving video to: {OUT_VIDEO_PATH}")

    delay_ms = int(1000 / FPS)

    for i, path in enumerate(depth_files):
        depth_m = np.load(path)

        frame = depth_to_gradient_view(
            depth_m,
            vmax=GRADIENT_MAX_MM_PER_PX,
        )

        filename = os.path.basename(path)
        frame = add_text_overlay(
            frame,
            f"{i + 1}/{len(depth_files)}  {filename}",
        )

        cv2.imshow("Depth Gradient Video - Press Q to quit", frame)

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(delay_ms) & 0xFF

        if key == ord("q"):
            break

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

    print("Done.")

    if SAVE_VIDEO:
        print(f"Saved video: {OUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()