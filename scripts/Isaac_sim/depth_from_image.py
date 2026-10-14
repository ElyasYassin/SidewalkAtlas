#IMAGE_PATH = r"C:\Users\Elyas\OneDrive - The University of Colorado Denver\Desktop\projects\SidewalkAtlas\scripts\Isaac_sim\depth_images_examples\frame_00042.jpg"

#OUT_DIR = r"C:\Users\Elyas\OneDrive - The University of Colorado Denver\Desktop\projects\SidewalkAtlas\scripts\Isaac_sim\depth_images_examples\example_depth_output"

import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import pipeline


# =====================================================================
# CONFIG
# =====================================================================

IMAGE_PATH = r"C:\Users\Elyas\OneDrive - The University of Colorado Denver\Desktop\projects\SidewalkAtlas\scripts\Isaac_sim\depth_images_examples\frame_00042.jpg"

OUT_DIR = r"C:\Users\Elyas\OneDrive - The University of Colorado Denver\Desktop\projects\SidewalkAtlas\scripts\Isaac_sim\depth_images_examples\example_depth_output"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_DEPTH_NPY = os.path.join(OUT_DIR, "depth_anything_depth.npy")
OUT_DEPTH_VIS = os.path.join(OUT_DIR, "depth_anything_visual.png")
OUT_GRADIENT_VIS = os.path.join(OUT_DIR, "depth_anything_gradient.png")

MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

# Since monocular depth is relative, this is just for visualization scaling.
DEPTH_MIN_M = 1.0
DEPTH_MAX_M = 5.0

# Try 10, 20, 50, 100 depending on how saturated the gradient looks.
GRADIENT_MAX_MM_PER_PX = 50.0


# =====================================================================
# VISUALIZATION HELPERS
# =====================================================================

def depth_to_gradient_view(depth_m, vmax=50.0):
    depth = depth_m.astype(np.float32)

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    valid = np.isfinite(depth) & (depth > 0)

    # Smooth the depth before gradient to suppress small noise
    depth_smooth = cv2.bilateralFilter(depth, d=7, sigmaColor=0.1, sigmaSpace=7)
    depth_safe = np.where(valid, depth_smooth, 0.0)

    gy, gx = np.gradient(depth_safe)

    # meters/pixel -> millimeters/pixel
    grad_mm = np.sqrt(gx ** 2 + gy ** 2) * 1000.0

    grad_clipped = np.clip(grad_mm, 0.0, vmax)
    gray = (grad_clipped / vmax * 255.0).astype(np.uint8)

    colored = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    colored[~valid] = (0, 0, 0)

    return colored


def depth_to_visual(depth_m):
    valid = np.isfinite(depth_m) & (depth_m > 0)

    min_d = np.percentile(depth_m[valid], 2)
    max_d = np.percentile(depth_m[valid], 98)

    depth_norm = np.clip((depth_m - min_d) / (max_d - min_d + 1e-6), 0, 1)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    colored[~valid] = (0, 0, 0)

    return colored


# =====================================================================
# DEPTH ESTIMATION
# =====================================================================

def estimate_depth(image_path):
    device = 0 if torch.cuda.is_available() else -1

    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")

    pipe = pipeline(
        task="depth-estimation",
        model=MODEL_NAME,
        device=device,
    )

    image = Image.open(image_path).convert("RGB")
    result = pipe(image)

    # Hugging Face depth-estimation pipeline returns a PIL depth image
    depth_pil = result["depth"]
    depth_raw = np.array(depth_pil).astype(np.float32)

    # Normalize relative depth to 0..1
    depth_norm = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min() + 1e-6)

    # Convert relative depth into fake metric range for compatibility with your gradient code.
    # This is NOT true metric depth, but the spatial structure comes from the model.
    depth_m = DEPTH_MIN_M + depth_norm * (DEPTH_MAX_M - DEPTH_MIN_M)

    # Resize back to original image size if needed
    original = cv2.imread(image_path)
    h, w = original.shape[:2]

    if depth_m.shape[:2] != (h, w):
        depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_CUBIC)

    return depth_m.astype(np.float32)


# =====================================================================
# MAIN
# =====================================================================

def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    image_bgr = cv2.imread(IMAGE_PATH)

    print(f"Loaded image: {IMAGE_PATH}")
    print(f"Image shape: {image_bgr.shape}")

    depth_m = estimate_depth(IMAGE_PATH)

    np.save(OUT_DEPTH_NPY, depth_m)
    print(f"Saved depth .npy: {OUT_DEPTH_NPY}")

    depth_vis = depth_to_visual(depth_m)
    gradient_vis = depth_to_gradient_view(
        depth_m,
        vmax=GRADIENT_MAX_MM_PER_PX,
    )

    cv2.imwrite(OUT_DEPTH_VIS, depth_vis)
    cv2.imwrite(OUT_GRADIENT_VIS, gradient_vis)

    print(f"Saved depth visual: {OUT_DEPTH_VIS}")
    print(f"Saved gradient visual: {OUT_GRADIENT_VIS}")

    cv2.imshow("Original", image_bgr)
    cv2.imshow("Depth Anything Depth", depth_vis)
    cv2.imshow("Depth Gradient", gradient_vis)

    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()