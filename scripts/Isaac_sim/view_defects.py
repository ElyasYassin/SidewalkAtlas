import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter
import os

DATASET = "C:/Users/Elyas/sidewalk_dataset"
RUN = "run_20260519_220910"  # ← change to your folder
RUN_DIR = os.path.join(DATASET, RUN)

depth_m = np.load(os.path.join(RUN_DIR, "depth_clean.npy"))
rgb = np.array(Image.open(os.path.join(RUN_DIR, "rgb.png")))

valid = (depth_m > 0) & np.isfinite(depth_m)
depth_safe = np.where(valid, depth_m, 0)

# --- Local plane subtraction (handles multi-slab scenes) ---
window = 80
depth_smooth = uniform_filter(depth_safe, size=window)
# Account for invalid pixels in the smoothing
weight = uniform_filter(valid.astype(np.float32), size=window)
depth_smooth = np.where(weight > 0.1, depth_smooth / np.maximum(weight, 1e-3), 0)

local_dev_mm = (depth_m - depth_smooth) * 1000
local_dev_mm[~valid] = np.nan

# --- Gradient magnitude (edge detector) ---
gy, gx = np.gradient(depth_safe)
grad_mm = np.sqrt(gx**2 + gy**2) * 1000

fig, axes = plt.subplots(2, 2, figsize=(16, 9))

axes[0, 0].imshow(rgb)
axes[0, 0].set_title("RGB")
axes[0, 0].axis("off")

im1 = axes[0, 1].imshow(local_dev_mm, cmap="RdBu_r", vmin=-5, vmax=5)
axes[0, 1].set_title("Local deviation ±5mm (slabs averaged out)")
axes[0, 1].axis("off")
plt.colorbar(im1, ax=axes[0, 1], label="mm", fraction=0.04)

im2 = axes[1, 0].imshow(local_dev_mm, cmap="RdBu_r", vmin=-15, vmax=15)
axes[1, 0].set_title("Local deviation ±15mm")
axes[1, 0].axis("off")
plt.colorbar(im2, ax=axes[1, 0], label="mm", fraction=0.04)

im3 = axes[1, 1].imshow(grad_mm, cmap="hot", vmin=0, vmax=20)
axes[1, 1].set_title("Depth gradient — slab edges / cracks")
axes[1, 1].axis("off")
plt.colorbar(im3, ax=axes[1, 1], label="mm/px", fraction=0.04)

plt.tight_layout()
out_path = os.path.join(RUN_DIR, "defect_view_local.png")
plt.savefig(out_path, dpi=100, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

# Stats per region
sw = local_dev_mm[~np.isnan(local_dev_mm)]
print(f"Local deviation stats:")
print(f"  Std: {sw.std():.2f} mm")
print(f"  Range: {sw.min():.1f} to {sw.max():.1f} mm")