import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json

DATASET = "C:/Users/Elyas/sidewalk_dataset/sidewalk_run_001"
OUTPUT_DIR = f"{DATASET}/previews"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get list of frames
frame_files = sorted(os.listdir(f"{DATASET}/rgb"))
frame_ids = [f.replace(".png", "") for f in frame_files]

print(f"Found {len(frame_ids)} frames")

def view_frame(frame_id, save_path=None):
    """Show all channels for one frame in a 2x3 grid."""
    
    # Load all channels
    rgb = np.array(Image.open(f"{DATASET}/rgb/{frame_id}.png"))
    depth_clean = np.load(f"{DATASET}/depth_clean/{frame_id}.npy")
    depth_sensor = np.load(f"{DATASET}/depth_sensor/{frame_id}.npy")
    normals_img = np.array(Image.open(f"{DATASET}/normals/{frame_id}.png"))
    
    # Load semantic if available
    try:
        semantic = np.load(f"{DATASET}/semantic/{frame_id}.npy")
        with open(f"{DATASET}/semantic/{frame_id}_labels.json") as f:
            labels = json.load(f)
    except Exception:
        semantic = None
        labels = {}
    
    # Compute valid depth range for visualization
    valid = depth_clean > 0
    vmin = np.percentile(depth_clean[valid], 1) if valid.any() else 0
    vmax = np.percentile(depth_clean[valid], 99) if valid.any() else 10
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis("off")
    
    im1 = axes[0, 1].imshow(depth_clean, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"Depth Clean ({vmin:.2f}-{vmax:.2f}m)")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.04)
    
    im2 = axes[0, 2].imshow(depth_sensor, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("Depth Sensor (with noise)")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.04)
    
    axes[1, 0].imshow(normals_img)
    axes[1, 0].set_title("Surface Normals")
    axes[1, 0].axis("off")
    
    if semantic is not None:
        im3 = axes[1, 1].imshow(semantic, cmap="tab20")
        axes[1, 1].set_title(f"Semantic ({len(labels)} classes)")
        axes[1, 1].axis("off")
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.04)
    
    # Difference between clean and sensor depth (shows noise pattern)
    diff = (depth_sensor - depth_clean) * 1000  # mm
    diff[depth_sensor == 0] = 0
    diff[depth_clean == 0] = 0
    im4 = axes[1, 2].imshow(diff, cmap="RdBu_r", vmin=-50, vmax=50)
    axes[1, 2].set_title("Sensor noise (mm)")
    axes[1, 2].axis("off")
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.04)
    
    plt.suptitle(f"Frame {frame_id}", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=80, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# --- Generate previews for first 10 frames ---
preview_count = min(10, len(frame_ids))
for fid in frame_ids[:preview_count]:
    out_path = f"{OUTPUT_DIR}/preview_{fid}.png"
    view_frame(fid, save_path=out_path)
    print(f"Saved {out_path}")

print(f"\nPreviews in: {OUTPUT_DIR}")
print("Open them in any image viewer to inspect.")