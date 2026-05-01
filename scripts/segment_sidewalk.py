"""
Run SegFormer on local images and visualize segmentation results.

Supports two model presets:
  cityscapes — 19 classes, street-level real-world photos
  ade20k     — 150 classes, broader scene understanding (better for sim)

Usage:
    python scripts/segment_sidewalk.py                                        # default: ade20k, images/
    python scripts/segment_sidewalk.py --images-dir images/original_screenshots
    python scripts/segment_sidewalk.py --model cityscapes --images-dir images/original_screenshots
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

PRESETS = {
    "cityscapes": {
        "model_name": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        # Classes that represent walkable sidewalk surface
        "sidewalk_labels": {1},   # sidewalk
        "classes": [
            "road", "sidewalk", "building", "wall", "fence",
            "pole", "traffic light", "traffic sign", "vegetation", "terrain",
            "sky", "person", "rider", "car", "truck",
            "bus", "train", "motorcycle", "bicycle",
        ],
        "colors": np.array([
            [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
            [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
            [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
            [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
        ], dtype=np.uint8),
    },
    "ade20k": {
        "model_name": "nvidia/segformer-b5-finetuned-ade-640-640",
        # ADE20K classes relevant to walkable surfaces:
        #   11 = sidewalk/pavement, 52 = path, 53 = stairs, 6 = road
        "sidewalk_labels": {11, 52},
        "classes": None,   # generated below from model config
        "colors": None,    # generated below
    },
}

# ADE20K has 150 classes — generate a stable color palette
_rng = np.random.default_rng(42)
ADE20K_COLORS = _rng.integers(60, 240, size=(150, 3), dtype=np.uint8)
# Override a few key classes with distinct, recognizable colors
ADE20K_COLORS[6]  = [128,  64, 128]   # road       — purple
ADE20K_COLORS[11] = [244,  35, 232]   # sidewalk   — pink
ADE20K_COLORS[52] = [255, 140,   0]   # path       — orange
ADE20K_COLORS[1]  = [ 70,  70,  70]   # building   — dark gray
ADE20K_COLORS[2]  = [ 70, 130, 180]   # sky        — steel blue
ADE20K_COLORS[4]  = [107, 142,  35]   # tree       — olive
ADE20K_COLORS[9]  = [152, 251, 152]   # grass      — light green
ADE20K_COLORS[13] = [210, 180, 140]   # earth/ground — tan

PRESETS["ade20k"]["colors"] = ADE20K_COLORS

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(preset: dict):
    model_name = preset["model_name"]
    print(f"[model] Loading {model_name} ...")
    extractor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[model] Running on {device}")

    # For ADE20K, pull class names from the model's id2label config
    if preset["classes"] is None:
        preset["classes"] = [
            model.config.id2label[i] for i in range(len(model.config.id2label))
        ]

    return extractor, model, device

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def segment_image(image, extractor, model, device) -> np.ndarray:
    """Return (H, W) int array of predicted class indices at original resolution."""
    inputs = extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits  # (1, C, H/4, W/4)

    logits_up = F.interpolate(
        logits,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )
    return logits_up.argmax(dim=1).squeeze(0).cpu().numpy()

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def colorize_segmap(pred: np.ndarray, colors: np.ndarray) -> np.ndarray:
    return colors[pred % len(colors)]


def sidewalk_overlay(image: Image.Image, pred: np.ndarray, sidewalk_labels: set) -> np.ndarray:
    """Highlight sidewalk/path pixels; darken everything else."""
    img = np.array(image.convert("RGB")).astype(np.float32)
    mask = np.isin(pred, list(sidewalk_labels))

    img[~mask] *= 0.35
    img[mask] = img[mask] * 0.5 + np.array([244, 35, 232], dtype=np.float32) * 0.5

    return np.clip(img, 0, 255).astype(np.uint8)


def plot_results(image, pred, image_name, preset, model_key, output_dir: Path):
    colors         = preset["colors"]
    classes        = preset["classes"]
    sidewalk_labels = preset["sidewalk_labels"]

    seg_rgb      = colorize_segmap(pred, colors)
    sw_overlay   = sidewalk_overlay(image, pred, sidewalk_labels)
    sidewalk_pct = np.isin(pred, list(sidewalk_labels)).mean() * 100

    present = np.unique(pred)
    legend_patches = [
        mpatches.Patch(color=colors[c] / 255.0, label=classes[c])
        for c in present
        if c < len(classes)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{image_name}  [{model_key}]", fontsize=14)

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(seg_rgb)
    axes[1].set_title("Segmentation (all classes)")
    axes[1].axis("off")
    axes[1].legend(handles=legend_patches, loc="lower left", fontsize=5, ncol=2, framealpha=0.7)

    labels_str = " + ".join(classes[c] for c in sorted(sidewalk_labels) if c < len(classes))
    axes[2].imshow(sw_overlay)
    axes[2].set_title(f"Sidewalk mask: {labels_str}\n({sidewalk_pct:.1f}% of pixels)")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = output_dir / f"{Path(image_name).stem}_{model_key}_seg.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[save] {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cityscapes", "ade20k"], default="ade20k")
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "images",
                        help="Directory containing input images")
    args = parser.parse_args()

    images_dir = args.images_dir
    output_dir = images_dir / "segmentation_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        if p.parent == images_dir
    ]
    if not image_paths:
        print(f"[error] No images found in {images_dir}")
        return

    print(f"[info] {len(image_paths)} images found in {images_dir}")
    preset = PRESETS[args.model]
    extractor, model, device = load_model(preset)

    for img_path in image_paths:
        print(f"\n[segment] {img_path.name}")
        image = Image.open(img_path).convert("RGB")
        pred  = segment_image(image, extractor, model, device)

        classes = preset["classes"]
        sidewalk_labels = preset["sidewalk_labels"]
        detected = [classes[c] for c in np.unique(pred) if c < len(classes)]
        print(f"  Classes detected: {detected}")
        print(f"  Sidewalk/path coverage: {np.isin(pred, list(sidewalk_labels)).mean() * 100:.1f}%")

        plot_results(image, pred, img_path.name, preset, args.model, output_dir)

    print(f"\n[done] Results saved to {output_dir}")


if __name__ == "__main__":
    main()
