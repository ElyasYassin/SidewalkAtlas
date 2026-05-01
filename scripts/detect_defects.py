"""
Sidewalk defect detection using Grounding DINO (zero-shot).

Pipeline:
  1. SegFormer (ADE20K) → sidewalk mask
  2. Mask applied to image — non-sidewalk pixels blacked out
  3. Grounding DINO → detect defects via text prompts inside the mask
  4. Bounding boxes drawn on the original image

Usage:
    python scripts/detect_defects.py
    python scripts/detect_defects.py --images-dir images/original_screenshots
    python scripts/detect_defects.py --images-dir images/original_screenshots --box-threshold 0.25
"""

import argparse
import inspect
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent

SEGFORMER_MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"
GDINO_MODEL     = "IDEA-Research/grounding-dino-base"

# ADE20K class indices for walkable sidewalk surfaces
SIDEWALK_LABELS = {11, 52}   # sidewalk/pavement, path

# Text prompts fed to Grounding DINO — period-separated, each is a category
DEFECT_PROMPTS = "crack . pothole . broken concrete . uneven surface . damaged pavement . collapsed sidewalk ."

# Detection thresholds — lower = more detections but more false positives
DEFAULT_BOX_THRESHOLD  = 0.30
DEFAULT_TEXT_THRESHOLD = 0.25

# Color per defect label (BGR-style for drawing)
LABEL_COLORS = {
    "crack":              "#FF4444",
    "pothole":            "#FF8C00",
    "broken concrete":    "#FFD700",
    "uneven surface":     "#00BFFF",
    "damaged pavement":   "#DA70D6",
    "collapsed sidewalk": "#FF69B4",
}
DEFAULT_COLOR = "#FFFFFF"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_segformer(device: str):
    print(f"[segformer] Loading {SEGFORMER_MODEL} ...")
    extractor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
    model.eval().to(device)
    return extractor, model


def load_gdino(device: str):
    print(f"[grounding-dino] Loading {GDINO_MODEL} ...")
    processor = AutoProcessor.from_pretrained(GDINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL)
    model.eval().to(device)
    return processor, model

# ---------------------------------------------------------------------------
# Step 1 — Sidewalk mask
# ---------------------------------------------------------------------------

def get_sidewalk_mask(
    image: Image.Image,
    extractor,
    seg_model,
    device: str,
) -> np.ndarray:
    """Return a boolean (H, W) mask — True where sidewalk/path is predicted."""
    inputs = extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = seg_model(**inputs).logits  # (1, C, H/4, W/4)

    logits_up = F.interpolate(
        logits,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )
    pred = logits_up.argmax(dim=1).squeeze(0).cpu().numpy()
    mask = np.isin(pred, list(SIDEWALK_LABELS))
    coverage = mask.mean() * 100
    print(f"  Sidewalk mask coverage: {coverage:.1f}%")
    return mask

# ---------------------------------------------------------------------------
# Step 2 — Apply mask
# ---------------------------------------------------------------------------

def apply_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Black out all pixels outside the sidewalk mask."""
    img_arr = np.array(image.convert("RGB"))
    img_arr[~mask] = 0
    return Image.fromarray(img_arr)

# ---------------------------------------------------------------------------
# Step 3 — Grounding DINO detection
# ---------------------------------------------------------------------------

def detect_defects(
    masked_image: Image.Image,
    processor,
    gdino_model,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> list[dict]:
    """Return list of {label, score, box} dicts. Box is [x0, y0, x1, y1] in pixels."""
    inputs = processor(
        images=masked_image,
        text=DEFECT_PROMPTS,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = gdino_model(**inputs)

    # transformers 5.x renamed the threshold params — handle both API versions
    sig_params = inspect.signature(processor.post_process_grounded_object_detection).parameters
    if "box_threshold" in sig_params:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[masked_image.size[::-1]],
        )[0]
    else:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            target_sizes=[masked_image.size[::-1]],
        )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append({
            "label": label,
            "score": float(score),
            "box":   [float(v) for v in box],
        })
    return detections

# ---------------------------------------------------------------------------
# Step 4 — Visualize
# ---------------------------------------------------------------------------

def draw_detections(
    original: Image.Image,
    mask: np.ndarray,
    detections: list[dict],
    image_name: str,
    model_info: str,
    output_dir: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"{image_name}  |  {model_info}", fontsize=13)

    # Panel 1 — original
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Panel 2 — sidewalk mask
    mask_vis = np.array(original.convert("RGB")).astype(np.float32)
    mask_vis[~mask] *= 0.3
    mask_vis[mask] = mask_vis[mask] * 0.5 + np.array([244, 35, 232]) * 0.5
    axes[1].imshow(np.clip(mask_vis, 0, 255).astype(np.uint8))
    axes[1].set_title(f"Sidewalk mask ({mask.mean()*100:.1f}% of pixels)")
    axes[1].axis("off")

    # Panel 3 — detections on original
    axes[2].imshow(original)
    axes[2].set_title(f"Defect detections ({len(detections)} found)")
    axes[2].axis("off")

    for det in detections:
        x0, y0, x1, y1 = det["box"]
        label = det["label"]
        score = det["score"]
        color = LABEL_COLORS.get(label, DEFAULT_COLOR)

        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        axes[2].add_patch(rect)
        axes[2].text(
            x0, y0 - 4,
            f"{label} {score:.2f}",
            color=color,
            fontsize=7,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"),
        )

    if not detections:
        axes[2].text(
            0.5, 0.5, "No defects detected\n(try lowering --box-threshold)",
            ha="center", va="center", transform=axes[2].transAxes,
            color="white", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6, pad=6),
        )

    plt.tight_layout()
    out_path = output_dir / f"{Path(image_name).stem}_defects.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [save] {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "images",
                        help="Directory containing input images")
    parser.add_argument("--box-threshold",  type=float, default=DEFAULT_BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    args = parser.parse_args()

    images_dir = args.images_dir
    output_dir = images_dir / "defect_detection_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}\n")

    seg_extractor, seg_model = load_segformer(device)
    gdino_processor, gdino_model = load_gdino(device)

    image_paths = [
        p for p in sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        if p.parent == images_dir
    ]
    if not image_paths:
        print(f"[error] No images found in {images_dir}")
        return

    print(f"[info] {len(image_paths)} images found in {images_dir}")

    for img_path in image_paths:
        print(f"\n[image] {img_path.name}")
        image = Image.open(img_path).convert("RGB")

        mask = get_sidewalk_mask(image, seg_extractor, seg_model, device)
        if mask.mean() == 0:
            print(f"  [skip] No sidewalk detected — likely non-street-level view")
            continue

        masked_image = apply_mask(image, mask)
        detections   = detect_defects(
            masked_image, gdino_processor, gdino_model, device,
            args.box_threshold, args.text_threshold,
        )

        print(f"  Detections ({len(detections)}):")
        for d in detections:
            print(f"    {d['label']:30s}  score={d['score']:.3f}  box={[round(v) for v in d['box']]}")

        model_info = f"SegFormer ADE20K + Grounding DINO  |  box≥{args.box_threshold}  text≥{args.text_threshold}"
        draw_detections(image, mask, detections, img_path.name, model_info, output_dir)

    print(f"\n[done] Results in {output_dir}")


if __name__ == "__main__":
    main()
