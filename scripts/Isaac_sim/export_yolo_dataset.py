"""
Convert the captured defect dataset (RGB + instance segmentation masks)
into YOLO-seg format: one label .txt per image with normalized polygon
coordinates, a data.yaml, and a train/val split.

Runs OFFLINE (no Isaac needed) -- pure Python/numpy/cv2/PIL, once
capture_defect_dataset.py has produced the raw dataset.

YOLO-seg label format per image (one line per instance):
  <class_id> x1 y1 x2 y2 x3 y3 ... xn yn      (all coords normalized 0-1)

We use INSTANCE segmentation (not semantic) as the mask source, since each
crack is its own instance ID -- this gives one polygon per defect even
when several appear in one frame, which semantic segmentation (one ID per
CLASS) can't distinguish.

Also renders a handful of mask-overlay sanity-check images so you can
visually confirm the polygons are correct BEFORE training -- catching a
bad labeling assumption here is much cheaper than after a training run.
"""

import os
import json
import random
import numpy as np
import cv2
from PIL import Image


# =====================================================================
# CONFIG
# =====================================================================

DATASET_DIR = "C:/Users/Elyas/sidewalk_dataset/crack_dataset/capture"
OUT_DIR = "C:/Users/Elyas/sidewalk_dataset/crack_dataset/yolo"

# Class map -- extend this as more classes come online (Meshy assets).
CLASS_NAMES = ["crack"]  # scope narrowed to crack-only detection

VAL_FRACTION = 0.15
RNG_SEED = 3

# Polygon simplification: cv2.approxPolyDP epsilon as a fraction of the
# contour's perimeter. Too low = huge polygons with noisy vertices; too
# high = lost detail. 0.005 is a reasonable default for smooth defect
# shapes.
POLY_EPSILON_FRAC = 0.005
MIN_CONTOUR_AREA_PX = 15   # drop tiny specks (rendering noise, not defects)

N_SANITY_IMAGES = 20


# =====================================================================
# MASK -> POLYGON
# =====================================================================

def instance_mask_to_polygons(inst_map, background_ids=(0,)):
    """Given an instance-ID image (HxW int), return {instance_id: [poly]}
    where each poly is an (N,2) array of pixel coords. An instance can
    yield multiple polygons if its mask is disjoint (rare, but handled)."""
    polys_by_id = {}
    ids = np.unique(inst_map)
    for iid in ids:
        if iid in background_ids:
            continue
        mask = (inst_map == iid).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polys = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA_PX:
                continue
            peri = cv2.arcLength(c, True)
            eps = POLY_EPSILON_FRAC * max(peri, 1e-6)
            approx = cv2.approxPolyDP(c, eps, True)
            pts = approx.reshape(-1, 2)
            if len(pts) >= 3:
                polys.append(pts)
        if polys:
            polys_by_id[int(iid)] = polys
    return polys_by_id


def normalize_polygon(pts, w, h):
    """(N,2) pixel coords -> flat normalized [x1,y1,x2,y2,...] in [0,1]."""
    norm = pts.astype(np.float64).copy()
    norm[:, 0] /= w
    norm[:, 1] /= h
    norm = np.clip(norm, 0.0, 1.0)
    return norm.flatten().tolist()


# =====================================================================
# LABEL WRITING
# =====================================================================

def class_id_for_defect(defect_name, defect_class, class_names):
    """Map a defect's recorded class (from the capture index / ground
    truth) to a YOLO class id. Falls back to 0 with a warning if unknown
    -- keeps the exporter usable even if new classes show up before the
    class list is updated."""
    if defect_class in class_names:
        return class_names.index(defect_class)
    return 0


def build_yolo_labels(dataset_dir, out_dir, class_names,
                      poly_epsilon_frac=POLY_EPSILON_FRAC,
                      min_area=MIN_CONTOUR_AREA_PX):
    """Main conversion: reads capture_index.json + instance masks, writes
    YOLO-seg .txt labels. Returns list of (image_file, label_file, n_objs)
    for reporting / sanity selection."""
    index_path = os.path.join(dataset_dir, "capture_index.json")
    with open(index_path) as fh:
        index = json.load(fh)

    w, h = index["resolution"]
    inst_dir = os.path.join(dataset_dir, "instance")
    rgb_dir = os.path.join(dataset_dir, "rgb")

    labels_dir = os.path.join(out_dir, "labels")
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    results = []
    for frame in index["frames"]:
        fname = frame["file"]
        inst_path = os.path.join(inst_dir, fname + ".png")
        if not os.path.exists(inst_path):
            continue
        inst_map = np.array(Image.open(inst_path))

        polys_by_id = instance_mask_to_polygons(inst_map)

        lines = []
        for iid, polys in polys_by_id.items():
            cls_id = class_id_for_defect(frame["defect"], frame["class"],
                                          class_names)
            for pts in polys:
                coords = normalize_polygon(pts, w, h)
                lines.append(
                    str(cls_id) + " " + " ".join(f"{c:.6f}" for c in coords)
                )

        label_path = os.path.join(labels_dir, fname + ".txt")
        with open(label_path, "w") as fh:
            fh.write("\n".join(lines))

        results.append((fname, len(lines)))

    return results, rgb_dir, images_dir, labels_dir, w, h


def split_and_link(results, rgb_dir, images_dir, labels_dir, out_dir,
                   val_fraction, rng):
    """Copy images into images/train, images/val (labels stay flat and get
    referenced by matching filename, per YOLO convention)."""
    names = [r[0] for r in results]
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_fraction))
    val_set = set(names[:n_val])

    for split in ("train", "val"):
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

    for fname in names:
        split = "val" if fname in val_set else "train"
        src_img = os.path.join(rgb_dir, fname + ".png")
        dst_img = os.path.join(images_dir, split, fname + ".png")
        if os.path.exists(src_img):
            Image.open(src_img).save(dst_img)

        src_lbl = os.path.join(labels_dir, fname + ".txt")
        dst_lbl = os.path.join(labels_dir, split, fname + ".txt")
        if os.path.exists(src_lbl):
            with open(src_lbl) as fh:
                content = fh.read()
            with open(dst_lbl, "w") as fh:
                fh.write(content)
            os.remove(src_lbl)

    return len(names) - n_val, n_val


def write_data_yaml(out_dir, class_names):
    path = os.path.join(out_dir, "data.yaml")
    with open(path, "w") as fh:
        fh.write(f"path: {out_dir}\n")
        fh.write("train: images/train\n")
        fh.write("val: images/val\n")
        fh.write(f"nc: {len(class_names)}\n")
        fh.write("names: [" + ", ".join(f"'{n}'" for n in class_names) + "]\n")
    return path


# =====================================================================
# SANITY CHECK: render mask overlays so labels can be visually verified
# =====================================================================

def render_sanity_overlays(images_dir, labels_dir, out_dir, n, rng):
    check_dir = os.path.join(out_dir, "sanity_check")
    os.makedirs(check_dir, exist_ok=True)

    all_imgs = []
    for split in ("train", "val"):
        d = os.path.join(images_dir, split)
        if os.path.isdir(d):
            all_imgs += [(split, f) for f in os.listdir(d)
                        if f.endswith(".png")]
    sample = rng.sample(all_imgs, min(n, len(all_imgs)))

    for split, fname in sample:
        img_path = os.path.join(images_dir, split, fname)
        lbl_path = os.path.join(labels_dir, split,
                                fname.replace(".png", ".txt"))
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]
        overlay = img.copy()

        if os.path.exists(lbl_path):
            with open(lbl_path) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    coords = list(map(float, parts[1:]))
                    pts = np.array(coords).reshape(-1, 2)
                    pts[:, 0] *= w
                    pts[:, 1] *= h
                    pts = pts.astype(np.int32)
                    cv2.polylines(overlay, [pts], True, (255, 60, 60), 2)
                    cv2.fillPoly(overlay, [pts],
                                (255, 60, 60), lineType=cv2.LINE_AA)
        blended = (0.6 * img + 0.4 * overlay).astype(np.uint8)
        Image.fromarray(blended).save(
            os.path.join(check_dir, f"check_{split}_{fname}"))

    print(f"✓ Wrote {len(sample)} sanity-check overlays -> {check_dir}")
    print("  Open a few of these and confirm the highlighted region "
          "actually traces the crack before training.")


# =====================================================================
# MAIN
# =====================================================================

def run():
    rng = random.Random(RNG_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    results, rgb_dir, images_dir, labels_dir, w, h = build_yolo_labels(
        DATASET_DIR, OUT_DIR, CLASS_NAMES
    )
    n_train, n_val = split_and_link(
        results, rgb_dir, images_dir, labels_dir, OUT_DIR,
        VAL_FRACTION, rng
    )
    yaml_path = write_data_yaml(OUT_DIR, CLASS_NAMES)

    n_empty = sum(1 for _, n in results if n == 0)
    print(f"Converted {len(results)} frames -> {OUT_DIR}")
    print(f"  train: {n_train}   val: {n_val}")
    print(f"  frames with zero instances found: {n_empty} "
          f"({'check instance masks if this is high' if n_empty > len(results)*0.1 else 'OK'})")
    print(f"  data.yaml -> {yaml_path}")

    render_sanity_overlays(images_dir, labels_dir, OUT_DIR,
                           N_SANITY_IMAGES, rng)


if __name__ == "__main__":
    run()
