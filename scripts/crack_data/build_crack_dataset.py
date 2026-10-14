"""
Assemble rendered frames into a training dataset (offline, no Isaac).

    python scripts/crack_data/build_crack_dataset.py
    python scripts/crack_data/build_crack_dataset.py --no-yolo

Writes into <render>/dataset/:
  cracks.csv            one row per crack (scene-level truth), columns of
                        crack_detection's visual_review.csv + generator extras
  crack_instances.csv   one row per (frame, visible crack): the same
                        measurements plus image-space visibility -- the
                        detection + measurement training table
  splits.json           train / val / test by SCENE, never by frame
  yolo/                 YOLO-seg export (images hard-linked, polygons from
                        the exact masks), data.yaml

Measurements are the crack's true geometry and do not change with the view;
`visible_pixels` and `centerline_visible_fraction` say how much of the crack
the frame actually shows. Train the measurement head only on instances that
are mostly in view (e.g. centerline_visible_fraction >= 0.9), or a model is
asked to predict the length of a crack it can only half see.
"""

import argparse
import csv
import hashlib
import json
import os

import cv2
import numpy as np


# =====================================================================
# CONFIG
# =====================================================================

RENDER_DIR = "D:/sidewalk_crack_synth/render"
SCENES_DIR = "D:/sidewalk_crack_synth/scenes"
SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}
MIN_VISIBLE_PX = 20
POLY_EPSILON_PX = 0.8
MIN_POLY_AREA_PX = 4.0

# visual_review.csv column order, so both sources load with one reader.
REVIEW_COLUMNS = [
    "sidewalk_prim", "defect_id", "source_label", "detected_class", "polarity",
    "review_decision", "confirmed_class", "review_notes", "review_image",
    "review_source", "length_m", "mean_width_m", "median_width_m",
    "max_width_m", "max_depth_m", "mean_depth_m", "median_depth_m",
    "max_height_m", "mean_height_m", "projected_area_m2", "pixel_count",
]
EXTRA_CRACK = ["scene_id", "instance_id", "split", "pattern", "width_class",
               "n_branches", "tortuosity", "pca_length_m", "pca_width_m",
               "area_over_length_m", "visible_raster_cells", "fill_fraction"]
EXTRA_INSTANCE = ["frame", "rgb", "depth", "depth_stereo", "mask",
                  "visible_pixels", "ignore_pixels",
                  "centerline_visible_fraction", "width_px_at_center",
                  "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "range_m",
                  "view_elev_deg", "cam_height_m", "cam_pitch_deg", "hfov_deg",
                  "wet", "overcast", "hdri", "surface", "fill_fraction"]


def split_of(scene_id):
    """Deterministic by scene id: adding scenes never moves existing ones."""
    h = int(hashlib.sha1(scene_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, frac in SPLIT.items():
        acc += frac
        if h < acc:
            return name
    return "test"


def review_row(scene_id, rec, image=""):
    row = {c: rec.get(c, "") for c in REVIEW_COLUMNS}
    row.update({"sidewalk_prim": scene_id, "review_decision": "synthetic_exact",
                "review_notes": "", "review_image": image,
                "review_source": "isaac_synthetic"})
    return row


def mask_polygons(mask, iid):
    m = (mask == iid).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < MIN_POLY_AREA_PX and len(c) < 4:
            continue
        a = cv2.approxPolyDP(c, POLY_EPSILON_PX, True).reshape(-1, 2)
        if len(a) >= 3:
            out.append(a)
    return out


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default=RENDER_DIR)
    ap.add_argument("--scenes", default=SCENES_DIR)
    ap.add_argument("--no-yolo", action="store_true")
    a = ap.parse_args()
    out = os.path.join(a.render, "dataset")
    os.makedirs(out, exist_ok=True)

    frames = [json.loads(l) for l in open(os.path.join(a.render, "frames.jsonl"))
              if l.strip()]
    scene_ids = sorted({f["scene_id"] for f in frames})
    metas = {s: json.load(open(os.path.join(a.scenes, s, "scene.json")))
             for s in scene_ids}
    splits = {s: split_of(s) for s in scene_ids}

    # --- per-crack truth ------------------------------------------------
    with open(os.path.join(out, "cracks.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, REVIEW_COLUMNS + EXTRA_CRACK)
        w.writeheader()
        for s in scene_ids:
            for rec in metas[s]["cracks"]:
                row = review_row(s, rec)
                row.update({k: rec.get(k, "") for k in EXTRA_CRACK})
                row.update({"scene_id": s, "split": splits[s]})
                w.writerow(row)

    # --- per-frame instances + YOLO ------------------------------------
    ydir = os.path.join(out, "yolo")
    if not a.no_yolo:
        for sp in SPLIT:
            os.makedirs(os.path.join(ydir, "images", sp), exist_ok=True)
            os.makedirs(os.path.join(ydir, "labels", sp), exist_ok=True)
    n_inst = 0
    counts = {sp: {"scenes": 0, "frames": 0, "instances": 0} for sp in SPLIT}
    for s in scene_ids:
        counts[splits[s]]["scenes"] += 1
    with open(os.path.join(out, "crack_instances.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, REVIEW_COLUMNS + EXTRA_CRACK + EXTRA_INSTANCE)
        w.writeheader()
        for f in frames:
            s, name = f["scene_id"], f["file"]
            sp = splits[s]
            counts[sp]["frames"] += 1
            recs = {r["instance_id"]: r for r in metas[s]["cracks"]}
            rgb = os.path.join(a.render, "rgb", name + ".png")
            mask_p = os.path.join(a.render, "mask", name + ".png")
            mask = None
            lines = []
            for v in f["cracks"]:
                if v["visible_pixels"] < MIN_VISIBLE_PX:
                    continue
                rec = recs[v["instance_id"]]
                row = review_row(s, rec, rgb)
                row.update({k: rec.get(k, "") for k in EXTRA_CRACK})
                bb = v.get("bbox_xyxy", ["", "", "", ""])
                row.update({
                    "scene_id": s, "split": sp, "frame": name, "rgb": rgb,
                    "depth": os.path.join(a.render, "depth", name + ".png"),
                    "depth_stereo": os.path.join(a.render, "depth_stereo", name + ".png"),
                    "mask": mask_p, "visible_pixels": v["visible_pixels"],
                    "ignore_pixels": v.get("ignore_pixels", 0),
                    "width_px_at_center": v.get("width_px_at_center", ""),
                    "surface": f.get("surface", ""),
                    "centerline_visible_fraction": v["centerline_visible_fraction"],
                    "bbox_x0": bb[0], "bbox_y0": bb[1], "bbox_x1": bb[2],
                    "bbox_y1": bb[3], "range_m": v.get("range_m", ""),
                    "view_elev_deg": v.get("view_elev_deg", ""),
                    "cam_height_m": f["pose"]["height_m"],
                    "cam_pitch_deg": f["pose"]["pitch_deg"],
                    "hfov_deg": f["camera"]["hfov_deg"],
                    "wet": f["material"].get("wet", False),
                    "overcast": f["lighting"]["overcast"],
                    "hdri": f["lighting"].get("hdri") or "",
                })
                w.writerow(row)
                n_inst += 1
                counts[sp]["instances"] += 1
                if not a.no_yolo:
                    if mask is None:
                        mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)
                    H, W = mask.shape
                    for poly in mask_polygons(mask, v["instance_id"]):
                        xy = poly.astype(np.float64) / [W, H]
                        lines.append("0 " + " ".join(f"{c:.6f}" for c in
                                                     np.clip(xy, 0, 1).ravel()))
            if not a.no_yolo:
                link(rgb, os.path.join(ydir, "images", sp, name + ".png"))
                with open(os.path.join(ydir, "labels", sp, name + ".txt"), "w") as lf:
                    lf.write("\n".join(lines))

    with open(os.path.join(out, "splits.json"), "w") as fh:
        json.dump({"by": "scene_id", "fractions": SPLIT, "counts": counts,
                   "scenes": splits}, fh, indent=1)
    if not a.no_yolo:
        with open(os.path.join(ydir, "data.yaml"), "w") as fh:
            fh.write(f"path: {ydir}\ntrain: images/train\nval: images/val\n"
                     f"test: images/test\nnc: 1\nnames: ['crack']\n")
    print(f"{len(frames)} frames, {n_inst} crack instances, "
          f"{sum(len(m['cracks']) for m in metas.values())} unique cracks")
    for sp, c in counts.items():
        print(f"  {sp:5s}: {c['scenes']} scenes, {c['frames']} frames, "
              f"{c['instances']} instances")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
