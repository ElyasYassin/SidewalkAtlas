# Synthetic crack dataset

Isaac Sim as a ground-truth factory: images of sidewalk cracks with exact
pixel masks and exact measurements (length, width, depth), for training a
model that detects cracks **and** measures them.

## Why not the Meshy assets

In the Meshy city scene the concrete texture shows many cracks that have no
geometry, so the raycast labeller never sees them. A model trained on that
learns visible cracks are background. The labelled cracks are measured on
Meshy's lumpy geometry, so widths and depths do not describe the crack the
camera sees. Here one crack definition (centreline, width profile, depth
profile) produces the geometry, the texture and the labels, so what is
visible is labelled and what is labelled is visible.

The Meshy scenes remain useful as an **out-of-distribution test set**.

## Pipeline

```bash
# 1. scenes (offline, ~1 s/scene/worker)
python scripts/crack_data/generate_crack_scenes.py --n 300

# 2. render (headless Isaac, ~1.5-2 s/frame on an RTX 3090; resumable)
C:\isaacsim\python.bat scripts/crack_data/render_crack_dataset.py --views 16

# 3. dataset tables, scene-level splits, YOLO-seg export (offline)
python scripts/crack_data/build_crack_dataset.py

# checks
python scripts/crack_data/crack_scene.py                       # self-test
C:\isaacsim\python.bat scripts/crack_data/render_crack_dataset.py --validate --count 2 \
    --scenes <dir> --out <dir>                                 # mask vs renderer
python scripts/crack_data/contact_sheet.py --render <dir>      # eyeball masks
```

Output defaults to `D:/sidewalk_crack_synth/` (C: is nearly full).

## What is randomised

| | |
|---|---|
| Crack | pattern (transverse, longitudinal, diagonal, branching, map), width class hairline 0.8–2 mm / fine 2–5 / medium 5–12 / wide 12–30, depth 1–3x width-scaled, tortuosity, mm-scale jaggedness, tapering ends, dirt or weed fill |
| Concrete | gray level, tint, broom finish, aggregate speckle, stains, gum spots, leaves (which hide cracks, and the label follows), slab-to-slab variation, tooled joints (unlabelled: a joint is not a crack) |
| Lighting | sun elevation/azimuth/temperature/softness, overcast, 60 sky HDRIs, shadow-casting occluders, exposure (auto-exposure-like, per scene) |
| Surface | wet (low roughness, darker) 15% |
| Camera | Spot-like: height 0.45–0.75 m, pitch 30–65° down, roll ±3°, HFOV 70–95°, yaw ±30° along the walk, 20% looking back |
| Surround | asphalt / road textures, grass, soil, gravel |

## Labels

- **Masks** come from depth: each pixel's depth gives the surface point it
  shows; the texture is planar-UV mapped, so that point's (x, y) indexes the
  label raster at exactly the spot whose colour the pixel shows. Occlusion and
  perspective are therefore free. `--validate` renders the label raster as an
  emissive texture and compares: same pixel count (no size bias), 1 px
  boundary agreement ~0.94; strict IoU ~0.75 is the renderer's own edge
  anti-aliasing on 2–5 px-wide structures.
- **Measurements** are the generator's parameters, sampled every 0.5 mm along
  the centreline: exact by construction. Columns match
  `crack_detection`'s `visual_review.csv`.
- **Visibility** per frame: `visible_pixels` and
  `centerline_visible_fraction` (share of the centreline in frame and
  unoccluded). Train measurement only on mostly-visible instances.

## Splits

By scene (`sha1(scene_id)`), never by frame: 16 views of one slab are not
independent samples. Adding scenes never moves existing ones between splits.

## Known limits

- Concrete is procedural; no photographed concrete textures are available
  locally. Real-photo validation is the open question.
- Measurement labels are 3-D truth; from one image, depth of a narrow crack
  is not observable. Expect width and length to be learnable, depth weakly.
- Hairline cracks under ~1.5 mm are below one pixel beyond ~1 m range; they
  are labelled where they render, but the image may show them only faintly.
