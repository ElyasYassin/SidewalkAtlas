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
python scripts/crack_data/fetch_textures.py                    # CC0 ambientCG -> D:/textures/acg
python scripts/crack_data/generate_crack_scenes.py --n 500     # ~10 s/scene/worker, ~2.5 GB RAM/worker
C:\isaacsim\python.bat scripts/crack_data/render_crack_dataset.py --views 16   # resumable
python scripts/crack_data/build_crack_dataset.py               # CSVs, splits, YOLO-seg

# or all three, detached and resumable:
powershell -ExecutionPolicy Bypass -File scripts/crack_data/run_full.ps1

# checks
python scripts/crack_data/crack_scene.py                       # self-test
C:\isaacsim\python.bat scripts/crack_data/render_crack_dataset.py --validate --count 3 \
    --scenes <dir> --out <dir>                                 # mask vs renderer
python scripts/crack_data/contact_sheet.py --render <dir>      # eyeball masks
```

Output defaults to `D:/sidewalk_crack_synth/` (C: is nearly full).

## Scene content

**Walking surface** — broom concrete with photo detail (50%), exposed
aggregate (12%), asphalt (16%), pavers / brick (22%, never cracked: a joint
grid is a hard negative). Photo textures are CC0 ambientCG sets, each checked
by eye for cracks of its own (Asphalt026C was excluded for exactly that).

**Cracks** — on concrete, aggregate and asphalt. Patterns transverse,
longitudinal, diagonal, branching, map (alligator on asphalt). Width classes
hairline 0.8–2 mm / fine 2–5 / medium 5–12 / wide 12–30. Open depth up to
30 mm; 40% partly dirt-filled, and the label is the open depth. Edge
roughness varies along the crack (chips), with a worn bevelled lip.

**Look-alikes, unlabelled** — tooled joints, driveway apron joints, sealant
lines (repaired cracks), asphalt patches, locate paint, gum spots, stains,
leaves. **Covers, which hide cracks under them** — manholes, utility lids,
drain grates, tree pits (with trees), tactile pads.

**Street** — curb + gutter + road (75%) or a verge, optional parkway strip,
frontage (lawn, wall, building, fence, hedge), poles, trees, hydrants, bins,
signs. The sidewalk continues past the labelled patch.

**Lighting and camera** — sun + 60 sky HDRIs, overcast, wet surfaces,
per-scene auto-exposure. Spot-like camera: 0.45–0.75 m high, 30–65° down,
HFOV 70–95°.

## Labels

- `mask/` uint16: crack instance id, 0 = none, **65535 = ignore** — crack
  pixels narrower than 0.4 px at their range, or in clipped highlight /
  crushed shadow. Labelled but not visible: train neither way on them.
- Masks come from depth: each pixel's depth gives the surface point it
  shows; the texture is planar-UV mapped, so that point indexes the label
  raster exactly where the pixel's colour came from. `--validate` renders the
  label raster as an emissive texture: IoU 0.94–0.98 on cracks ≥2 px wide;
  lower only for sub-pixel cracks (texture filtering), which is the ignore
  regime.
- Measurements are the generator's parameters, sampled every 0.5 mm:
  exact. Columns match `crack_detection`'s `visual_review.csv`.
- Per frame visibility: `visible_pixels`, `ignore_pixels`,
  `centerline_visible_fraction`, `width_px_at_center`. Train measurement
  only on mostly-visible, resolvable instances.

## Depth

- `depth/` uint16 exact z-depth in 0.1 mm (float16 was too coarse for crack
  depths: 1–2 mm steps at 1–4 m).
- `depth_stereo/` uint16, half resolution, Spot-like stereo: noise σ = k·z²
  (k 0.004–0.012 /m), matching-window blur that erases narrow cracks, edge
  dropouts, holes. Train depth-using models on this one; a real stereo pair
  never sees the exact map.

## Render settings

Isaac's default anti-aliasing is DLSS *Performance*, which renders at about
half resolution and upscales — it smeared texture and erased hairline
cracks. The renderer sets DLAA (`/rtx/post/aa/op = 4`, native resolution):
~40% sharper, validation IoU 0.75 → 0.86 mean.

## Splits

By scene (`sha1(scene_id)`), never by frame: 16 views of one slab are not
independent samples. Adding scenes never moves existing ones between splits.

## Known limits

- Single image cannot observe the depth of a narrow crack; expect width and
  length to be learnable, depth weakly (stereo helps only for wide cracks).
- Pavers carry no cracks; cracked pavers are a separate defect.
- Props are primitive shapes (cylinders, boxes); they serve as occluders and
  shadow casters, not as realistic objects.
