# `scripts/Isaac_sim` — script guide

Everything in this folder belongs to one pipeline: drive Spot down a synthetic
sidewalk in Isaac Sim, measure how uneven that sidewalk actually is, and turn
the captured frames into a trainable defect-detection dataset.

The scripts split into two execution environments, and mixing them up is the
most common way to waste time:

| Environment | How it runs | Scripts |
|---|---|---|
| **Inside Isaac Sim** (Script Editor) | Paste and execute; needs `omni.*`, `pxr`, an open USD stage | `walking.py`, `spot_walk_with_evenness_autosave.py`, `capture_crack_dataset.py`, `crack_detection/*` |
| **Offline** (plain Python) | `python <script>.py`; needs only numpy / cv2 / PIL / matplotlib / torch | `export_yolo_dataset.py`, `depth_from_image.py`, `depth_video.py`, `view_dataset.py`, `view_defects.py` |

Nearly every script has its paths hardcoded in a `CONFIG` block at the top.
Datasets are written outside the repo, under `C:/Users/Elyas/sidewalk_dataset/`.

---

## Pipeline at a glance

```
 Isaac Sim scene (/World/assembly_City + /World/spot_03)
   |
   +- crack_detection/validate_geometry.py   which defects have real geometry?
   |     +- crack_detection/measure_defect.py        (the measurement engine)
   |          +- crack_detection/batch_measure_defects.py -> ground_truth/*.csv|json
   |
   +- walking.py                             autonomous camera-driven walk
   |     +- spot_walk_with_evenness_autosave.py   walk + evenness measurement
   |          -> walk_captures/run_*/  (rgb, depth_npy, depth_vis, evenness CSV+JSON)
   |
   +- capture_crack_dataset.py               posed crack captures + refined masks
         -> crack_dataset/capture/  (rgb, depth, instance, capture_index.json)
              |
              +- export_yolo_dataset.py  (offline) -> crack_dataset/yolo/ + data.yaml

 Offline inspection: view_dataset.py | view_defects.py | depth_video.py | depth_from_image.py
```

---

## Navigation / walking

### `walking.py` (1616 lines) — base autonomous sidewalk walker

Runs in the Script Editor. Ends by scheduling `run_demo()` as an asyncio task,
with a guard at the top that cancels a previously scheduled task so re-running
the script is safe.

What it does:

- **Semantic tagging.** `tag_walkables()` stamps the class `"sidewalk"` onto a
  hardcoded prim list — the sidewalk mesh plus every defect patch
  (`crack_1..4`, `pothole_1..2`, `uneven_1..7`, `collapsed_1..4`). Curbs and
  everything else stay untagged and read as "not sidewalk".
- **Perception.** A Replicator render product on
  `/World/spot_03/body/spot_camera` at 640x360 feeds three annotators: `rgb`,
  `semantic_segmentation`, and `distance_to_image_plane`.
- **Steering.** The binary walkable mask is split into a **near band** (bottom
  third) and a **lookahead band** (middle). Contiguous walkable column
  intervals in the near band are scored by `|center offset| - 0.25*width`, and
  the winner's center is blended 50/50 with the lookahead centroid — so Spot
  follows where the sidewalk *goes*, not just what is under its feet. The
  resulting normalized error drives a **PD controller** (`KP=0.60`, `KD=0.04`,
  `KI=0`, dead band 0.04, output clamped to +/-0.55) with a filtered
  derivative.
- **Turn sign.** `INVERT_TURN_COMMAND = True` — the Spot policy's yaw
  convention is the opposite of the image-space convention, so the final
  command is negated. Don't "fix" this without re-testing.
- **Speed schedule.** Keyed on *centerline* error, not the weave target:
  1.0 m/s when well-aligned on a wide walk, stepping down through 0.6 / 0.4 /
  0.2; capped at 0.4 m/s during hard turns.
- **Coverage weave** (`COVERAGE_SWEEP_ENABLED`). Rather than hugging the
  centerline, it aims 70% of the way toward one edge and flips sides when that
  edge's near-band pixels fall below 15% of the near total — zig-zagging so the
  full sidewalk width gets covered. A 5 cm world grid records a 1.6 x 1.6 m
  swath as it moves, drawn live as a top-down map and saved to
  `walk_debug/coverage_map.png`.
- **Recovery state machine.** When the mask empties or the near band thins:
  `backup` (-0.10 m/s, 350 steps) -> `scan` (back up while turning at 0.45
  toward the last known sidewalk side, 500 steps). Reacquires at >=5000
  near-band pixels with >1200 in the lower center. After 4 failed scan cycles
  it flips the assumed direction and does a double-length backup.
- **Capture and live views.** RGB PNG, raw depth `.npy`, and colormapped depth
  PNG saved periodically into a timestamped `run_YYYYmmdd_HHMMSS/` folder. Four
  `omni.ui` windows update at ~10 Hz: RGB, mask overlay, binary
  sidewalk/not-sidewalk, and the coverage map.

Run parameters: `PHYSICS_RATE = 500` (**must** match the trained policy's
`sim.dt = 0.002` x `decimation = 10`; do not lower it to 200), rendering at
1/10 that rate, 500 settle steps standing still, then 100,000 walk steps =
200 simulated seconds. At the end it saves the coverage map and runs
`check_spot_orientation()` to report whether Spot is still upright.

### `spot_walk_with_evenness_autosave.py` (1951 lines) — walker + evenness measurement

A superset of `walking.py`. Everything above still applies; this variant adds
the research payload and crash-resistant saving.

- **Evenness sampling** (`sample_sidewalk_evenness`). Every 5 physics steps
  (100 Hz), a ray is cast straight down from 1.5 m above Spot's body through
  the PhysX scene query. `raycast_closest_filtered()` walks the ray *past* any
  hit on Spot itself, so what gets sampled is the real collision surface the
  feet encounter — not the visual mesh.
- **Detrended relief.** Over a rolling 1 m distance window, the local grade is
  fitted and removed (`np.polyfit`, degree 1), then peak-to-peak, RMS and
  max-abs residual are computed in millimetres.
- **Classification.** <=3 mm `smooth`, <=10 mm `mild`, <=25 mm `uneven`, above
  that `severe`. A status line prints whenever the class changes, and once per
  simulated second regardless.
- **Secondary signal.** Body vertical acceleration, derived by
  double-differencing body Z, is logged as a traversability proxy.
- **Autosave — the reason this variant exists.** Every 500 samples (~5
  simulated seconds) the full CSV is rewritten. A Script Editor task that gets
  interrupted or throws still leaves usable data on disk.

Outputs, alongside the RGB/depth run folder:

- `sidewalk_evenness_timeseries.csv` — one row per sample: step, time, path
  distance, body XYZ, ground XYZ, clearance, vertical accel, collider path.
- `sidewalk_evenness_summary.json` — run-level metrics plus classifications
  recomputed over 1 m windows at ~10 cm spacing.

**Which one to run:** use the autosave variant for anything you intend to keep.
`walking.py` is the navigation-only baseline.

---

## Defect ground truth — `crack_detection/`

### `measure_defect.py` (421 lines) — the measurement engine

Imported by the other two; not usually run directly. Measures a defect from the
actual PhysX collision surface:

1. Take the prim's world-space bounding box (padded).
2. Cast a dense top-down ray grid over it (default 151x151 from 1.5 m up).
3. Fit the intact sidewalk plane using only samples near the grid border,
   robustly (3-sigma outlier rejection).
4. Compute signed displacement from that reference plane.
5. Segment the defect by height/depth threshold, drop small components,
   optionally keep only the largest.
6. Report length and width from a **PCA of the segmented footprint** (major /
   minor horizontal extent), plus max depth, max height, vertical range, and
   projected area.

Requires PhysX collision on the mesh and a Z-up scene. Defects that are only
textures or decals have no geometric depth and cannot be measured — that is
exactly what the next script checks. Results serialize via
`save_results_json()` / `save_results_csv()`.

### `validate_geometry.py` (60 lines) — run this first

Sweeps all 16 candidate prims with a coarse config (51x51 grid, 3 mm
threshold) and prints a pass/fail line per prim with depth, height, LxW and ray
count. Use it to find which candidates have real geometry before spending time
on a full measurement pass.

### `batch_measure_defects.py` (108 lines) — production measurement pass

Loads `measure_defect.py` by file path via `importlib` (the Script Editor has
no package context), then measures a curated `SELECTED_DEFECTS` list with
**per-class configs** — cracks get a finer grid (201x201) and a tighter 4 mm
threshold; `uneven` uses 10 mm; `collapsed` is coarser still. Writes CSV and
JSON into `ground_truth/`.

> **Path caveat:** both scripts point `SCRIPT_DIR` at
> `C:\Users\Elyas\sidewalk_dataset\scripts`, not at this repo folder. Either
> copy `measure_defect.py` there or update `SCRIPT_DIR` before running.

---

## Detection dataset

### `capture_crack_dataset.py` (536 lines) — posed crack capture, Isaac Sim

Builds a single-class (`"crack"`) detection dataset. The header docstring
records three findings worth keeping:

1. **Prim-level tagging works; GeomSubset-level tagging does not** in this
   Isaac version — a subset tagged `"crack"` produced no `crack` ID in the
   segmentation output. So whole prims are tagged, and the true boundary is
   recovered geometrically instead.
2. **Per-frame geometric refinement.** The semantic map only says "this pixel
   is on the crack_1 prim" (the whole slab). Per frame, the gated pixels are
   deprojected to world points using depth, then a **local-neighborhood
   plane-deviation** test isolates the actual crack. A naive *global* plane fit
   fails on these ~2 m patches because of genuine curvature (sigma 51.7 mm,
   threshold 207 mm, 0% flagged); local fitting recovers precise confidence.
3. **Quality gates, not best-effort labels.** Frames with too few gated pixels,
   unreliable isolation, or a scattered (non-coherent) result are *skipped*
   rather than saved with a bad mask. Skip-reason counts print at the end.

Viewpoints are sampled Spot-plausibly and biased toward steeper pitch
(45-80 deg) because grazing views amplify depth-sensor noise into false
isolation: height 0.35-0.60 m, standoff 0.3-0.8 m, azimuth +/-30 deg, seeded
RNG. Tuning constants (`K_NEIGHBORS=400`, `K_SIGMA=5.5`,
`MIN_LARGEST_COMPONENT_FRAC=0.35`) carry inline comments explaining why each
was moved off its original value — the Meshy-generated meshes have real
small-scale surface texture that synthetic tests never modeled.

Target is 150 frames x 4 cracks = 600. **Set `FRAMES_PER_CRACK` low (~8) for a
first run** and check the skip counts before committing to a full pass.

Output: `crack_dataset/capture/{rgb,depth,instance}/` plus
`capture_index.json`.

### `export_yolo_dataset.py` (276 lines) — offline, converts to YOLO-seg

Reads the capture folder and writes YOLO segmentation labels: one `.txt` per
image, one line per instance, `<class_id> x1 y1 x2 y2 ...` normalized to 0-1.

- Uses the **instance** masks, not semantic — semantic gives one ID per class
  and cannot separate two cracks in the same frame.
- Contours via `cv2.findContours` -> `approxPolyDP` with epsilon at 0.5% of
  perimeter; contours under 15 px^2 are dropped as rendering noise.
- Seeded 85/15 train/val split, writes `data.yaml`.
- Renders 20 **mask-overlay sanity images** so a bad labeling assumption gets
  caught before a training run rather than after one.

Run it with `python export_yolo_dataset.py`.

---

## Offline inspection tools

### `view_dataset.py` (92 lines)
Renders a 2x3 preview grid per frame for the older multi-channel dataset
layout: RGB, clean depth, sensor depth, surface normals, semantic, and a
clean-vs-sensor difference map in millimetres (the noise pattern). Writes the
first 10 frames into `previews/`.

### `view_defects.py` (61 lines)
Single-frame defect view. Subtracts a **locally smoothed** depth surface
(80 px uniform filter, invalid-pixel weighted) instead of a global plane, so
multi-slab scenes don't get skewed by an overall tilt. Shows local deviation at
+/-5 mm and +/-15 mm plus a depth-gradient map that highlights slab edges and
cracks, and prints std / min / max deviation. Point `RUN` at the run folder.

### `depth_video.py` (149 lines)
Plays a walk run's `depth_npy/` folder as a depth-gradient video (hot colormap,
gradient clamped at 20 mm/px), with a frame counter overlay, and optionally
writes `depth_gradient_video.mp4`. `q` quits. Point `RUN_DIR` at the run
folder.

### `depth_from_image.py` (160 lines)
Monocular depth on a single real photo using
`depth-anything/Depth-Anything-V2-Small-hf` via the HF `pipeline`, on CUDA when
available. Relative depth is rescaled into a **fake** 1-5 m range purely so the
existing gradient-view code applies — **this is not metric depth**. Saves the
`.npy`, a turbo depth visualization, and a hot gradient visualization, then
opens three OpenCV windows. This is the bridge for applying the sim-developed
gradient analysis to real-world sidewalk images.

---

## Gotchas

- `PHYSICS_RATE = 500` in both walkers is fixed by the trained policy
  (`sim.dt = 0.002` x `decimation = 10`). Changing it breaks the gait.
- `INVERT_TURN_COMMAND = True` compensates for the policy's yaw sign — it is
  not a bug.
- All output paths are absolute and machine-specific
  (`C:/Users/Elyas/sidewalk_dataset/...`). Edit the `CONFIG` block per machine.
- The `SIDEWALK` / `CRACK_PRIMS` / `SELECTED_DEFECTS` lists are hardcoded to
  `/World/assembly_City/...`. A renamed or re-authored scene needs these
  updated; the walkers print a warning per prim they cannot find.
- `crack_detection/` scripts expect `measure_defect.py` at `SCRIPT_DIR`, which
  currently points outside this repo.
