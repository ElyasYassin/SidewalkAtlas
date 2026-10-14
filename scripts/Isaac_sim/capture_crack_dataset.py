"""
Crack detection dataset capture -- the real pipeline, single class "crack".

Runs INSIDE Isaac Sim (Script Editor). Combines everything validated in
this session:

  1. PRIM-LEVEL TAGGING (proven to work): tag crack_1-4 with class
     "crack". GeomSubset-level tagging was tested and confirmed NOT to
     render in this Isaac version (a subset tagged "crack" produced no
     "crack" ID in the segmentation output even with the parent's own
     tag removed) -- so we tag whole prims, which DOES work, and get the
     pixel-accurate boundary a different way (step 3).

  2. Replicator capture: RGB + depth + semantic segmentation per frame,
     multiple Spot-plausible viewpoints per crack prim.

  3. PER-FRAME GEOMETRIC REFINEMENT: the semantic map only tells us "this
     pixel belongs to the crack_1 prim" (the whole slab). To get the true
     crack boundary within that prim, per frame: deproject the
     semantic-gated pixels to world points using depth, then run local-
     neighborhood plane-deviation isolation (validated against crack_1's
     actual geometry: recovers from confidence=unreliable_too_few with a
     naive global plane fit -- sigma 51.7mm/threshold 207mm/0% flagged,
     due to real curvature across the ~2m patch -- to confidence=precise
     with local fitting). The refined mask, not Isaac's raw prim mask, is
     what gets saved as ground truth.

  4. Frames where too few pixels are gated, or isolation reports
     unreliable confidence, are SKIPPED rather than saved with a bad
     mask -- consistent with the "precise vs unreliable" approach used
     throughout this project.

Output layout (compatible with export_yolo_dataset.py, CLASS_NAMES=["crack"]):
  <OUT_DIR>/rgb/frame_XXXXXX.png
  <OUT_DIR>/depth/frame_XXXXXX.npy
  <OUT_DIR>/instance/frame_XXXXXX.png   (refined mask: 0=bg, 1+=connected crack regions)
  <OUT_DIR>/capture_index.json
"""

import os
import json
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy import ndimage

import omni.usd
import omni.kit.app
import omni.replicator.core as rep
from pxr import UsdGeom, Usd, Gf


# =====================================================================
# CONFIG
# =====================================================================

CRACK_PRIMS = [
    "/World/assembly_City/crack_1",
    "/World/assembly_City/crack_2",
    "/World/assembly_City/crack_3",
    "/World/assembly_City/crack_4",
]
CLASS_NAME = "crack"

CAPTURE_CAM_PATH = "/World/crack_dataset_cam"
RESOLUTION = (640, 360)
FOCAL_MM, H_APERTURE_MM, V_APERTURE_MM = 2.2, 5.0, 2.8
FX = FOCAL_MM / H_APERTURE_MM * RESOLUTION[0]
FY = FOCAL_MM / V_APERTURE_MM * RESOLUTION[1]
CX, CY = RESOLUTION[0] / 2.0, RESOLUTION[1] / 2.0

FRAMES_PER_CRACK = 150       # 4 cracks x 150 = 600 target
# RECOMMENDED: set this low (e.g. 8) for a first test run. The gates below
# were tuned against synthetic data; real Isaac depth-sensor noise may
# behave differently. Check the skip-reason counts printed at the end
# before committing to the full FRAMES_PER_CRACK=150 run.
# Biased toward steeper pitch: grazing/oblique views amplify depth-sensor
# noise into large apparent surface-height noise (confirmed empirically --
# a wide-FOV frame at shallow pitch produced scattered noise-driven false
# isolation, not a coherent crack shape). Still varied for realism, just
# not as shallow as originally planned.
PITCH_RANGE_DEG = (45.0, 80.0)
HEIGHT_RANGE_M = (0.35, 0.60)
STANDOFF_RANGE_M = (0.3, 0.8)
AZIMUTH_RANGE_DEG = (-30.0, 30.0)
RNG_SEED = 13

# Geometric refinement (validated against crack_1's real scale/curvature;
# see module docstring). Frames with fewer gated pixels than the minimum
# are skipped rather than refined unreliably on too little data.
K_NEIGHBORS = 400         # widened from 200: live testing against the REAL
                          # crack_1 mesh showed 72% 'scattered' + 25%
                          # 'unreliable' with too_few_gated=0 -- gating
                          # works fine, isolation doesn't. Cause: Meshy-
                          # generated meshes have genuine small-scale
                          # surface texture/bumpiness baked in from the AI
                          # reconstruction (the same messiness found much
                          # earlier extracting ground truth directly from
                          # these assets), which synthetic tests (smooth
                          # curvature + uniform noise) never modeled. A
                          # wider local reference averages over that
                          # texture instead of reacting to individual bumps.
K_SIGMA = 5.5             # up from 4.5: less sensitive to small texture-
                          # scale bumps, still well above real sensor noise
MIN_THRESHOLD_M = 0.001
MIN_ISOLATION_FRAC = 0.002
MAX_ISOLATION_FRAC = 0.35
MIN_GATED_PIXELS = 800
MAX_ISOLATION_POINTS = 13000  # cap density fed to isolation; a raw gate
                              # far denser than this (e.g. a crack prim
                              # filling most of the frame) is subsampled
                              # down to the density range actually
                              # validated, since a fixed K_NEIGHBORS
                              # becomes too small a physical neighborhood
                              # at very high point density (confirmed
                              # empirically: un-subsampled full-frame gate
                              # produced scattered noise, not a coherent
                              # crack -- see MIN_LARGEST_COMPONENT_FRAC).
MIN_LARGEST_COMPONENT_FRAC = 0.35  # down from 0.5: on a real (not
                                   # perfectly smooth) mesh, a genuine
                                   # crack may isolate as a dominant region
                                   # plus a few small fragments, not one
                                   # perfect blob -- still rejects true
                                   # noise-scattered results (those tested
                                   # at ~1% largest-component fraction,
                                   # far below this)
MAX_DEPTH_M = 10.0

SETTLE_UPDATES = 12
OUT_DIR = "C:/Users/Elyas/sidewalk_dataset/crack_dataset/capture"
RGB_DIR = os.path.join(OUT_DIR, "rgb")
DEPTH_DIR = os.path.join(OUT_DIR, "depth")
INST_DIR = os.path.join(OUT_DIR, "instance")


# =====================================================================
# TAGGING (proven: prim-level tags render correctly)
# =====================================================================

def tag_crack_prims():
    stage = omni.usd.get_context().get_stage()
    from isaacsim.core.utils.semantics import add_update_semantics
    tagged = 0
    for p in CRACK_PRIMS:
        prim = stage.GetPrimAtPath(p)
        if prim.IsValid():
            add_update_semantics(prim, CLASS_NAME)
            tagged += 1
        else:
            print(f"⚠ Prim not found: {p}")
    print(f"✓ Tagged {tagged}/{len(CRACK_PRIMS)} crack prims as '{CLASS_NAME}'")
    return tagged > 0


# =====================================================================
# GEOMETRIC REFINEMENT (validated: see module docstring + test history)
# =====================================================================

def isolate_defect_pixels(world_pts, k_neighbors=K_NEIGHBORS, k_sigma=K_SIGMA,
                          min_thr=MIN_THRESHOLD_M, min_frac=MIN_ISOLATION_FRAC,
                          max_frac=MAX_ISOLATION_FRAC, refit_passes=2):
    n = len(world_pts)
    k_use = min(k_neighbors + 1, n)
    tree = cKDTree(world_pts)
    _, idx = tree.query(world_pts, k=k_use)
    flag = np.zeros(n, dtype=bool)
    sigma = thr = 0.0
    for pass_i in range(refit_passes + 1):
        resid = np.zeros(n)
        for i in range(n):
            nb_idx = idx[i]
            if pass_i > 0:
                clean = nb_idx[~flag[nb_idx]]
                if len(clean) >= max(k_neighbors // 4, 10):
                    nb_idx = clean
            nb = world_pts[nb_idx]
            c = nb.mean(axis=0)
            _, _, vt = np.linalg.svd(nb - c, full_matrices=False)
            nrm = vt[-1]
            resid[i] = (world_pts[i] - c) @ nrm
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
        thr = max(k_sigma * sigma, min_thr)
        flag = np.abs(resid) > thr
    frac = float(flag.sum()) / n
    if frac > max_frac:
        confidence = "unreliable_over_fraction"
    elif frac < min_frac:
        confidence = "unreliable_too_few"
    else:
        confidence = "precise"
    return flag, {"fraction": frac, "confidence": confidence, "sigma_mm": sigma * 1000}


def deproject_world(depth, M):
    """M: camera world matrix, row-vector convention (world = p_local @ M).
    Non-finite depth (background/out-of-range pixels) is zeroed before the
    matmul -- those pixels get excluded downstream by the valid mask
    anyway, so this is purely to avoid a harmless-but-log-cluttering
    'invalid value in matmul' warning from propagating inf/nan through."""
    h, w = depth.shape
    depth_clean = np.where(np.isfinite(depth), depth, 0.0)
    vs, us = np.mgrid[0:h, 0:w].astype(np.float64)
    xn = (us - CX) / FX
    yn = (CY - vs) / FY
    pc = np.stack([xn * depth_clean, yn * depth_clean, -depth_clean,
                  np.ones_like(depth_clean)], axis=-1)
    world = (pc.reshape(-1, 4) @ M)[:, :3]
    return world.reshape(h, w, 3)


def refine_frame(depth, sem, crack_ids, cam_matrix):
    """Full per-frame refinement: gate by semantic class, deproject,
    isolate, return a labeled instance mask (0=bg, 1..N=connected crack
    regions) or None if the frame should be skipped.

    Beyond the isolation fraction gate, this also rejects SCATTERED
    results: a real crack forms one (or a few) coherent connected
    region(s); a noise-driven false isolation (confirmed empirically on
    a wide-FOV, grazing-angle frame -- depth noise at oblique pixels
    amplifies into apparent surface deviation) produces many small
    disconnected specks instead. The fraction gate alone missed this
    case (it looked like a reasonable total fraction while being
    spatially meaningless); requiring the largest component to dominate
    catches it directly."""
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < MAX_DEPTH_M)
    gate = np.isin(sem, list(crack_ids)) & valid
    n_gated = int(gate.sum())
    if n_gated < MIN_GATED_PIXELS:
        return None, {"reason": f"too few gated pixels ({n_gated})"}

    world = deproject_world(depth, cam_matrix)
    gate_idx = np.where(gate.ravel())[0]
    if len(gate_idx) > MAX_ISOLATION_POINTS:
        gate_idx = np.random.default_rng(0).choice(
            gate_idx, MAX_ISOLATION_POINTS, replace=False)
    gated_pts = world.reshape(-1, 3)[gate_idx]
    flag, info = isolate_defect_pixels(gated_pts)

    if info["confidence"] != "precise":
        return None, {"reason": f"isolation {info['confidence']} "
                                f"(fraction={info['fraction']*100:.1f}%)"}

    mask = np.zeros(depth.shape, dtype=bool)
    mask.ravel()[gate_idx] = flag
    labeled, n_comp = ndimage.label(mask)
    if n_comp == 0:
        return None, {"reason": "no connected regions after isolation"}

    sizes = ndimage.sum(labeled > 0, labeled, index=np.arange(1, n_comp + 1))
    largest_frac = sizes.max() / sizes.sum()
    if largest_frac < MIN_LARGEST_COMPONENT_FRAC:
        return None, {"reason": f"scattered result ({n_comp} regions, "
                                f"largest only {largest_frac*100:.0f}% of "
                                f"flagged pixels -- likely noise, not a "
                                f"coherent defect)"}

    return labeled.astype(np.uint16), {"reason": "ok", "n_regions": n_comp,
                                       "n_gated": n_gated,
                                       "fraction": info["fraction"]}


# =====================================================================
# CAMERA + ANNOTATORS
# =====================================================================

def world_bbox(prim_path):
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return None
    pts_all = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not pts:
            continue
        local = np.asarray(pts, dtype=np.float64)
        M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()), dtype=np.float64)
        w = (np.column_stack([local, np.ones(len(local))]) @ M)[:, :3]
        pts_all.append(w)
    if not pts_all:
        return None
    p = np.concatenate(pts_all)
    return p.min(axis=0), p.max(axis=0), p.mean(axis=0), len(p)


def look_at_matrix(eye, target, up=(0, 0, 1)):
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    fwd = target - eye
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm < 1e-6:
        # eye == target (degenerate); nudge target so callers get a valid,
        # if arbitrary, matrix instead of a NaN-propagating divide-by-zero.
        fwd = np.array([0.0, 0.0, -1.0])
        fwd_norm = 1.0
    fwd = fwd / fwd_norm
    right = np.cross(fwd, up)
    # 1e-3, not 1e-6: near-miss cases (fwd nearly parallel to up, common
    # at steep sampled pitch) can be numerically shaky well before they're
    # exactly zero -- confirmed empirically as the source of an 'invalid
    # value in matmul' warning downstream, traced to a non-finite camera
    # matrix for some sampled viewpoints.
    if np.linalg.norm(right) < 1e-3:
        up = np.array([0, 1.0, 0])
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-3:
            up = np.array([1.0, 0, 0])
            right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    R = np.column_stack([right, true_up, -fwd])
    Mo = np.eye(4)
    Mo[:3, :3] = R.T
    Mo[3, :3] = eye
    return Mo


def sample_viewpoint(centroid, rng):
    """Camera pose at a sampled pitch/height/azimuth, standoff DERIVED
    from pitch+height (standoff = height / tan(pitch)) so PITCH_RANGE_DEG
    is the actual controlled variable, not an implicit byproduct."""
    cx, cy, cz = centroid
    pitch_deg = rng.uniform(*PITCH_RANGE_DEG)
    pitch = np.radians(pitch_deg)
    height = rng.uniform(*HEIGHT_RANGE_M)
    standoff = height / max(np.tan(pitch), 1e-3)
    standoff = float(np.clip(standoff, STANDOFF_RANGE_M[0], STANDOFF_RANGE_M[1]))
    az = np.radians(rng.uniform(*AZIMUTH_RANGE_DEG))
    ex = cx - standoff * np.cos(az)
    ey = cy - standoff * np.sin(az)
    ez = cz + height
    return look_at_matrix([ex, ey, ez], [cx, cy, cz])


def ensure_camera():
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(CAPTURE_CAM_PATH).IsValid():
        cam = UsdGeom.Camera.Define(stage, CAPTURE_CAM_PATH)
        cam.GetFocalLengthAttr().Set(FOCAL_MM)
        cam.GetHorizontalApertureAttr().Set(H_APERTURE_MM)
        cam.GetVerticalApertureAttr().Set(V_APERTURE_MM)
        cam.GetClippingRangeAttr().Set((0.05, 100.0))
    return stage.GetPrimAtPath(CAPTURE_CAM_PATH)


def set_pose(prim, M):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(Gf.Matrix4d(*[float(x) for x in M.flatten()]))


def cam_matrix():
    stage = omni.usd.get_context().get_stage()
    m = UsdGeom.Xformable(stage.GetPrimAtPath(CAPTURE_CAM_PATH)) \
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m, dtype=np.float64)


_an = {}


def setup_annotators():
    rp = rep.create.render_product(CAPTURE_CAM_PATH, resolution=RESOLUTION)
    _an["rp"] = rp
    _an["rgb"] = rep.AnnotatorRegistry.get_annotator("rgb")
    _an["rgb"].attach(rp)
    _an["depth"] = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    _an["depth"].attach(rp)
    _an["semantic"] = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
    _an["semantic"].attach(rp)


async def settle(n):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


async def render_and_grab():
    try:
        await rep.orchestrator.step_async(rt_subframes=8)
    except Exception:
        pass
    exp_h, exp_w = RESOLUTION[1], RESOLUTION[0]
    for _ in range(30):
        await settle(4)
        d = _an["depth"].get_data()
        if isinstance(d, dict):
            d = d.get("data")
        depth = np.asarray(d, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        r = np.asarray(_an["rgb"].get_data())
        sem_data = _an["semantic"].get_data()
        sem = sem_data.get("data") if isinstance(sem_data, dict) else sem_data
        sem = np.asarray(sem) if sem is not None else None
        if (depth.ndim == 2 and depth.shape == (exp_h, exp_w) and r.size > 0
                and sem is not None and sem.shape == (exp_h, exp_w)):
            return depth, r, sem, sem_data.get("info", {}) if isinstance(sem_data, dict) else {}
    return None, None, None, {}


def crack_ids_from_info(info):
    ids = set()
    id_to_labels = info.get("idToLabels", {}) if isinstance(info, dict) else {}
    for id_str, label_info in id_to_labels.items():
        cls = label_info.get("class", "") if isinstance(label_info, dict) else str(label_info)
        if cls.lower() == CLASS_NAME:
            try:
                ids.add(int(id_str))
            except (ValueError, TypeError):
                pass
    return ids


# =====================================================================
# MAIN
# =====================================================================

async def run_capture():
    print(f"=== ACTIVE PARAMS: K_NEIGHBORS={K_NEIGHBORS} K_SIGMA={K_SIGMA} "
          f"MIN_LARGEST_COMPONENT_FRAC={MIN_LARGEST_COMPONENT_FRAC} "
          f"MIN_GATED_PIXELS={MIN_GATED_PIXELS} "
          f"PITCH_RANGE_DEG={PITCH_RANGE_DEG} RNG_SEED={RNG_SEED} ===")
    for d in [OUT_DIR, RGB_DIR, DEPTH_DIR, INST_DIR]:
        os.makedirs(d, exist_ok=True)

    if not tag_crack_prims():
        print("✗ No crack prims tagged; aborting.")
        return

    centroids = {}
    for p in CRACK_PRIMS:
        bbox = world_bbox(p)
        if bbox is None:
            print(f"⚠ Could not find geometry for {p}, skipping.")
            continue
        _, _, ctr, n = bbox
        centroids[p] = ctr
        print(f"  {p}: {n} verts, center ({ctr[0]:.2f},{ctr[1]:.2f},{ctr[2]:.2f})")

    if not centroids:
        print("✗ No crack geometry found; aborting.")
        return

    rng = np.random.default_rng(RNG_SEED)
    cam = ensure_camera()
    setup_annotators()
    await settle(SETTLE_UPDATES)

    index = []
    frame_idx = 0
    skipped = {"too_few_gated": 0, "unreliable": 0, "scattered": 0,
              "no_regions": 0, "no_frame": 0}

    for prim_path, centroid in centroids.items():
        n_saved = 0
        n_attempted = 0
        print(f"  starting {prim_path} ...")
        while n_saved < FRAMES_PER_CRACK and n_attempted < FRAMES_PER_CRACK * 3:
            n_attempted += 1
            if n_attempted % 10 == 0:
                print(f"    ...{prim_path}: attempt {n_attempted}, "
                     f"{n_saved} saved so far  |  skipped so far: {skipped}")
            M = sample_viewpoint(centroid, rng)
            set_pose(cam, M)
            depth, rgb, sem, sem_info = await render_and_grab()
            if depth is None:
                skipped["no_frame"] += 1
                continue

            crack_ids = crack_ids_from_info(sem_info)
            if not crack_ids:
                skipped["no_frame"] += 1
                continue

            Mw = cam_matrix()
            if not np.all(np.isfinite(Mw)):
                # Camera transform came back with non-finite entries --
                # confirmed cause of the 'invalid value in matmul'
                # warning: a degenerate look_at basis for some sampled
                # pitch/azimuth combo, not a depth-data issue (depth is
                # already sanitized in deproject_world). Deprojecting
                # with a bad matrix would corrupt the WHOLE frame's world
                # points, not just a few pixels, so skip outright rather
                # than let it silently feed bad data into isolation.
                skipped["no_frame"] += 1
                continue
            mask, refine_info = refine_frame(depth, sem, crack_ids, Mw)
            if mask is None:
                reason = refine_info["reason"]
                if "too few" in reason:
                    skipped["too_few_gated"] += 1
                elif "scattered" in reason:
                    skipped["scattered"] += 1
                elif "unreliable" in reason:
                    skipped["unreliable"] += 1
                else:
                    skipped["no_regions"] += 1
                continue

            fname = f"frame_{frame_idx:06d}"
            rgb_u8 = rgb[..., :3].astype(np.uint8) if rgb.ndim == 3 else rgb
            Image.fromarray(rgb_u8).save(os.path.join(RGB_DIR, fname + ".png"))
            np.save(os.path.join(DEPTH_DIR, fname + ".npy"), depth)
            Image.fromarray(mask).save(os.path.join(INST_DIR, fname + ".png"))

            index.append({"file": fname, "defect": os.path.basename(prim_path),
                         "class": CLASS_NAME, "n_regions": refine_info["n_regions"],
                         "n_gated": refine_info["n_gated"]})
            frame_idx += 1
            n_saved += 1

        print(f"  {prim_path}: {n_saved}/{FRAMES_PER_CRACK} frames saved "
              f"({n_attempted} attempted)")

    with open(os.path.join(OUT_DIR, "capture_index.json"), "w") as fh:
        json.dump({"num_frames": frame_idx, "resolution": list(RESOLUTION),
                   "frames": index}, fh, indent=2)

    print(f"\n✓ Captured {frame_idx} frames -> {OUT_DIR}")
    print(f"  skipped: {skipped}")
    print("  Next: run export_yolo_dataset.py (CLASS_NAMES=['crack']) to "
          "convert to YOLO-seg format.")


import asyncio
try:
    if "_crack_capture_task" in globals() and _crack_capture_task and not _crack_capture_task.done():
        _crack_capture_task.cancel()
except Exception:
    pass
_crack_capture_task = asyncio.ensure_future(run_capture())
print("Crack dataset capture scheduled.")