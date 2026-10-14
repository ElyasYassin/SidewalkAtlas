"""
Synthetic cracked-sidewalk scene: geometry, matching texture, exact labels.

Pure numpy/scipy/cv2 -- no Isaac. One crack definition (centreline, width
profile, depth profile) drives all three outputs, so the picture and the
labels cannot disagree:

  * the 1 mm height raster (mesh + normal map) -- what the geometry is
  * the 1 mm albedo / roughness rasters         -- what the camera sees
  * the 1 mm instance-label raster + per-crack measurements -- the truth

This fixes the Meshy-asset problem, where cracks painted into the texture had
no geometry and so were never labelled, and labelled geometry did not match
the visible crack.

Raster convention: arrays are indexed [iy, ix]; cell centre is
    x = x0 + (ix + 0.5) * RES_M,   y = y0 + (iy + 0.5) * RES_M.
Texture images are written with rows flipped (image row 0 = max y), which is
what a UV mapping u = (x - x0) / L, v = (y - y0) / W expects.

Measurement fields match crack_detection's visual_review.csv so both sources
can be compared directly. Width and depth come from the generative profile,
sampled every 0.5 mm along the centreline: they are exact by construction,
not measured.

Run directly for a self-test.
"""

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy import ndimage


RES_M = 0.001          # texture / label / fine-height raster
MESH_CELL_M = 0.005    # render mesh cell; finer detail goes in the normal map
SAMPLE_STEP_M = 0.0005  # centreline resampling for rasterisation

# Total crack width classes (m) and sampling weights.
WIDTH_CLASSES = {
    "hairline": ((0.0008, 0.002), 0.25),
    "fine": ((0.002, 0.005), 0.35),
    "medium": ((0.005, 0.012), 0.28),
    "wide": ((0.012, 0.030), 0.12),
}
PATTERNS = {
    "transverse": 0.28,
    "longitudinal": 0.20,
    "diagonal": 0.20,
    "branching": 0.22,
    "map": 0.10,
}


# =====================================================================
# NOISE
# =====================================================================

def fbm(shape, rng, base_cell_px, octaves=5, persistence=0.5, aniso=(1.0, 1.0)):
    """Fractal value noise, zero mean, roughly unit std. `aniso` stretches the
    feature size per axis (y, x) -- (1, 0.05) makes streaks running along y."""
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, total, cell = 1.0, 0.0, float(base_cell_px)
    for _ in range(octaves):
        gh = max(2, int(math.ceil(h / max(cell * aniso[0], 1.0))) + 1)
        gw = max(2, int(math.ceil(w / max(cell * aniso[1], 1.0))) + 1)
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= persistence
        cell /= 2.0
        if cell < 1.0:
            break
    out /= total
    s = out.std()
    return out / s if s > 0 else out


def smooth_noise_1d(n, rng, corr_samples):
    """1-D smooth noise of length n, unit std, correlation ~corr_samples."""
    k = max(2, int(n / max(corr_samples, 1)) + 2)
    g = rng.standard_normal(k)
    xs = np.linspace(0, k - 1, n)
    v = np.interp(xs, np.arange(k), g)
    v = ndimage.gaussian_filter1d(v, max(corr_samples / 4.0, 1.0))
    s = v.std()
    return v / s if s > 0 else v


# =====================================================================
# PARAMETERS
# =====================================================================

@dataclass
class SceneParams:
    seed: int
    length_m: float          # along walk (x)
    width_m: float           # across walk (y)
    joint_spacing_m: float
    longitudinal_joint: bool
    n_cracks: int
    base_gray: float
    tint: Tuple[float, float, float]
    broom_amp: float
    stain_amp: float
    n_gum: int
    n_leaves: int
    micro_amp_m: float
    roughness_base: float
    surround: str            # asphalt | grass | soil | gravel


def sample_scene_params(seed: int) -> SceneParams:
    rng = np.random.default_rng(seed)
    tints = [(1.0, 1.0, 1.0), (1.04, 1.0, 0.93), (0.96, 0.99, 1.04),
             (1.05, 1.02, 0.88), (1.02, 0.98, 0.95)]
    width = float(rng.uniform(1.2, 2.0))
    return SceneParams(
        seed=seed,
        length_m=3.5,
        width_m=round(width, 3),
        joint_spacing_m=float(rng.uniform(1.1, 1.8)),
        longitudinal_joint=bool(width > 1.6 and rng.random() < 0.5),
        n_cracks=0 if rng.random() < 0.08 else int(rng.integers(1, 7)),
        base_gray=float(rng.uniform(0.40, 0.74)),
        tint=tuple(tints[int(rng.integers(len(tints)))]),
        broom_amp=float(rng.uniform(0.0, 0.05)),
        stain_amp=float(rng.uniform(0.0, 0.22)),
        n_gum=int(rng.integers(0, 30)),
        n_leaves=int(rng.integers(0, 16)) if rng.random() < 0.5 else 0,
        micro_amp_m=float(rng.uniform(0.0001, 0.0005)),
        roughness_base=float(rng.uniform(0.55, 0.9)),
        surround=str(rng.choice(["asphalt", "grass", "soil", "gravel"])),
    )


# =====================================================================
# CRACK GEOMETRY
# =====================================================================

@dataclass
class CrackPath:
    pts: np.ndarray     # (N, 2) metres
    width: np.ndarray   # (N,) total width, m
    depth: np.ndarray   # (N,) max depth of cross-section, m


def _walk(rng, start, heading, length, bounds, tort, step=0.004, kink_p=0.01):
    """Mean-reverting heading random walk from `start`; stops at `length` or
    when leaving `bounds` (x0, x1, y0, y1). Returns (N, 2) points."""
    x0, x1, y0, y1 = bounds
    pts = [np.asarray(start, float)]
    h0 = h = heading
    n = int(length / step)
    for _ in range(n):
        h += -0.08 * (h - h0) + tort * rng.standard_normal()
        if rng.random() < kink_p:
            h += rng.choice([-1, 1]) * rng.uniform(0.2, 0.6)
        p = pts[-1] + step * np.array([math.cos(h), math.sin(h)])
        if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
            break
        pts.append(p)
    return np.array(pts)


def _profile(rng, pts, w0, depth_ratio, free_start, free_end):
    """Width/depth along a resampled polyline, tapering at free ends."""
    n = len(pts)
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    L = seg[-1] if n > 1 else 0.0
    mod = np.exp(0.25 * smooth_noise_1d(n, rng, corr_samples=max(n // 6, 4)))
    taper = np.ones(n)
    tl = max(min(rng.uniform(0.03, 0.08), L / 3), 1e-6)
    if free_start:
        taper = np.minimum(taper, 0.25 + 0.75 * np.clip(seg / tl, 0, 1))
    if free_end:
        taper = np.minimum(taper, 0.25 + 0.75 * np.clip((L - seg) / tl, 0, 1))
    width = np.maximum(w0 * mod * taper, 0.0006)
    dmod = np.exp(0.2 * smooth_noise_1d(n, rng, corr_samples=max(n // 5, 4)))
    depth = np.clip(depth_ratio * 0.6 * width ** 0.85 * dmod,
                    0.0008, 0.045)
    return width, depth


def _resample(pts, step=SAMPLE_STEP_M):
    if len(pts) < 2:
        return pts
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.arange(0.0, seg[-1], step)
    return np.c_[np.interp(s, seg, pts[:, 0]), np.interp(s, seg, pts[:, 1])]


def _jitter(rng, pts, amp_m):
    """Lateral mm-scale wander: real cracks follow aggregate boundaries, so
    the centreline is jagged at a few mm even when the overall run is
    straight. Two scales of smooth noise, applied along the local normal."""
    n = len(pts)
    if n < 8:
        return pts
    t = np.gradient(pts, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    nrm = np.c_[-t[:, 1], t[:, 0]]
    off = amp_m * (0.7 * smooth_noise_1d(n, rng, int(rng.uniform(8, 24)))
                   + 0.3 * smooth_noise_1d(n, rng, int(rng.uniform(3, 6))))
    off[:4] *= np.linspace(0, 1, 4)
    off[-4:] *= np.linspace(1, 0, 4)
    return pts + nrm * off[:, None]


def make_crack(rng, slab, pattern):
    """Build one crack instance on `slab` (x0, x1, y0, y1).
    Returns (list[CrackPath], meta dict)."""
    x0, x1, y0, y1 = slab
    sw, sl = y1 - y0, x1 - x0
    m = 0.01
    inner = (x0 + m, x1 - m, y0 + m, y1 - m)
    wclass = rng.choice(list(WIDTH_CLASSES),
                        p=[v[1] for v in WIDTH_CLASSES.values()])
    if pattern == "map" and wclass in ("medium", "wide"):
        wclass = rng.choice(["hairline", "fine"])   # map cracking is fine
    w0 = float(rng.uniform(*WIDTH_CLASSES[wclass][0]))
    depth_ratio = float(rng.uniform(1.0, 3.0))
    tort = float(rng.uniform(0.03, 0.14))
    polylines = []   # (pts, free_start, free_end, width_scale)

    if pattern == "transverse":
        full = rng.random() < 0.5
        side = rng.choice([-1, 1])
        sx = rng.uniform(x0 + 0.1 * sl, x1 - 0.1 * sl)
        sy = y0 + m if side > 0 else y1 - m
        L = sw * (1.2 if full else rng.uniform(0.2, 0.85))
        pts = _walk(rng, (sx, sy), side * math.pi / 2 + rng.normal(0, 0.15),
                    L, inner, tort)
        polylines.append((pts, False, not full, 1.0))
    elif pattern == "longitudinal":
        sx = x0 + m if rng.random() < 0.6 else rng.uniform(x0, x1 - 0.3 * sl)
        sy = rng.uniform(y0 + 0.15 * sw, y1 - 0.15 * sw)
        L = sl * rng.uniform(0.3, 1.0)
        pts = _walk(rng, (sx, sy), rng.normal(0, 0.12), L, inner, tort)
        polylines.append((pts, sx > x0 + m, True, 1.0))
    elif pattern in ("diagonal", "branching"):
        sx = rng.uniform(x0 + 0.1 * sl, x1 - 0.1 * sl)
        sy = rng.uniform(y0 + 0.1 * sw, y1 - 0.1 * sw)
        L = float(rng.uniform(0.25, 1.1))
        pts = _walk(rng, (sx, sy), rng.uniform(0, 2 * math.pi), L, inner, tort)
        polylines.append((pts, True, True, 1.0))
        if pattern == "branching" and len(pts) > 20:
            main_h = math.atan2(pts[-1, 1] - pts[0, 1], pts[-1, 0] - pts[0, 0])
            for _ in range(int(rng.integers(1, 4))):
                k = int(rng.integers(len(pts) // 5, len(pts) - 1))
                bh = main_h + rng.choice([-1, 1]) * rng.uniform(0.4, 1.2)
                bl = (len(pts) * 0.004) * rng.uniform(0.2, 0.6)
                bp = _walk(rng, pts[k], bh, bl, inner, tort)
                if len(bp) > 5:
                    polylines.append((bp, False, True, rng.uniform(0.4, 0.8)))
    elif pattern == "map":
        cx = rng.uniform(x0 + 0.25 * sl, x1 - 0.25 * sl)
        cy = rng.uniform(y0 + 0.25 * sw, y1 - 0.25 * sw)
        r = float(rng.uniform(0.15, 0.3))
        box = (max(cx - r, inner[0]), min(cx + r, inner[1]),
               max(cy - r, inner[2]), min(cy + r, inner[3]))
        for _ in range(int(rng.integers(4, 9))):
            s = (rng.uniform(box[0], box[1]), rng.uniform(box[2], box[3]))
            bp = _walk(rng, s, rng.uniform(0, 2 * math.pi),
                       rng.uniform(0.1, 0.35), box, tort * 1.5)
            if len(bp) > 5:
                polylines.append((bp, True, True, rng.uniform(0.6, 1.0)))
    else:
        raise ValueError(pattern)

    paths = []
    for pts, fs, fe, ws in polylines:
        if len(pts) < 6:
            continue
        pts = _resample(pts)
        pts = _jitter(rng, pts, 0.35 * w0 * ws + rng.uniform(0.0004, 0.0015))
        width, depth = _profile(rng, pts, w0 * ws, depth_ratio, fs, fe)
        paths.append(CrackPath(pts, width, depth))
    meta = {
        "pattern": pattern, "width_class": str(wclass),
        "nominal_width_m": w0, "depth_ratio": depth_ratio,
        "tortuosity": tort, "n_branches": max(len(paths) - 1, 0),
    }
    return paths, meta


# =====================================================================
# SCENE
# =====================================================================

def _world_to_idx(xy, x0, y0):
    return (np.floor((xy[:, 0] - x0) / RES_M).astype(int),
            np.floor((xy[:, 1] - y0) / RES_M).astype(int))


def rasterise_crack(paths, x0, y0, shape, edge_noise):
    """Rasterise one crack instance. Returns (window slices, mask, depth_m,
    nearest-width) for the bounding window only."""
    ny, nx = shape
    pts = np.concatenate([p.pts for p in paths])
    wid = np.concatenate([p.width for p in paths])
    dep = np.concatenate([p.depth for p in paths])
    ix, iy = _world_to_idx(pts, x0, y0)
    pad = int(math.ceil(wid.max() / RES_M)) + 3
    c0, c1 = max(ix.min() - pad, 0), min(ix.max() + pad + 1, nx)
    r0, r1 = max(iy.min() - pad, 0), min(iy.max() + pad + 1, ny)
    sl = (slice(r0, r1), slice(c0, c1))
    seed = np.full((r1 - r0, c1 - c0), -1, np.int64)
    ok = (ix >= c0) & (ix < c1) & (iy >= r0) & (iy < r1)
    seed[iy[ok] - r0, ix[ok] - c0] = np.nonzero(ok)[0]
    dist, (ri, ci) = ndimage.distance_transform_edt(seed < 0,
                                                    return_indices=True)
    nearest = seed[ri, ci]
    dist_m = dist * RES_M
    r_eff = 0.5 * wid[nearest] * edge_noise[sl]
    mask = dist_m <= np.maximum(r_eff, 0.5 * RES_M)
    frac = np.clip(dist_m / np.maximum(r_eff, 1e-6), 0, 1)
    depth = np.where(mask, dep[nearest] * (1 - frac ** 2), 0.0)
    return sl, mask, depth.astype(np.float32), r_eff.astype(np.float32), frac


def measure_crack(paths, mask, sl, x0, y0, z_top):
    """Per-crack record in visual_review.csv's fields."""
    wid = np.concatenate([p.width for p in paths])
    dep = np.concatenate([p.depth for p in paths])
    length = float(sum(np.linalg.norm(np.diff(p.pts, axis=0), axis=1).sum()
                       for p in paths))
    rr, cc = np.nonzero(mask)
    rr = rr + sl[0].start
    cc = cc + sl[1].start
    xs = x0 + (cc + 0.5) * RES_M
    ys = y0 + (rr + 0.5) * RES_M
    area = float(len(xs)) * RES_M ** 2
    xy = np.c_[xs, ys]
    c = xy.mean(axis=0)
    cov = np.cov((xy - c).T) if len(xy) > 2 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, -1]
    minor = evecs[:, 0]
    pl = (xy - c) @ major
    pw = (xy - c) @ minor
    return {
        "source_label": "crack",
        "detected_class": "crack",
        "confirmed_class": "crack",
        "polarity": "depression",
        "length_m": length,
        "mean_width_m": float(wid.mean()),
        "median_width_m": float(np.median(wid)),
        "max_width_m": float(wid.max()),
        "max_depth_m": float(dep.max()),
        "mean_depth_m": float(dep.mean()),
        "median_depth_m": float(np.median(dep)),
        "max_height_m": 0.0,
        "mean_height_m": 0.0,
        "projected_area_m2": area,
        "area_over_length_m": area / length if length > 0 else 0.0,
        "center_world_m": [float(c[0]), float(c[1]), float(z_top)],
        "bbox_world_xy_m": [float(xs.min()), float(ys.min()),
                            float(xs.max()), float(ys.max())],
        "major_axis_world_xy": [float(major[0]), float(major[1])],
        "pixel_count": int(len(xs)),
        "pca_length_m": float(pl.max() - pl.min() + RES_M),
        "pca_width_m": float(pw.max() - pw.min() + RES_M),
        "grid_bbox_rc": [int(rr.min()), int(cc.min()),
                         int(rr.max()), int(cc.max())],
    }


@dataclass
class Scene:
    params: SceneParams
    x0: float
    y0: float
    height: np.ndarray       # (ny, nx) m, fine
    albedo: np.ndarray       # (ny, nx, 3) linear-ish 0..1
    roughness: np.ndarray    # (ny, nx)
    labels: np.ndarray       # (ny, nx) uint16, 0 = no crack visible
    cracks: List[Dict] = field(default_factory=list)
    centerlines: List[Dict] = field(default_factory=list)
    joints: List[Dict] = field(default_factory=list)


def generate_scene(seed: int, params: SceneParams = None) -> Scene:
    p = params or sample_scene_params(seed)
    rng = np.random.default_rng(seed + 1_000_003)
    L, W = p.length_m, p.width_m
    nx, ny = int(round(L / RES_M)), int(round(W / RES_M))
    x0, y0 = 0.0, -W / 2
    shape = (ny, nx)

    # --- slabs and joints -------------------------------------------------
    xj = list(np.arange(p.joint_spacing_m * rng.uniform(0.3, 1.0), L - 0.2,
                        p.joint_spacing_m))
    xedges = [0.0] + xj + [L]
    yedges = [-W / 2, 0.0, W / 2] if p.longitudinal_joint else [-W / 2, W / 2]
    slabs = [(xedges[i], xedges[i + 1], yedges[j], yedges[j + 1])
             for i in range(len(xedges) - 1) for j in range(len(yedges) - 1)]

    height = np.zeros(shape, np.float32)
    lum = np.ones(shape, np.float32)
    xs = x0 + (np.arange(nx) + 0.5) * RES_M
    ys = y0 + (np.arange(ny) + 0.5) * RES_M

    # Slab-level offsets (small, sub-ADA faulting) and brightness.
    for (a, b, c, d) in slabs:
        ca, cb = int(a / RES_M), int(b / RES_M)
        ra, rb = int((c - y0) / RES_M), int((d - y0) / RES_M)
        height[ra:rb, ca:cb] += rng.uniform(-0.0015, 0.0015)
        lum[ra:rb, ca:cb] *= 1 + rng.uniform(-0.05, 0.05)

    height += 0.0008 * fbm(shape, rng, 400, octaves=3)
    height += p.micro_amp_m * fbm(shape, rng, 6, octaves=3)

    # Tooled joints -- grooves, NOT cracks. They stay label 0: a model must
    # learn that a straight tooled joint is normal sidewalk.
    joint_mask = np.zeros(shape, bool)
    jw = rng.uniform(0.006, 0.011)
    jd = rng.uniform(0.008, 0.015)
    for xjv in xj:
        d = np.abs(xs - xjv)
        prof = np.clip(1 - (d / (jw / 2)) ** 2, 0, 1)
        height -= (jd * prof)[None, :].astype(np.float32)
        joint_mask[:, d <= jw / 2] = True
    if p.longitudinal_joint:
        d = np.abs(ys - 0.0)
        prof = np.clip(1 - (d / (jw / 2)) ** 2, 0, 1)
        height -= (jd * prof)[:, None].astype(np.float32)
        joint_mask[d <= jw / 2, :] = True
    joints = ([{"axis": "x", "pos_m": float(v)} for v in xj]
              + ([{"axis": "y", "pos_m": 0.0}] if p.longitudinal_joint else []))

    # --- concrete appearance ---------------------------------------------
    lum *= 1 + 0.10 * fbm(shape, rng, 350, octaves=4)
    lum *= 1 + 0.025 * fbm(shape, rng, 25, octaves=4)
    if p.broom_amp > 0:
        lum *= 1 + p.broom_amp * fbm(shape, rng, 60, octaves=3,
                                     aniso=(1.0, 0.04))
    for size, dens in ((1, rng.uniform(0.004, 0.02)), (2, rng.uniform(0.0, 0.004)),
                       (4, rng.uniform(0.0, 0.0008))):
        speck = rng.random(shape) < dens
        if size > 1:
            speck = cv2.dilate(speck.astype(np.uint8),
                               np.ones((size, size), np.uint8)) > 0
        lum[speck] *= rng.uniform(0.72, 1.2, size=int(speck.sum()))
    if p.stain_amp > 0:
        st = fbm(shape, rng, 500, octaves=3)
        lum *= 1 - p.stain_amp * np.clip(st - 0.8, 0, None)
    lum[joint_mask] *= 0.72

    rough = np.full(shape, p.roughness_base, np.float32)
    rough += 0.06 * fbm(shape, rng, 80, octaves=3)

    edge_noise = 1 + 0.3 * fbm(shape, rng, 5, octaves=2)
    edge_noise = np.clip(edge_noise, 0.6, 1.4).astype(np.float32)

    # --- cracks -----------------------------------------------------------
    labels = np.zeros(shape, np.uint16)
    crack_depth = np.zeros(shape, np.float32)
    dark = np.ones(shape, np.float32)
    fill = np.zeros(shape, np.float32)
    weed = np.zeros(shape, np.float32)
    records, centerlines = [], []
    names, wts = list(PATTERNS), list(PATTERNS.values())
    cid = 0
    attempts = 0
    while cid < p.n_cracks and attempts < p.n_cracks * 4:
        attempts += 1
        slab = slabs[int(rng.integers(len(slabs)))]
        pattern = str(rng.choice(names, p=wts))
        paths, meta = make_crack(rng, slab, pattern)
        if not paths:
            continue
        length = sum(np.linalg.norm(np.diff(q.pts, axis=0), axis=1).sum()
                     for q in paths)
        if length < 0.08:
            continue
        sl, mask, dep, r_eff, frac = rasterise_crack(paths, x0, y0, shape,
                                                     edge_noise)
        if mask.sum() == 0:
            continue
        cid += 1
        free = labels[sl] == 0
        labels[sl][mask & free] = cid
        crack_depth[sl] = np.maximum(crack_depth[sl], dep)
        # Appearance: dark interior, darker for deeper/wider cracks; hairline
        # cracks are subtle. Shaded across the section.
        w0 = meta["nominal_width_m"]
        lo, hi = {"hairline": (0.45, 0.72), "fine": (0.28, 0.55),
                  "medium": (0.14, 0.36), "wide": (0.07, 0.25)}[meta["width_class"]]
        core = float(rng.uniform(lo, hi))
        # Flat dark interior with a thin transition at the lip: the inside
        # of a crack is in shadow; only the edge is grey.
        shade = 1 - (1 - core) * np.clip((1 - frac) * 3.0, 0, 1) ** 0.7
        dark[sl] = np.where(mask, np.minimum(dark[sl], shade), dark[sl])
        if rng.random() < 0.3:
            fill[sl] = np.where(mask, rng.uniform(0.3, 0.8), fill[sl])
        if w0 > 0.008 and rng.random() < 0.15:
            weed[sl] = np.where(mask, rng.uniform(0.4, 0.9), weed[sl])
        rec = measure_crack(paths, mask, sl, x0, y0, z_top=0.0)
        rec.update(meta)
        rec["defect_id"] = f"crack_{cid:03d}"
        rec["instance_id"] = cid
        records.append(rec)
        centerlines.append({
            "instance_id": cid,
            "paths": [{"xy": np.round(q.pts[::20], 5).tolist(),
                       "width_m": np.round(q.width[::20], 5).tolist()}
                      for q in paths],
        })
    height -= crack_depth

    # --- distractors (texture only) --------------------------------------
    base = p.base_gray * np.asarray(p.tint, np.float32)
    albedo = np.clip(lum, 0.05, 2.0)[..., None] * base[None, None, :]
    albedo *= dark[..., None]
    if fill.any():
        dirt = np.array([0.20, 0.16, 0.11], np.float32)
        albedo = albedo * (1 - fill[..., None]) + dirt * fill[..., None]
    if weed.any():
        green = np.array([0.12, 0.22, 0.07], np.float32)
        albedo = albedo * (1 - weed[..., None]) + green * weed[..., None]
    rough[labels > 0] += 0.1

    for _ in range(p.n_gum):   # gum / tar spots: a stain, crack stays visible
        cx, cy = rng.integers(nx), rng.integers(ny)
        r = int(rng.uniform(4, 14))
        g = np.zeros(shape, np.uint8)
        cv2.circle(g, (int(cx), int(cy)), r, 1, -1)
        albedo[g > 0] = albedo[g > 0] * rng.uniform(0.35, 0.6)
    occluded = np.zeros(shape, bool)
    leaf_cols = [(0.35, 0.22, 0.09), (0.55, 0.42, 0.12), (0.18, 0.28, 0.08),
                 (0.45, 0.15, 0.06)]
    for _ in range(p.n_leaves):         # leaves sit ON TOP of cracks
        cx, cy = int(rng.integers(nx)), int(rng.integers(ny))
        ax = (int(rng.uniform(15, 40)), int(rng.uniform(8, 20)))
        g = np.zeros(shape, np.uint8)
        cv2.ellipse(g, (cx, cy), ax, float(rng.uniform(0, 180)), 0, 360, 1, -1)
        m = g > 0
        col = np.asarray(leaf_cols[int(rng.integers(len(leaf_cols)))],
                         np.float32) * rng.uniform(0.8, 1.2)
        albedo[m] = col
        rough[m] = 0.5
        height[m] = np.maximum(height[m], 0.0005)
        occluded |= m
    labels[occluded] = 0   # a crack hidden under a leaf is not visible

    albedo = np.clip(albedo, 0, 1).astype(np.float32)
    rough = np.clip(rough, 0.05, 1.0).astype(np.float32)

    # Visible-pixel count per crack after occlusion and overlaps.
    counts = np.bincount(labels.ravel(), minlength=len(records) + 1)
    for rec in records:
        rec["visible_raster_cells"] = int(counts[rec["instance_id"]])

    return Scene(p, x0, y0, height, albedo, rough, labels, records,
                 centerlines, joints)


# =====================================================================
# EXPORT
# =====================================================================

def _to_srgb8(lin):
    return (np.clip(lin, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


def mesh_height(scene: Scene) -> np.ndarray:
    k = int(round(MESH_CELL_M / RES_M))
    ny, nx = scene.height.shape
    h = scene.height[: ny // k * k, : nx // k * k]
    return h.reshape(ny // k, k, nx // k, k).mean(axis=(1, 3)).astype(np.float32)


def normal_map(scene: Scene, hm: np.ndarray) -> np.ndarray:
    """Tangent-space (OpenGL, +v = +y) normal map of the detail the render
    mesh cannot carry: fine height minus the upsampled mesh height."""
    ny, nx = scene.height.shape
    coarse = cv2.resize(hm, (nx, ny), interpolation=cv2.INTER_LINEAR)
    # Smooth one pixel before differencing: raw 1 mm edge noise turns into
    # saw-tooth normal spikes along crack lips.
    detail = cv2.GaussianBlur(scene.height - coarse, (0, 0), 1.5)
    gy, gx = np.gradient(detail, RES_M)
    gx, gy = np.clip(gx, -0.6, 0.6), np.clip(gy, -0.6, 0.6)
    n = np.dstack([-gx, -gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return ((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)


def save_scene(scene: Scene, out_dir: str, preview: bool = True) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    flip = lambda a: a[::-1]
    cv2.imwrite(os.path.join(out_dir, "albedo.jpg"),
                cv2.cvtColor(flip(_to_srgb8(scene.albedo)), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir, "roughness.jpg"),
                flip((scene.roughness * 255).astype(np.uint8)),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    hm = mesh_height(scene)
    nm = normal_map(scene, hm)
    cv2.imwrite(os.path.join(out_dir, "normal.png"),
                cv2.cvtColor(flip(nm), cv2.COLOR_RGB2BGR))
    np.save(os.path.join(out_dir, "height_mesh.npy"), hm)
    np.savez_compressed(os.path.join(out_dir, "labels.npz"),
                        labels=scene.labels)
    meta = {
        "scene_id": f"scene_{scene.params.seed:05d}",
        "params": {k: v for k, v in asdict(scene.params).items()
                   if not k.startswith("_")},
        "res_m": RES_M, "mesh_cell_m": MESH_CELL_M,
        "x0": scene.x0, "y0": scene.y0,
        "length_m": scene.params.length_m, "width_m": scene.params.width_m,
        "raster_shape": list(scene.labels.shape),
        "mesh_shape": list(hm.shape),
        "labels_layout": "labels.npz['labels'][iy, ix], NOT flipped",
        "texture_layout": "images flipped: row 0 = max y; u=(x-x0)/L, v=(y-y0)/W",
        "joints": scene.joints,
        "cracks": scene.cracks,
        "centerlines": scene.centerlines,
    }
    with open(os.path.join(out_dir, "scene.json"), "w") as fh:
        json.dump(meta, fh)
    if preview:
        s = 4
        img = _to_srgb8(scene.albedo[::s, ::s]).copy()
        lab = ndimage.maximum_filter(scene.labels, size=s)[::s, ::s]
        img[lab > 0] = (0.4 * img[lab > 0] + [153, 0, 0]).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "preview.jpg"),
                    cv2.cvtColor(flip(img), cv2.COLOR_RGB2BGR))
    return meta


# =====================================================================
# SELF-TEST
# =====================================================================

def _self_test():
    n_total = 0
    for seed in (1, 2, 3, 4):
        sc = generate_scene(seed)
        n_total += len(sc.cracks)
        assert sc.labels.shape == sc.albedo.shape[:2] == sc.height.shape
        for r in sc.cracks:
            assert r["pixel_count"] > 0, r
            # area ~ length x mean width: rasterised width tracks the profile
            ratio = r["area_over_length_m"] / r["mean_width_m"]
            floor = RES_M / r["mean_width_m"]   # 1-px minimum for hairlines
            assert 0.6 < ratio < max(1.5, 1.3 * floor + 0.5), (seed, r["defect_id"], ratio)
            assert r["max_depth_m"] >= r["mean_depth_m"] > 0
            assert r["max_width_m"] >= r["median_width_m"] > 0
            # crack region is below the surrounding surface
            rr0, cc0, rr1, cc1 = r["grid_bbox_rc"]
            win = sc.labels[rr0:rr1 + 1, cc0:cc1 + 1] == r["instance_id"]
            if win.any():
                hwin = sc.height[rr0:rr1 + 1, cc0:cc1 + 1]
                assert hwin[win].mean() < np.median(hwin[~win]) if (~win).any() else True
        # joints are never labelled as cracks unless a crack crosses them
        assert sc.labels.dtype == np.uint16
    print(f"self-test OK: 4 scenes, {n_total} cracks")


if __name__ == "__main__":
    _self_test()
