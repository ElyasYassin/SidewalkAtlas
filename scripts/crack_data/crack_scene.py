"""
Synthetic cracked-sidewalk scene: geometry, matching texture, exact labels,
inside a street context.

Pure numpy/scipy/cv2 -- no Isaac. One crack definition (centreline, width
profile, depth profile) drives all three outputs, so the picture and the
labels cannot disagree:

  * the 1 mm height raster (mesh + normal map) -- what the geometry is
  * the 1 mm albedo / roughness rasters         -- what the camera sees
  * the 1 mm instance-label raster + per-crack measurements -- the truth

Layering, in order -- the order is what keeps labels honest:
  1. base surface: concrete (photo detail + procedural), exposed aggregate,
     asphalt or pavers; weathering; driveway apron; locate paint
  2. cracks (labelled) -- only on concrete, aggregate and asphalt
  3. covers that sit ON TOP and erase crack labels where they lie: manholes,
     utility lids, drain grates, tree pits, tactile pads, asphalt patches,
     sealant lines, leaves
Paint is base, not cover: a crack under spray paint is still an open crack.

The street around the sidewalk (curb, gutter, road, parkway, frontage, poles,
trees, hydrants) is described in scene.json["street"] and built by the
renderer; only the walking surface is a raster.

Raster convention: arrays are indexed [iy, ix]; cell centre is
    x = x0 + (ix + 0.5) * RES_M,   y = y0 + (iy + 0.5) * RES_M.
x runs along the walk; y = 0 is the road-side edge, y = W the frontage edge.
Texture images are written with rows flipped (image row 0 = max y), which is
what a UV mapping u = (x - x0) / L, v = (y - y0) / W expects.

Measurement fields match crack_detection's visual_review.csv. Width and depth
come from the generative profile, sampled every 0.5 mm along the centreline:
exact by construction. Depth is the OPEN depth -- to the dirt fill where a
crack is partly filled, since that is what a camera or probe can reach.

Run directly for a self-test.
"""

import glob
import json
import math
import os
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy import ndimage


RES_M = 0.001          # texture / label / fine-height raster
MESH_CELL_M = 0.005    # render mesh cell; finer detail goes in the normal map
SAMPLE_STEP_M = 0.0005  # centreline resampling for rasterisation
MAX_DEPTH_M = 0.030

TEX_DIR = "D:/textures/acg"    # fetch_textures.py; empty dir -> procedural only

# Checked by eye (contact sheets): none of these has cracks of its own, which
# would be unlabelled cracks in the data.
TEXTURES = {
    "concrete": ["Concrete002", "Concrete006", "Concrete014", "Concrete015",
                 "Concrete016", "Concrete017", "Concrete030", "Concrete034",
                 "Concrete037", "Concrete040", "Concrete041A", "Concrete043A",
                 "Concrete044A", "Concrete047A", "Concrete048"],
    "aggregate": ["Concrete003", "Concrete018", "Concrete025"],
    "asphalt": ["Asphalt023S", "Asphalt025C", "Asphalt031", "Asphalt032",
                "Asphalt033"],        # Asphalt026C excluded: it is cracked
    "road": ["Road012A", "Road013A", "Road014A", "Road015A", "Road008A",
             "Road011A", "Road009C", "Asphalt025C", "Asphalt031"],
    "pavers": ["PavingStones070", "PavingStones081", "PavingStones111",
               "PavingStones122", "PavingStones127", "PavingStones128",
               "PavingStones130", "PavingStones131", "PavingStones136",
               "PavingStones138", "PavingStones141", "PavingStones142",
               "PavingStones146", "PavingStones147", "PavingStones149",
               "PavingStones150", "PavingStones151", "Bricks085", "Bricks101",
               "Bricks104", "Bricks066"],
    "bricks": ["Bricks066", "Bricks075A", "Bricks085", "Bricks092",
               "Bricks094", "Bricks097", "Bricks101", "Bricks102",
               "Bricks104", "Bricks105"],
    "metal": ["DiamondPlate001", "DiamondPlate003", "DiamondPlate008A",
              "DiamondPlate009"],
    "tactile": ["TactilePaving001", "TactilePaving002", "TactilePaving003",
                "TactilePaving005", "TactilePaving006"],
    "grass": ["Grass001", "Grass004", "Grass005", "Ground003", "Ground037"],
    "soil": ["Ground039", "Ground062S"],
    "gravel": ["Gravel022", "Gravel023", "Gravel040"],
}

SURFACES = {"concrete": 0.50, "aggregate": 0.12, "asphalt": 0.16, "pavers": 0.22}

WIDTH_CLASSES = {          # total width (m), weight on concrete, on asphalt
    "hairline": ((0.0008, 0.002), 0.25, 0.05),
    "fine": ((0.002, 0.005), 0.35, 0.30),
    "medium": ((0.005, 0.012), 0.28, 0.40),
    "wide": ((0.012, 0.030), 0.12, 0.25),
}
PATTERNS = {               # weight on concrete, on asphalt
    "transverse": (0.28, 0.20),
    "longitudinal": (0.20, 0.25),
    "diagonal": (0.20, 0.10),
    "branching": (0.22, 0.20),
    "map": (0.10, 0.25),
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
# PHOTO TEXTURES
# =====================================================================

def available(category):
    return [a for a in TEXTURES.get(category, [])
            if glob.glob(os.path.join(TEX_DIR, a, "*_Color.jpg"))]


def tex_path(asset, kind="Color"):
    f = glob.glob(os.path.join(TEX_DIR, asset, f"*_{kind}.jpg"))
    return f[0].replace("\\", "/") if f else None


@lru_cache(maxsize=24)
def _load(asset, kind):
    p = tex_path(asset, kind)
    if p is None:
        return None
    im = cv2.imread(p, cv2.IMREAD_COLOR if kind == "Color" else cv2.IMREAD_GRAYSCALE)
    if kind == "Color":
        return (cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255) ** 2.2
    return im.astype(np.float32) / 255


def tile(asset, kind, shape, tile_m, rng, x0=0.0, y0=0.0, rot90=None, region=None):
    """Sample a photo texture over the raster, the texture spanning `tile_m`
    metres, random offset, rotated by a multiple of 90 deg (pavers stay
    aligned with the walk). `region` = (r0, r1, c0, c1) restricts output."""
    tex = _load(asset, kind)
    if tex is None:
        return None
    th, tw = tex.shape[:2]
    r0, r1, c0, c1 = region or (0, shape[0], 0, shape[1])
    ys = y0 + (np.arange(r0, r1, dtype=np.float32) + 0.5) * RES_M
    xs = x0 + (np.arange(c0, c1, dtype=np.float32) + 0.5) * RES_M
    X, Y = np.meshgrid(xs, ys)
    k = int(rng.integers(4)) if rot90 is None else rot90
    if k == 1:
        X, Y = Y, -X
    elif k == 2:
        X, Y = -X, -Y
    elif k == 3:
        X, Y = -Y, X
    s = tw / tile_m
    mx = ((X * s + rng.uniform(0, tw)) % tw).astype(np.float32)
    my = ((Y * s + rng.uniform(0, th)) % th).astype(np.float32)
    return cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# =====================================================================
# PARAMETERS
# =====================================================================

@dataclass
class SceneParams:
    seed: int
    surface: str             # concrete | aggregate | asphalt | pavers
    surface_texture: str     # photo asset or "" (procedural)
    length_m: float          # along walk (x)
    width_m: float           # across walk (y)
    joint_spacing_m: float
    longitudinal_joint: bool
    n_cracks: int
    base_gray: float
    tint: Tuple[float, float, float]
    photo_strength: float
    photo_color_mix: float
    broom_amp: float
    stain_amp: float
    n_gum: int
    n_leaves: int
    micro_amp_m: float
    roughness_base: float
    joint_dirt: float
    edge_dirt: float


def sample_scene_params(seed: int) -> SceneParams:
    rng = np.random.default_rng(seed)
    tints = [(1.0, 1.0, 1.0), (1.04, 1.0, 0.93), (0.96, 0.99, 1.04),
             (1.05, 1.02, 0.88), (1.02, 0.98, 0.95)]
    surface = str(rng.choice(list(SURFACES), p=list(SURFACES.values())))
    pool = available(surface)
    tex = str(rng.choice(pool)) if pool and (surface != "concrete"
                                             or rng.random() < 0.85) else ""
    if surface in ("asphalt", "pavers") and not tex:
        surface = "concrete"          # these need a photo; fall back
    width = float(rng.uniform(1.2, 2.2))
    cracked = surface in ("concrete", "aggregate", "asphalt")
    return SceneParams(
        seed=seed, surface=surface, surface_texture=tex,
        length_m=6.0,
        width_m=round(width, 3),
        joint_spacing_m=float(rng.uniform(1.1, 1.8)),
        longitudinal_joint=bool(width > 1.7 and rng.random() < 0.5),
        n_cracks=(0 if (not cracked or rng.random() < 0.07)
                  else int(rng.integers(1, 9))),
        base_gray=float(rng.uniform(0.38, 0.72)),
        tint=tuple(tints[int(rng.integers(len(tints)))]),
        photo_strength=float(rng.uniform(0.6, 1.3)),
        photo_color_mix=float(rng.uniform(0.0, 0.7)),
        broom_amp=float(rng.uniform(0.0, 0.045)) if surface == "concrete" else 0.0,
        stain_amp=float(rng.uniform(0.0, 0.25)),
        n_gum=int(rng.integers(0, 40)),
        n_leaves=int(rng.integers(0, 25)) if rng.random() < 0.45 else 0,
        micro_amp_m=float(rng.uniform(0.0001, 0.0005)),
        roughness_base=float(rng.uniform(0.55, 0.9)),
        joint_dirt=float(rng.uniform(0.0, 1.0)),
        edge_dirt=float(rng.uniform(0.0, 1.0)),
    )


def sample_street(seed: int, p: SceneParams, inserts: List[Dict]) -> Dict:
    """Street context around the sidewalk, built by the renderer. The road is
    at y < 0. Trees stand in tree pits when there are any."""
    rng = np.random.default_rng(seed + 7_919)
    L, W = p.length_m, p.width_m
    curb = rng.random() < 0.75
    park_w = float(rng.uniform(0.5, 1.8)) if rng.random() < 0.45 else 0.0
    road_pool = available("road") or [""]
    st = {
        "curb": {"present": bool(curb),
                 "height_m": float(rng.uniform(0.10, 0.18)),
                 "top_width_m": float(rng.uniform(0.12, 0.2)),
                 "gutter_width_m": float(rng.uniform(0.3, 0.6)),
                 "color_gray": float(np.clip(p.base_gray * rng.uniform(0.85, 1.15), 0.3, 0.8))},
        "parkway": {"width_m": park_w,
                    "kind": str(rng.choice(["grass", "grass", "soil", "gravel"])),
                    "z_m": -float(rng.uniform(0.0, 0.04))},
        "road": {"texture": str(rng.choice(road_pool)),
                 "tile_m": float(rng.uniform(3.0, 5.0))},
        "verge": {"kind": str(rng.choice(["grass", "soil", "gravel"])),
                  "width_m": float(rng.uniform(0.5, 2.5)),
                  "z_m": -float(rng.uniform(0.005, 0.04))},
        "frontage": {"kind": str(rng.choice(["lawn", "lawn", "wall", "building",
                                             "fence", "hedge"])),
                     "offset_m": float(rng.uniform(0.0, 0.6)),
                     "height_m": float(rng.uniform(0.6, 6.0)),
                     "texture": str(rng.choice(available("bricks") or [""]))},
        "props": [],
    }
    for k in ("parkway", "verge"):
        pool = available(st[k]["kind"])
        st[k]["texture"] = str(rng.choice(pool)) if pool else ""
    pool = available("grass")
    st["frontage"]["lawn_texture"] = str(rng.choice(pool)) if pool else ""
    # road-side strip where poles/hydrants stand: parkway, else sidewalk edge
    def road_side_y():
        if park_w > 0:
            return -float(rng.uniform(0.15, park_w - 0.15))
        return float(rng.uniform(0.15, 0.35))
    for ins in inserts:
        if ins["type"] == "tree_pit":
            st["props"].append({"type": "tree", "x": ins["center"][0],
                                "y": ins["center"][1],
                                "trunk_r": float(rng.uniform(0.08, 0.22)),
                                "height": float(rng.uniform(2.5, 5.0)),
                                "canopy_r": float(rng.uniform(1.0, 2.8))})
    if park_w > 0.6 and rng.random() < 0.5:
        for _ in range(int(rng.integers(1, 3))):
            st["props"].append({"type": "tree", "x": float(rng.uniform(-1, L + 1)),
                                "y": -park_w / 2,
                                "trunk_r": float(rng.uniform(0.08, 0.22)),
                                "height": float(rng.uniform(2.5, 5.0)),
                                "canopy_r": float(rng.uniform(1.0, 2.8))})
    for _ in range(int(rng.integers(0, 3))):
        st["props"].append({"type": "pole", "x": float(rng.uniform(-1, L + 1)),
                            "y": road_side_y(),
                            "radius": float(rng.uniform(0.04, 0.12)),
                            "height": float(rng.uniform(2.5, 7.0))})
    if rng.random() < 0.3:
        st["props"].append({"type": "hydrant", "x": float(rng.uniform(0, L)),
                            "y": road_side_y(),
                            "color": [[0.6, 0.05, 0.03], [0.75, 0.6, 0.05],
                                      [0.1, 0.25, 0.6]][int(rng.integers(3))]})
    if rng.random() < 0.3:
        st["props"].append({"type": "bin", "x": float(rng.uniform(0, L)),
                            "y": float(W - rng.uniform(0.25, 0.4)),
                            "radius": float(rng.uniform(0.2, 0.3)),
                            "height": float(rng.uniform(0.7, 1.1))})
    if rng.random() < 0.25:
        st["props"].append({"type": "sign", "x": float(rng.uniform(0, L)),
                            "y": road_side_y(), "height": float(rng.uniform(2.0, 2.8))})
    return st


# =====================================================================
# CRACK GEOMETRY
# =====================================================================

@dataclass
class CrackPath:
    pts: np.ndarray     # (N, 2) metres
    width: np.ndarray   # (N,) total width, m
    depth: np.ndarray   # (N,) max OPEN depth of cross-section, m
    edge: np.ndarray    # (N,) edge-roughness factor (smooth along the crack)


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


def _profile(rng, pts, w0, depth_ratio, free_start, free_end, edge_amp):
    """Width / depth / edge factor along a resampled polyline."""
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
    depth = np.clip(depth_ratio * 0.6 * width ** 0.85 * dmod, 0.0008, MAX_DEPTH_M)
    # Edge roughness varies ALONG the crack at 3-15 mm, not per pixel: chips
    # and aggregate pull-outs, not single-pixel spikes.
    edge = 1 + edge_amp * (0.75 * smooth_noise_1d(n, rng, int(rng.uniform(6, 30)))
                           + 0.25 * smooth_noise_1d(n, rng, int(rng.uniform(3, 6))))
    return width, depth, np.clip(edge, 0.45, 1.8)


def make_crack(rng, slab, pattern, asphalt=False):
    """Build one crack instance on `slab` (x0, x1, y0, y1).
    Returns (list[CrackPath], meta dict)."""
    x0, x1, y0, y1 = slab
    sw, sl = y1 - y0, x1 - x0
    m = 0.01
    inner = (x0 + m, x1 - m, y0 + m, y1 - m)
    col = 2 if asphalt else 1
    wclass = rng.choice(list(WIDTH_CLASSES),
                        p=[v[col] for v in WIDTH_CLASSES.values()])
    if pattern == "map" and wclass in ("medium", "wide") and not asphalt:
        wclass = rng.choice(["hairline", "fine"])   # concrete map cracking is fine
    w0 = float(rng.uniform(*WIDTH_CLASSES[wclass][0]))
    depth_ratio = float(rng.uniform(0.8, 2.0))
    tort = float(rng.uniform(0.03, 0.14)) * (1.4 if asphalt else 1.0)
    edge_amp = float(rng.uniform(0.12, 0.25)) * (1.6 if asphalt else 1.0)
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
        L = min(sl, 2.5) * rng.uniform(0.3, 1.0)
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
        r = float(rng.uniform(0.15, 0.3) * (1.6 if asphalt else 1.0))
        box = (max(cx - r, inner[0]), min(cx + r, inner[1]),
               max(cy - r, inner[2]), min(cy + r, inner[3]))
        for _ in range(int(rng.integers(4, 9) * (1.5 if asphalt else 1))):
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
        width, depth, edge = _profile(rng, pts, w0 * ws, depth_ratio, fs, fe,
                                      edge_amp)
        paths.append(CrackPath(pts, width, depth, edge))
    meta = {
        "pattern": pattern, "width_class": str(wclass),
        "nominal_width_m": w0, "depth_ratio": depth_ratio,
        "tortuosity": tort, "n_branches": max(len(paths) - 1, 0),
        "surface": "asphalt" if asphalt else "concrete",
    }
    return paths, meta


# =====================================================================
# RASTER HELPERS
# =====================================================================

def _world_to_idx(xy, x0, y0):
    return (np.floor((xy[:, 0] - x0) / RES_M).astype(int),
            np.floor((xy[:, 1] - y0) / RES_M).astype(int))


def rasterise_crack(paths, x0, y0, shape, edge_noise, bevel_m=0.0):
    """Rasterise one crack instance in its bounding window. Returns slices,
    mask, depth, per-pixel nearest width, frac (0 centre .. 1 edge), and the
    bevel band just outside the edge with its drop profile (0..1)."""
    ny, nx = shape
    pts = np.concatenate([p.pts for p in paths])
    wid = np.concatenate([p.width for p in paths])
    dep = np.concatenate([p.depth for p in paths])
    edg = np.concatenate([p.edge for p in paths])
    ix, iy = _world_to_idx(pts, x0, y0)
    pad = int(math.ceil((wid.max() * 1.8 / 2 + bevel_m) / RES_M)) + 4
    c0, c1 = max(ix.min() - pad, 0), min(ix.max() + pad + 1, nx)
    r0, r1 = max(iy.min() - pad, 0), min(iy.max() + pad + 1, ny)
    sl = (slice(r0, r1), slice(c0, c1))
    seed = np.full((r1 - r0, c1 - c0), -1, np.int64)
    ok = (ix >= c0) & (ix < c1) & (iy >= r0) & (iy < r1)
    seed[iy[ok] - r0, ix[ok] - c0] = np.nonzero(ok)[0]
    dist, (ri, ci) = ndimage.distance_transform_edt(seed < 0, return_indices=True)
    nearest = seed[ri, ci]
    dist_m = dist * RES_M
    r_eff = 0.5 * wid[nearest] * edg[nearest] * edge_noise[sl]
    mask = dist_m <= np.maximum(r_eff, 0.5 * RES_M)
    frac = np.clip(dist_m / np.maximum(r_eff, 1e-6), 0, 1)
    # Flat-ish floor, steep walls: open depth falls off near the lip.
    depth = np.where(mask, dep[nearest] * (1 - frac ** 3), 0.0).astype(np.float32)
    bevel = np.zeros(mask.shape, np.float32)
    if bevel_m > 0:
        out = (~mask) & (dist_m < r_eff + bevel_m)
        bevel[out] = 1 - (dist_m[out] - r_eff[out]) / bevel_m
    return sl, mask, depth, wid[nearest].astype(np.float32), frac, bevel


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


def _rect(shape, x0, y0, xa, xb, ya, yb):
    ny, nx = shape
    c0, c1 = int(np.clip((xa - x0) / RES_M, 0, nx)), int(np.clip((xb - x0) / RES_M, 0, nx))
    r0, r1 = int(np.clip((ya - y0) / RES_M, 0, ny)), int(np.clip((yb - y0) / RES_M, 0, ny))
    return r0, r1, c0, c1


def _polyline_mask(shape, x0, y0, pts, width_m):
    """Thick polyline mask (cv2, anti-alias free)."""
    m = np.zeros(shape, np.uint8)
    p = np.round(np.c_[(pts[:, 0] - x0) / RES_M, (pts[:, 1] - y0) / RES_M]).astype(np.int32)
    cv2.polylines(m, [p], False, 1, max(int(round(width_m / RES_M)), 1))
    return m > 0


# =====================================================================
# SCENE
# =====================================================================

@dataclass
class Scene:
    params: SceneParams
    x0: float
    y0: float
    height: np.ndarray       # (ny, nx) m, fine
    albedo: np.ndarray       # (ny, nx, 3) linear 0..1
    roughness: np.ndarray    # (ny, nx)
    labels: np.ndarray       # (ny, nx) uint16, 0 = no crack visible
    width01mm: np.ndarray    # (ny, nx) uint16 local crack width, 0.1 mm units
    cracks: List[Dict] = field(default_factory=list)
    centerlines: List[Dict] = field(default_factory=list)
    joints: List[Dict] = field(default_factory=list)
    inserts: List[Dict] = field(default_factory=list)
    distractors: List[Dict] = field(default_factory=list)
    street: Dict = field(default_factory=dict)


def generate_scene(seed: int, params: SceneParams = None) -> Scene:
    p = params or sample_scene_params(seed)
    rng = np.random.default_rng(seed + 1_000_003)
    L, W = p.length_m, p.width_m
    nx, ny = int(round(L / RES_M)), int(round(W / RES_M))
    x0, y0 = 0.0, 0.0
    shape = (ny, nx)
    xs = x0 + (np.arange(nx) + 0.5) * RES_M
    ys = y0 + (np.arange(ny) + 0.5) * RES_M
    jointed = p.surface in ("concrete", "aggregate")
    cracked = p.surface in ("concrete", "aggregate", "asphalt")

    # =================== 1. BASE SURFACE ===================
    height = np.zeros(shape, np.float32)
    lum = np.ones(shape, np.float32)
    height += 0.0008 * fbm(shape, rng, 400, octaves=3)
    height += p.micro_amp_m * fbm(shape, rng, 6, octaves=3)

    # --- slabs and joints (concrete); one region otherwise -------------
    if jointed:
        xj = list(np.arange(p.joint_spacing_m * rng.uniform(0.3, 1.0), L - 0.2,
                            p.joint_spacing_m))
        yj = [W / 2] if p.longitudinal_joint else []
    else:
        xj, yj = [], []
    xedges = [0.0] + xj + [L]
    yedges = [0.0] + yj + [W]
    slabs = [(xedges[i], xedges[i + 1], yedges[j], yedges[j + 1])
             for i in range(len(xedges) - 1) for j in range(len(yedges) - 1)]
    for (a, b, c, d) in slabs:
        r0, r1, c0, c1 = _rect(shape, x0, y0, a, b, c, d)
        if jointed:
            height[r0:r1, c0:c1] += rng.uniform(-0.0015, 0.0015)
        lum[r0:r1, c0:c1] *= 1 + rng.uniform(-0.05, 0.05)

    base = p.base_gray * np.asarray(p.tint, np.float32)
    photo_color = None
    if p.surface in ("concrete", "aggregate"):
        lum *= 1 + 0.07 * fbm(shape, rng, 350, octaves=4)
        lum *= 1 + 0.02 * fbm(shape, rng, 25, octaves=4)
        if p.broom_amp > 0:
            lum *= 1 + p.broom_amp * fbm(shape, rng, 60, octaves=3, aniso=(1.0, 0.04))
        if p.surface_texture:
            ph = tile(p.surface_texture, "Color", shape, rng.uniform(1.5, 3.0), rng)
            plum = ph.mean(axis=2) + 1e-4
            # High-pass the photo: its fine structure (pores, aggregate,
            # trowel marks) replaces the cloudy procedural look; its large-
            # scale tone is ours to control.
            low = cv2.GaussianBlur(plum, (0, 0), 120)
            lum *= np.clip(plum / low, 0.2, 3.0) ** p.photo_strength
            photo_color = ph / plum[..., None] * (ph.mean(axis=(0, 1)).mean())
            if p.surface == "aggregate":
                photo_color = ph
        for size, dens in ((1, rng.uniform(0.002, 0.012)), (2, rng.uniform(0.0, 0.003))):
            speck = rng.random(shape) < dens
            if size > 1:
                speck = cv2.dilate(speck.astype(np.uint8), np.ones((size, size), np.uint8)) > 0
            lum[speck] *= rng.uniform(0.75, 1.2, size=int(speck.sum()))
    if p.surface in ("asphalt", "pavers"):
        photo_color = tile(p.surface_texture, "Color", shape,
                           rng.uniform(1.2, 2.4) if p.surface == "pavers"
                           else rng.uniform(1.5, 3.0), rng,
                           rot90=int(rng.integers(2)) if p.surface == "pavers" else None)
        if p.surface == "pavers":
            # Relief from the colour's own seams (joints are darker than the
            # stones), so geometry and appearance stay registered.
            g = photo_color.mean(axis=2)
            seam = np.clip(1 - g / (cv2.GaussianBlur(g, (0, 0), 25) + 1e-4), 0, 1)
            height -= (rng.uniform(0.003, 0.008) * cv2.GaussianBlur(seam, (0, 0), 1.0))
    if p.stain_amp > 0:
        stn = fbm(shape, rng, 500, octaves=3)
        lum *= 1 - p.stain_amp * np.clip(stn - 0.8, 0, None)

    if photo_color is not None and p.surface in ("asphalt", "pavers", "aggregate"):
        albedo = photo_color * np.clip(lum, 0.05, 2.0)[..., None]
        if p.surface == "aggregate":
            albedo *= (p.base_gray / max(photo_color.mean(), 1e-3))
    else:
        albedo = np.clip(lum, 0.05, 2.0)[..., None] * base[None, None, :]
        if photo_color is not None and p.photo_color_mix > 0:
            pc = photo_color * (p.base_gray / max(photo_color.mean(), 1e-3))
            albedo = (1 - p.photo_color_mix) * albedo + p.photo_color_mix * pc * np.clip(lum, 0.05, 2)[..., None]
    albedo = albedo.astype(np.float32)
    if p.surface in ("concrete", "aggregate"):
        # Concrete is near-gray. A bluish photo times a cool tint under a
        # blue sky renders purple, so cap chroma relative to luminance.
        g = albedo.mean(axis=2, keepdims=True)
        chroma = albedo - g
        sat = np.linalg.norm(chroma, axis=2, keepdims=True) / np.maximum(g, 1e-4)
        albedo = g + chroma * np.minimum(1.0, 0.07 / np.maximum(sat, 1e-6))

    rough = np.full(shape, p.roughness_base, np.float32)
    rough += 0.06 * fbm(shape, rng, 80, octaves=3)
    if p.surface == "asphalt":
        rough = np.clip(rough + 0.05, 0, 1)

    # --- tooled joints (unlabelled: a joint is not a crack) ------------
    joint_mask = np.zeros(shape, bool)
    joint_dist = np.full(shape, 1.0, np.float32)
    jw = rng.uniform(0.006, 0.011)
    jd = rng.uniform(0.008, 0.015)
    for xjv in xj:
        d = np.abs(xs - xjv)
        prof = np.clip(1 - (d / (jw / 2)) ** 2, 0, 1)
        height -= (jd * prof)[None, :].astype(np.float32)
        joint_mask[:, d <= jw / 2] = True
        joint_dist = np.minimum(joint_dist, d[None, :].astype(np.float32))
    for yjv in yj:
        d = np.abs(ys - yjv)
        prof = np.clip(1 - (d / (jw / 2)) ** 2, 0, 1)
        height -= (jd * prof)[:, None].astype(np.float32)
        joint_mask[d <= jw / 2, :] = True
        joint_dist = np.minimum(joint_dist, d[:, None].astype(np.float32))
    joints = ([{"axis": "x", "pos_m": float(v)} for v in xj]
              + [{"axis": "y", "pos_m": float(v)} for v in yj])
    if jointed:
        albedo[joint_mask] *= 0.72
        # weathering: dirt collects in and beside joints, and along the
        # sidewalk edges (road splash on the road side)
        if p.joint_dirt > 0:
            band = np.clip(1 - joint_dist / rng.uniform(0.01, 0.03), 0, 1)
            band *= np.clip(0.6 + 0.4 * fbm(shape, rng, 40, octaves=3), 0, 1.5)
            dirt = np.array([0.16, 0.13, 0.09], np.float32)
            k = (0.5 * p.joint_dirt * band)[..., None]
            albedo = albedo * (1 - k) + dirt * k
    if p.edge_dirt > 0:
        road_band = np.clip(1 - ys / rng.uniform(0.05, 0.25), 0, 1)[:, None]
        k = (0.35 * p.edge_dirt * road_band
             * np.clip(0.6 + 0.4 * fbm(shape, rng, 60, octaves=3), 0, 1.5))[..., None]
        albedo = albedo * (1 - k) + np.array([0.12, 0.10, 0.08], np.float32) * k

    # --- driveway apron: different finish, cross-slope down to the road --
    inserts, distractors = [], []
    if jointed and rng.random() < 0.2:
        aw = float(rng.uniform(2.0, 3.2))
        ax = float(rng.uniform(0.3, L - aw - 0.3))
        r0, r1, c0, c1 = _rect(shape, x0, y0, ax, ax + aw, 0, W)
        slope = rng.uniform(0.02, 0.06)
        height[r0:r1, c0:c1] -= (slope * (W - ys[r0:r1]))[:, None].astype(np.float32) * 0.5
        alt = available("concrete")
        if alt:
            ph = tile(str(rng.choice(alt)), "Color", (r1 - r0, c1 - c0),
                      rng.uniform(1.5, 3.0), rng)
            g = ph.mean(axis=2) + 1e-4
            albedo[r0:r1, c0:c1] *= (np.clip(g / cv2.GaussianBlur(g, (0, 0), 120), 0.3, 3)
                                     * rng.uniform(0.85, 1.15))[..., None]
        for xe in (ax, ax + aw):           # apron joints
            d = np.abs(xs - xe)
            sel = d <= jw / 2
            joint_mask[:, sel] = True
            height[:, sel] -= jd * 0.8
            albedo[:, sel] *= 0.72
        inserts.append({"type": "driveway_apron", "x_m": [ax, ax + aw],
                        "cross_slope": float(slope)})

    # --- locate paint (base layer: cracks under paint stay cracks) ------
    if rng.random() < 0.2:
        cols = [(0.9, 0.25, 0.02), (0.05, 0.2, 0.75), (0.9, 0.9, 0.85),
                (0.85, 0.75, 0.05), (0.1, 0.55, 0.15), (0.85, 0.2, 0.5)]
        for _ in range(int(rng.integers(1, 5))):
            start = (rng.uniform(0.2, L - 0.2), rng.uniform(0.1, W - 0.1))
            pts = _walk(rng, start, rng.uniform(0, 2 * math.pi),
                        rng.uniform(0.2, 1.2), (0, L, 0, W), 0.02)
            if len(pts) < 3:
                continue
            m = _polyline_mask(shape, x0, y0, pts, rng.uniform(0.025, 0.06))
            spray = np.clip(0.75 + 0.35 * fbm(shape, rng, 3, octaves=2), 0, 1)
            col = np.asarray(cols[int(rng.integers(len(cols)))], np.float32) ** 2.2
            k = (m * spray)[..., None]
            albedo = albedo * (1 - k) + col * k
            distractors.append({"type": "paint", "xy": np.round(pts[::10], 4).tolist()})

    # gum / tar spots: a stain, cracks under them stay visible
    for _ in range(p.n_gum):
        cx, cy = rng.integers(nx), rng.integers(ny)
        r = int(rng.uniform(4, 16))
        g = np.zeros(shape, np.uint8)
        cv2.circle(g, (int(cx), int(cy)), r, 1, -1)
        albedo[g > 0] *= rng.uniform(0.35, 0.6)

    # =================== 2. CRACKS ===================
    edge_noise = np.clip(1 + 0.06 * fbm(shape, rng, 8, octaves=2), 0.8, 1.2).astype(np.float32)
    labels = np.zeros(shape, np.uint16)
    width01 = np.zeros(shape, np.uint16)
    crack_depth = np.zeros(shape, np.float32)
    bevel_drop = np.zeros(shape, np.float32)
    dark = np.ones(shape, np.float32)
    fill = np.zeros(shape, np.float32)
    weed = np.zeros(shape, np.float32)
    records, centerlines = [], []
    col = 1 if p.surface == "asphalt" else 0
    names = list(PATTERNS)
    wts = np.array([v[col] for v in PATTERNS.values()])
    wts = wts / wts.sum()
    cid = 0
    attempts = 0
    while cracked and cid < p.n_cracks and attempts < p.n_cracks * 4:
        attempts += 1
        slab = slabs[int(rng.integers(len(slabs)))]
        pattern = str(rng.choice(names, p=wts))
        paths, meta = make_crack(rng, slab, pattern, asphalt=p.surface == "asphalt")
        if not paths:
            continue
        length = sum(np.linalg.norm(np.diff(q.pts, axis=0), axis=1).sum() for q in paths)
        if length < 0.08:
            continue
        w0 = meta["nominal_width_m"]
        # Partial dirt fill: the open depth (what can be seen or probed) is
        # what gets labelled and what the geometry shows.
        fill_frac = float(rng.uniform(0.3, 0.85)) if rng.random() < 0.4 else 0.0
        for q in paths:
            q.depth = np.maximum(q.depth * (1 - fill_frac), 0.0005)
        bev = float(rng.uniform(0.3, 0.9) * w0) if w0 > 0.0025 else 0.0
        sl, mask, dep, wloc, frac, bevel = rasterise_crack(paths, x0, y0, shape,
                                                          edge_noise, bev)
        if mask.sum() == 0:
            continue
        cid += 1
        free = labels[sl] == 0
        labels[sl][mask & free] = cid
        width01[sl][mask & free] = np.clip(wloc[mask & free] * 1e4, 1, 65535).astype(np.uint16)
        crack_depth[sl] = np.maximum(crack_depth[sl], dep)
        if bev > 0:   # rounded, worn lip: slightly lower and darker
            bevel_drop[sl] = np.maximum(bevel_drop[sl],
                                        bevel * rng.uniform(0.2, 0.6) * min(w0, 0.006))
            dark[sl] = np.where(bevel > 0, np.minimum(dark[sl], 1 - 0.1 * bevel), dark[sl])
        lo, hi = {"hairline": (0.45, 0.72), "fine": (0.28, 0.55),
                  "medium": (0.14, 0.36), "wide": (0.07, 0.25)}[meta["width_class"]]
        if p.surface == "asphalt":
            lo, hi = lo * 0.7, hi * 0.8
        core = float(rng.uniform(lo, hi))
        shade = 1 - (1 - core) * np.clip((1 - frac) * 3.0, 0, 1) ** 0.7
        dark[sl] = np.where(mask, np.minimum(dark[sl], shade), dark[sl])
        if fill_frac > 0:
            fill[sl] = np.where(mask, rng.uniform(0.35, 0.8), fill[sl])
        if w0 > 0.008 and rng.random() < 0.15:
            weed[sl] = np.where(mask, rng.uniform(0.4, 0.9), weed[sl])
        rec = measure_crack(paths, mask, sl, x0, y0, z_top=0.0)
        rec.update(meta)
        rec.update({"defect_id": f"crack_{cid:03d}", "instance_id": cid,
                    "fill_fraction": fill_frac})
        records.append(rec)
        centerlines.append({
            "instance_id": cid,
            "paths": [{"xy": np.round(q.pts[::20], 5).tolist(),
                       "width_m": np.round(q.width[::20], 5).tolist()}
                      for q in paths],
        })
    height -= crack_depth + bevel_drop
    albedo *= dark[..., None]
    if fill.any():
        dirt = np.array([0.20, 0.16, 0.11], np.float32)
        albedo = albedo * (1 - fill[..., None]) + dirt * fill[..., None]
    if weed.any():
        green = np.array([0.12, 0.22, 0.07], np.float32)
        albedo = albedo * (1 - weed[..., None]) + green * weed[..., None]
    rough[labels > 0] += 0.1

    # =================== 3. COVERS (erase labels) ===================
    covered = np.zeros(shape, bool)
    occupied = []

    def free_spot(w, h, y_lo=0.05, y_hi=None, tries=30):
        y_hi = (W - 0.05) if y_hi is None else y_hi
        for _ in range(tries):
            cx = rng.uniform(w / 2 + 0.05, L - w / 2 - 0.05)
            cy = rng.uniform(y_lo + h / 2, max(y_lo + h / 2, y_hi - h / 2))
            box = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
            if all(box[1] < o[0] or box[0] > o[1] or box[3] < o[2] or box[2] > o[3]
                   for o in occupied):
                occupied.append(box)
                return cx, cy
        return None

    def metal(shape_, region):
        pool = available("metal")
        if pool:
            t = tile(str(rng.choice(pool)), "Color", shape_, rng.uniform(0.4, 0.9),
                     rng, region=region)
            return t * rng.uniform(0.5, 0.9)
        return np.full(((region[1] - region[0]), (region[3] - region[2]), 3),
                       0.08, np.float32)

    def place(mask_box, r0, c0, color, rough_v, dz):
        sl2 = (slice(r0, r0 + mask_box.shape[0]), slice(c0, c0 + mask_box.shape[1]))
        if color.ndim == 1:
            albedo[sl2][mask_box] = color
        else:
            albedo[sl2][mask_box] = color[mask_box]
        rough[sl2][mask_box] = rough_v
        height[sl2][mask_box] = np.median(height[sl2]) + dz
        covered[sl2] |= mask_box

    # manhole
    if rng.random() < 0.25:
        d = float(rng.uniform(0.55, 0.85))
        spot = free_spot(d + 0.1, d + 0.1, 0.1)
        if spot and d + 0.2 < W:
            cx, cy = spot
            r0, r1, c0, c1 = _rect(shape, x0, y0, cx - d / 2 - 0.03, cx + d / 2 + 0.03,
                                   cy - d / 2 - 0.03, cy + d / 2 + 0.03)
            yy, xx = np.mgrid[r0:r1, c0:c1]
            rr = np.hypot((xx + 0.5) * RES_M + x0 - cx, (yy + 0.5) * RES_M + y0 - cy)
            lid = rr <= d / 2
            gap = (rr > d / 2) & (rr <= d / 2 + 0.004)
            frame = (rr > d / 2 + 0.004) & (rr <= d / 2 + 0.03)
            col_ = metal((r1 - r0, c1 - c0), (r0, r1, c0, c1))
            rings = (np.sin(rr / rng.uniform(0.02, 0.05) * 2 * math.pi) > 0.85)
            col_[rings & lid] *= 0.6
            place(lid, r0, c0, col_, rng.uniform(0.35, 0.7), rng.uniform(-0.004, 0.004))
            place(frame, r0, c0, col_ * 0.8, 0.5, 0.0)
            place(gap, r0, c0, np.array([0.01, 0.01, 0.01], np.float32), 0.9, -0.01)
            inserts.append({"type": "manhole", "center": [cx, cy], "diameter_m": d})

    # utility lids
    for _ in range(int(rng.integers(0, 3)) if rng.random() < 0.4 else 0):
        w, h = float(rng.uniform(0.3, 0.9)), float(rng.uniform(0.25, 0.6))
        spot = free_spot(w + 0.06, h + 0.06)
        if not spot:
            continue
        cx, cy = spot
        r0, r1, c0, c1 = _rect(shape, x0, y0, cx - w / 2 - 0.03, cx + w / 2 + 0.03,
                               cy - h / 2 - 0.03, cy + h / 2 + 0.03)
        hh, ww = r1 - r0, c1 - c0
        yy, xx = np.mgrid[0:hh, 0:ww]
        bd = np.minimum(np.minimum(xx, ww - 1 - xx), np.minimum(yy, hh - 1 - yy)) * RES_M
        frame = bd < 0.025
        gap = (bd >= 0.025) & (bd < 0.029)
        lid = bd >= 0.029
        if rng.random() < 0.6:
            col_ = metal((hh, ww), (r0, r1, c0, c1))
        else:   # precast concrete lid, lighter, with a lifting slot
            col_ = np.broadcast_to(base * rng.uniform(0.9, 1.3), (hh, ww, 3)).copy()
            col_[(np.abs(yy - hh / 2) < 6) & (np.abs(xx - ww / 2) < 30)] = 0.02
        place(lid, r0, c0, col_, rng.uniform(0.4, 0.8), rng.uniform(-0.003, 0.003))
        place(frame, r0, c0, col_ * 0.7, 0.5, 0.0)
        place(gap, r0, c0, np.array([0.01, 0.01, 0.01], np.float32), 0.9, -0.008)
        inserts.append({"type": "utility_lid", "center": [cx, cy], "size_m": [w, h]})

    # drain grate at the road-side edge
    if rng.random() < 0.15:
        w, h = float(rng.uniform(0.4, 0.8)), float(rng.uniform(0.25, 0.4))
        spot = free_spot(w, h, 0.02, h + 0.1)
        if spot:
            cx, cy = spot
            r0, r1, c0, c1 = _rect(shape, x0, y0, cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
            hh, ww = r1 - r0, c1 - c0
            yy, xx = np.mgrid[0:hh, 0:ww]
            pitch = int(rng.uniform(35, 55))
            slot = ((xx % pitch) < pitch * 0.55) & (yy > 20) & (yy < hh - 20) & (xx > 15) & (xx < ww - 15)
            bar = ~slot
            col_ = metal((hh, ww), (r0, r1, c0, c1)) * 0.7
            place(bar, r0, c0, col_, 0.5, -0.003)
            place(slot, r0, c0, np.array([0.005, 0.005, 0.005], np.float32), 0.9, -0.06)
            inserts.append({"type": "drain_grate", "center": [cx, cy], "size_m": [w, h]})

    # tree pit on the road side (the renderer stands a tree in it)
    if rng.random() < 0.22 and W > 1.3:
        s = float(rng.uniform(0.8, min(1.3, W - 0.5)))
        spot = free_spot(s, s, 0.0, s + 0.02)
        if spot:
            cx, cy = spot
            r0, r1, c0, c1 = _rect(shape, x0, y0, cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2)
            hh, ww = r1 - r0, c1 - c0
            yy, xx = np.mgrid[0:hh, 0:ww]
            bd = np.minimum(np.minimum(xx, ww - 1 - xx), np.minimum(yy, hh - 1 - yy))
            edge = bd < 12
            pit = ~edge
            pool = available("soil") + available("gravel")
            col_ = (tile(str(rng.choice(pool)), "Color", shape, rng.uniform(0.8, 1.5), rng,
                         region=(r0, r1, c0, c1)) if pool
                    else np.full((hh, ww, 3), 0.08, np.float32))
            grate = rng.random() < 0.3
            if grate:
                rr = np.hypot(xx - ww / 2, yy - hh / 2)
                ang = np.arctan2(yy - hh / 2, xx - ww / 2)
                bars = (np.sin(ang * rng.integers(24, 48)) > 0.3) | (np.sin(rr / 18) > 0.7)
                col_ = np.where(bars[..., None], metal((hh, ww), (r0, r1, c0, c1)) * 0.6, col_)
            place(pit, r0, c0, col_, 0.9, -float(rng.uniform(0.02, 0.06)) if not grate else -0.002)
            place(edge, r0, c0, np.array([0.05, 0.05, 0.05], np.float32), 0.6, 0.0)
            inserts.append({"type": "tree_pit", "center": [cx, cy], "size_m": s,
                            "grate": bool(grate)})

    # tactile warning pad at a curb ramp (one end, road side)
    if rng.random() < 0.25:
        pool = available("tactile")
        dpt, wd = 0.61, float(rng.uniform(0.9, 1.5))
        cx = float(rng.choice([wd / 2 + 0.1, L - wd / 2 - 0.1]))
        cy = dpt / 2 + 0.02
        box = (cx - wd / 2, cx + wd / 2, cy - dpt / 2, cy + dpt / 2)
        if pool and all(box[1] < o[0] or box[0] > o[1] or box[3] < o[2] or box[2] > o[3]
                        for o in occupied):
            occupied.append(box)
            r0, r1, c0, c1 = _rect(shape, x0, y0, *box)
            asset = str(rng.choice(pool))
            col_ = tile(asset, "Color", shape, rng.uniform(1.0, 1.4), rng,
                        region=(r0, r1, c0, c1), rot90=0)
            m = np.ones((r1 - r0, c1 - c0), bool)
            place(m, r0, c0, col_, 0.6, 0.0)
            g = col_.mean(axis=2)
            dome = np.clip((g - cv2.GaussianBlur(g, (0, 0), 20)) * 8, 0, 1)
            height[r0:r1, c0:c1] += 0.004 * cv2.GaussianBlur(dome, (0, 0), 2)
            inserts.append({"type": "tactile_pad", "center": [cx, cy],
                            "size_m": [wd, dpt], "texture": asset})

    # asphalt patch (on concrete)
    if jointed and rng.random() < 0.25:
        pool = available("asphalt")
        w, h = float(rng.uniform(0.3, 1.2)), float(rng.uniform(0.3, min(1.0, W - 0.1)))
        spot = free_spot(w, h)
        if spot and pool:
            cx, cy = spot
            r0, r1, c0, c1 = _rect(shape, x0, y0, cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
            hh, ww = r1 - r0, c1 - c0
            yy, xx = np.mgrid[0:hh, 0:ww]
            ell = ((xx - ww / 2) / (ww / 2)) ** 2 + ((yy - hh / 2) / (hh / 2)) ** 2
            blob = ell + 0.35 * fbm((hh, ww), rng, 60, octaves=3) < 0.9
            col_ = tile(str(rng.choice(pool)), "Color", shape, 2.0, rng, region=(r0, r1, c0, c1))
            place(blob, r0, c0, col_, 0.85, float(rng.uniform(-0.001, 0.004)))
            inserts.append({"type": "asphalt_patch", "center": [cx, cy], "size_m": [w, h]})

    # sealant lines: repaired cracks, glossy black, unlabelled
    if cracked and rng.random() < 0.35:
        for _ in range(int(rng.integers(1, 4))):
            slab = slabs[int(rng.integers(len(slabs)))]
            paths, _ = make_crack(rng, slab, str(rng.choice(["transverse", "longitudinal", "diagonal"])))
            if not paths:
                continue
            pts = np.concatenate([q.pts for q in paths])
            m = _polyline_mask(shape, x0, y0, pts, rng.uniform(0.008, 0.03))
            m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5) > 0.4
            albedo[m] = np.array([0.015, 0.015, 0.016], np.float32) * rng.uniform(0.8, 1.6)
            rough[m] = rng.uniform(0.2, 0.4)
            height[m] = np.maximum(height[m], 0.0) + 0.001
            covered |= m
            distractors.append({"type": "sealant", "xy": np.round(pts[::40], 4).tolist()})

    # leaves sit on top of everything
    leaf_cols = [(0.35, 0.22, 0.09), (0.55, 0.42, 0.12), (0.18, 0.28, 0.08),
                 (0.45, 0.15, 0.06)]
    for _ in range(p.n_leaves):
        cx, cy = int(rng.integers(nx)), int(rng.integers(ny))
        ax = (int(rng.uniform(15, 40)), int(rng.uniform(8, 20)))
        g = np.zeros(shape, np.uint8)
        cv2.ellipse(g, (cx, cy), ax, float(rng.uniform(0, 180)), 0, 360, 1, -1)
        m = g > 0
        col_ = np.asarray(leaf_cols[int(rng.integers(len(leaf_cols)))], np.float32) * rng.uniform(0.8, 1.2)
        albedo[m] = col_
        rough[m] = 0.5
        height[m] = np.maximum(height[m], 0.0005)
        covered |= m

    labels[covered] = 0       # a covered crack is not visible
    width01[covered] = 0

    albedo = np.clip(albedo, 0, 1).astype(np.float32)
    rough = np.clip(rough, 0.05, 1.0).astype(np.float32)
    counts = np.bincount(labels.ravel(), minlength=len(records) + 1)
    for rec in records:
        rec["visible_raster_cells"] = int(counts[rec["instance_id"]])

    street = sample_street(seed, p, inserts)
    return Scene(p, x0, y0, height, albedo, rough, labels, width01, records,
                 centerlines, joints, inserts, distractors, street)


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
    cv2.imwrite(os.path.join(out_dir, "normal.png"), cv2.cvtColor(flip(nm), cv2.COLOR_RGB2BGR))
    np.save(os.path.join(out_dir, "height_mesh.npy"), hm)
    np.savez_compressed(os.path.join(out_dir, "labels.npz"),
                        labels=scene.labels, width01mm=scene.width01mm)
    meta = {
        "scene_id": f"scene_{scene.params.seed:05d}",
        "version": 2,
        "params": asdict(scene.params),
        "res_m": RES_M, "mesh_cell_m": MESH_CELL_M,
        "x0": scene.x0, "y0": scene.y0,
        "length_m": scene.params.length_m, "width_m": scene.params.width_m,
        "raster_shape": list(scene.labels.shape),
        "mesh_shape": list(hm.shape),
        "labels_layout": "labels.npz['labels'|'width01mm'][iy, ix], NOT flipped",
        "texture_layout": "images flipped: row 0 = max y; u=(x-x0)/L, v=(y-y0)/W",
        "joints": scene.joints,
        "inserts": scene.inserts,
        "distractors": scene.distractors,
        "street": scene.street,
        "cracks": scene.cracks,
        "centerlines": scene.centerlines,
    }
    with open(os.path.join(out_dir, "scene.json"), "w") as fh:
        json.dump(meta, fh)
    if preview:
        s = 5
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
    surfaces = set()
    for seed in (1, 2, 3, 4, 5, 6):
        sc = generate_scene(seed)
        surfaces.add(sc.params.surface)
        n_total += len(sc.cracks)
        assert sc.labels.shape == sc.albedo.shape[:2] == sc.height.shape
        assert ((sc.labels > 0) == (sc.width01mm > 0)).all(), "width raster must track labels"
        if sc.params.surface == "pavers":
            assert not sc.cracks, "pavers carry no cracks"
        for r in sc.cracks:
            assert r["pixel_count"] > 0, r
            ratio = r["area_over_length_m"] / r["mean_width_m"]
            floor = RES_M / r["mean_width_m"]   # 1-px minimum for hairlines
            assert 0.5 < ratio < max(1.6, 1.3 * floor + 0.5), (seed, r["defect_id"], ratio)
            assert MAX_DEPTH_M + 1e-9 >= r["max_depth_m"] >= r["mean_depth_m"] > 0
            assert r["max_width_m"] >= r["median_width_m"] > 0
            rr0, cc0, rr1, cc1 = r["grid_bbox_rc"]
            win = sc.labels[rr0:rr1 + 1, cc0:cc1 + 1] == r["instance_id"]
            hwin = sc.height[rr0:rr1 + 1, cc0:cc1 + 1]
            if win.sum() > 20 and (~win).any():
                assert hwin[win].mean() < np.median(hwin[~win]), (seed, r["defect_id"])
        for ins in sc.inserts:   # nothing labelled under a cover
            if "center" in ins and ins["type"] in ("manhole", "utility_lid"):
                cx, cy = ins["center"]
                ix, iy = int((cx - sc.x0) / RES_M), int((cy - sc.y0) / RES_M)
                assert sc.labels[iy, ix] == 0
    print(f"self-test OK: 6 scenes ({', '.join(sorted(surfaces))}), {n_total} cracks")


if __name__ == "__main__":
    _self_test()
