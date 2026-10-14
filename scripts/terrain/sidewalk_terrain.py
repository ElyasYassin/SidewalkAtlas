"""
sidewalk_terrain.py

Parametric sidewalk heightfield synthesis with exact ground-truth labels.

Runs OFFLINE -- pure numpy/scipy, no Isaac dependency, so it can be iterated
and tested without launching the simulator.

Why parametric
--------------
The asset library in `sidewalks/` holds 17 defect meshes. Placing them with
random yaw and scale jitter re-presents the same 17 height profiles; it does
not create new roughness. A model trained on that memorises 17 shapes. Here the
surface is authored from parameters, so the label IS the parameter -- if we
synthesise an 8.2 mm joint fault, ground truth is 8.2 mm exactly. No raycast,
no plane fitting, no confidence gating.

Modelled failure modes (how concrete sidewalks actually fail):
  - slab joint faulting      differential settlement between adjacent panels
  - tree root heave          smooth dome uplift
  - settlement depressions   broad low spots
  - slab curl                panel edges lifting as the top dries faster
  - cracks                   narrow channels
  - spalling                 high-frequency surface texture
  - grade / cross slope      ADA regulates these (5% / 2%)

Two label families are kept SEPARATE on purpose:

  vertical discontinuity -- the abrupt step across a joint or crack edge. This
      is what ADA/PROWAG regulates: >6.4 mm (1/4") requires beveling, >13 mm
      (1/2") is a barrier.

  roughness -- detrended peak-to-peak and RMS over a rolling window, matching
      the metric already computed by spot_walk_with_evenness_autosave.py.

A 13 mm step and a 13 mm smooth dome are different hazards with different
detectability. Collapsing them into one scalar throws away the distinction that
matters for compliance.

Coordinates: x = along track (m), y = cross track (m), z = height (m).
Heights are carried internally in MILLIMETRES and converted on export.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, grey_closing


# =====================================================================
# ADA / PROWAG THRESHOLDS
# =====================================================================

ADA_BEVEL_MM = 6.4    # 1/4 in -- above this a vertical change must be beveled
ADA_BARRIER_MM = 13.0  # 1/2 in -- above this it is a barrier requiring repair

ADA_CLASSES = ("compliant", "bevel_required", "barrier")


def ada_class(discontinuity_mm: float) -> str:
    """ADA class for a vertical discontinuity, in mm."""
    if discontinuity_mm < ADA_BEVEL_MM:
        return "compliant"
    if discontinuity_mm < ADA_BARRIER_MM:
        return "bevel_required"
    return "barrier"


# =====================================================================
# PARAMETERS
# =====================================================================

@dataclass
class SidewalkParams:
    """Generative distribution for one sidewalk surface.

    Ranges are (low, high) and sampled uniformly unless noted. Defaults are
    a moderately distressed residential walk; `sample_params()` randomises
    across the full study range.
    """

    # --- extent and resolution -------------------------------------
    length_m: float = 12.0
    width_m: float = 1.8
    cell_m: float = 0.01          # 1 cm grid; bounds the finest resolvable feature

    # --- global form (ADA: 5% running grade, 2% cross slope) -------
    running_grade_pct: float = 0.0
    cross_slope_pct: float = 0.0

    # --- slab joints -----------------------------------------------
    slab_length_m: float = 1.35    # typical scored panel length
    slab_length_jitter_m: float = 0.10
    joint_width_m: float = 0.02
    joint_groove_depth_mm: float = 5.0
    # Fraction of joints that are faulted at all, and by how much.
    joint_fault_prob: float = 0.45
    joint_fault_mm_range: Tuple[float, float] = (2.0, 22.0)

    # --- tree root heave -------------------------------------------
    n_heaves: int = 1
    heave_amp_mm_range: Tuple[float, float] = (5.0, 40.0)
    heave_radius_m_range: Tuple[float, float] = (0.35, 1.10)

    # --- settlement depressions ------------------------------------
    n_depressions: int = 1
    depression_amp_mm_range: Tuple[float, float] = (4.0, 25.0)
    depression_radius_m_range: Tuple[float, float] = (0.60, 1.60)

    # --- slab curl (edges lift relative to panel centre) -----------
    curl_mm_range: Tuple[float, float] = (0.0, 6.0)

    # --- cracks ----------------------------------------------------
    n_cracks: int = 2
    crack_width_m_range: Tuple[float, float] = (0.02, 0.05)
    crack_depth_mm_range: Tuple[float, float] = (3.0, 18.0)
    crack_skew_range: Tuple[float, float] = (-0.6, 0.6)   # dx/dy, 0 = square across

    # --- surface texture -------------------------------------------
    spall_rms_mm: float = 0.8
    spall_correlation_m: float = 0.04

    def to_dict(self) -> Dict:
        return asdict(self)


def sample_params(rng: np.random.Generator,
                  length_m: float = 12.0,
                  width_m: float = 1.8,
                  cell_m: float = 0.01,
                  severity: Optional[float] = None) -> SidewalkParams:
    """Draw a randomised parameter set.

    `severity` in [0,1] biases the surface from near-pristine to badly
    distressed. Pass None to draw it uniformly. Sampling severity explicitly
    is how the dataset gets a controlled class balance instead of whatever
    the defaults happen to produce.
    """
    if severity is None:
        severity = float(rng.uniform(0.0, 1.0))
    s = float(np.clip(severity, 0.0, 1.0))

    def lerp(lo, hi):
        return lo + (hi - lo) * s

    return SidewalkParams(
        length_m=length_m,
        width_m=width_m,
        cell_m=cell_m,

        running_grade_pct=float(rng.uniform(-5.0, 5.0)),
        cross_slope_pct=float(rng.uniform(-2.0, 2.0)),

        slab_length_m=float(rng.uniform(1.15, 1.55)),
        slab_length_jitter_m=float(rng.uniform(0.02, 0.15)),
        joint_width_m=float(rng.uniform(0.012, 0.035)),
        joint_groove_depth_mm=float(rng.uniform(2.0, 8.0)),
        joint_fault_prob=float(np.clip(lerp(0.05, 0.85), 0.0, 1.0)),
        joint_fault_mm_range=(1.0, float(lerp(4.0, 30.0))),

        n_heaves=int(rng.integers(0, 1 + int(round(lerp(1, 3))))),
        heave_amp_mm_range=(2.0, float(lerp(8.0, 55.0))),
        heave_radius_m_range=(0.30, 1.20),

        n_depressions=int(rng.integers(0, 1 + int(round(lerp(1, 3))))),
        depression_amp_mm_range=(2.0, float(lerp(6.0, 30.0))),
        depression_radius_m_range=(0.50, 1.70),

        curl_mm_range=(0.0, float(lerp(1.0, 9.0))),

        n_cracks=int(rng.integers(0, 1 + int(round(lerp(1, 5))))),
        crack_width_m_range=(0.015, 0.06),
        crack_depth_mm_range=(2.0, float(lerp(6.0, 25.0))),
        crack_skew_range=(-0.6, 0.6),

        spall_rms_mm=float(lerp(0.15, 2.2)),
        spall_correlation_m=float(rng.uniform(0.025, 0.07)),
    )


# =====================================================================
# SURFACE
# =====================================================================

@dataclass
class SidewalkSurface:
    """A generated surface plus the exact record of what was put into it."""

    height_mm: np.ndarray              # (nx, ny) heights in mm
    x_m: np.ndarray                    # (nx,) along-track coordinate
    y_m: np.ndarray                    # (ny,) cross-track coordinate
    params: SidewalkParams
    features: List[Dict] = field(default_factory=list)
    seed: Optional[int] = None

    @property
    def cell_m(self) -> float:
        return self.params.cell_m

    def height_m(self) -> np.ndarray:
        return self.height_mm / 1000.0

    def lane_profile(self, y_offset_m: float) -> np.ndarray:
        """Height profile (mm) along the track at a cross-track offset from
        the centreline. Offsets outside the walk are clamped to the edge."""
        centre = 0.5 * (self.y_m[0] + self.y_m[-1])
        target = centre + y_offset_m
        j = int(np.argmin(np.abs(self.y_m - target)))
        return self.height_mm[:, j]


# =====================================================================
# SYNTHESIS
# =====================================================================

def _slab_edges(params: SidewalkParams, rng: np.random.Generator) -> np.ndarray:
    """Along-track positions of the scored joints, with spacing jitter."""
    edges = [0.0]
    x = 0.0
    while x < params.length_m:
        step = params.slab_length_m + rng.normal(0.0, params.slab_length_jitter_m)
        step = float(np.clip(step, 0.5, 3.0))
        x += step
        edges.append(x)
    return np.asarray(edges, dtype=float)


def generate_surface(params: SidewalkParams,
                     rng: np.random.Generator,
                     seed: Optional[int] = None) -> SidewalkSurface:
    """Synthesise one sidewalk heightfield from `params`."""

    nx = max(2, int(round(params.length_m / params.cell_m)) + 1)
    ny = max(2, int(round(params.width_m / params.cell_m)) + 1)

    x = np.linspace(0.0, params.length_m, nx)
    y = np.linspace(0.0, params.width_m, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    height = np.zeros((nx, ny), dtype=float)
    features: List[Dict] = []

    # ---------------------------------------------------------------
    # Global form. Kept separate from roughness: the label pipeline
    # detrends, so grade must be removable rather than read as relief.
    # ---------------------------------------------------------------
    height += X * (params.running_grade_pct / 100.0) * 1000.0
    height += Y * (params.cross_slope_pct / 100.0) * 1000.0
    features.append({
        "type": "global_form",
        "running_grade_pct": params.running_grade_pct,
        "cross_slope_pct": params.cross_slope_pct,
    })

    # ---------------------------------------------------------------
    # Slab joints and differential settlement.
    #
    # Each joint gets an independently sampled step; slab levels are the
    # cumulative sum. Signs are random, so the levels random-walk rather
    # than drifting monotonically, and every authored step stays exact --
    # which matters because the step IS the ADA label.
    # ---------------------------------------------------------------
    edges = _slab_edges(params, rng)
    n_slabs = len(edges) - 1

    steps = np.zeros(n_slabs + 1, dtype=float)
    for i in range(1, n_slabs):
        if rng.random() < params.joint_fault_prob:
            lo, hi = params.joint_fault_mm_range
            steps[i] = rng.uniform(lo, hi) * rng.choice([-1.0, 1.0])

    levels = np.cumsum(steps)[:n_slabs]

    slab_index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_slabs - 1)
    height += levels[slab_index][:, None]

    for i in range(1, n_slabs):
        if steps[i] != 0.0:
            features.append({
                "type": "joint_fault",
                "x_m": float(edges[i]),
                "step_mm": float(abs(steps[i])),
                "signed_step_mm": float(steps[i]),
                "ada_class": ada_class(abs(steps[i])),
            })

    # ---------------------------------------------------------------
    # Slab curl -- edges lift relative to the panel centre.
    # ---------------------------------------------------------------
    curl_lo, curl_hi = params.curl_mm_range
    if curl_hi > 0.0:
        for i in range(n_slabs):
            curl = float(rng.uniform(curl_lo, curl_hi))
            if curl <= 0.0:
                continue
            a, b = edges[i], edges[i + 1]
            in_slab = (x >= a) & (x < b)
            if not in_slab.any():
                continue
            u = np.zeros_like(x)
            u[in_slab] = (x[in_slab] - a) / max(b - a, 1e-6)
            shape = (2.0 * u - 1.0) ** 2
            shape[~in_slab] = 0.0
            height += (curl * shape)[:, None]
        features.append({
            "type": "slab_curl",
            "max_mm": float(curl_hi),
            "n_slabs": int(n_slabs),
        })

    # ---------------------------------------------------------------
    # Joint grooves -- the tooled channel itself.
    # ---------------------------------------------------------------
    for i in range(1, n_slabs):
        xe = edges[i]
        band = np.abs(x - xe) < (params.joint_width_m / 2.0)
        if band.any():
            height[band, :] -= params.joint_groove_depth_mm

    # ---------------------------------------------------------------
    # Tree root heave -- smooth domes.
    # ---------------------------------------------------------------
    for _ in range(max(0, params.n_heaves)):
        amp = float(rng.uniform(*params.heave_amp_mm_range))
        radius = float(rng.uniform(*params.heave_radius_m_range))
        cx = float(rng.uniform(0.1 * params.length_m, 0.9 * params.length_m))
        cy = float(rng.uniform(0.0, params.width_m))
        sigma = max(radius / 2.0, 1e-3)
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        height += amp * np.exp(-d2 / (2.0 * sigma ** 2))
        features.append({
            "type": "root_heave",
            "x_m": cx, "y_m": cy,
            "amplitude_mm": amp, "radius_m": radius,
        })

    # ---------------------------------------------------------------
    # Settlement depressions -- broad low spots.
    # ---------------------------------------------------------------
    for _ in range(max(0, params.n_depressions)):
        amp = float(rng.uniform(*params.depression_amp_mm_range))
        radius = float(rng.uniform(*params.depression_radius_m_range))
        cx = float(rng.uniform(0.1 * params.length_m, 0.9 * params.length_m))
        cy = float(rng.uniform(0.0, params.width_m))
        sigma = max(radius / 2.0, 1e-3)
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        height -= amp * np.exp(-d2 / (2.0 * sigma ** 2))
        features.append({
            "type": "settlement_depression",
            "x_m": cx, "y_m": cy,
            "amplitude_mm": amp, "radius_m": radius,
        })

    # ---------------------------------------------------------------
    # Cracks -- narrow channels, optionally skewed across the walk.
    # ---------------------------------------------------------------
    for _ in range(max(0, params.n_cracks)):
        width = float(rng.uniform(*params.crack_width_m_range))
        depth = float(rng.uniform(*params.crack_depth_mm_range))
        skew = float(rng.uniform(*params.crack_skew_range))
        cx = float(rng.uniform(0.05 * params.length_m, 0.95 * params.length_m))
        # Crack centreline drifts with y according to the skew.
        line_x = cx + skew * (Y - 0.5 * params.width_m)
        channel = np.abs(X - line_x) < (width / 2.0)
        height[channel] -= depth
        features.append({
            "type": "crack",
            "x_m": cx, "width_m": width,
            "depth_mm": depth, "skew": skew,
        })

    # ---------------------------------------------------------------
    # Spalling -- correlated surface texture. Applied last so it rides
    # on top of the structural features.
    # ---------------------------------------------------------------
    if params.spall_rms_mm > 0.0:
        sigma_cells = max(params.spall_correlation_m / params.cell_m, 0.5)
        noise = rng.normal(0.0, 1.0, size=(nx, ny))
        noise = gaussian_filter(noise, sigma=sigma_cells, mode="reflect")
        std = float(noise.std())
        if std > 1e-9:
            noise *= params.spall_rms_mm / std
        height += noise
        features.append({
            "type": "spalling",
            "rms_mm": float(params.spall_rms_mm),
            "correlation_m": float(params.spall_correlation_m),
        })

    return SidewalkSurface(
        height_mm=height,
        x_m=x,
        y_m=y,
        params=params,
        features=features,
        seed=seed,
    )


# =====================================================================
# CONCATENATION
# =====================================================================

def _fit_plane(height_mm: np.ndarray,
               x_m: np.ndarray,
               y_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares plane through a heightfield. Returns (plane, coeffs)."""
    X, Y = np.meshgrid(x_m, y_m, indexing="ij")
    A = np.column_stack([X.ravel(), Y.ravel(), np.ones(height_mm.size)])
    coeff, *_ = np.linalg.lstsq(A, height_mm.ravel(), rcond=None)
    return (A @ coeff).reshape(height_mm.shape), coeff


def concatenate_surfaces(surfaces: Sequence[SidewalkSurface],
                         running_grade_pct: float = 0.0,
                         cross_slope_pct: float = 0.0,
                         blend_m: float = 0.30,
                         ) -> Tuple[SidewalkSurface, List[Dict]]:
    """Join surfaces end-to-end into one continuous walkway.

    A real sidewalk is one walkway made of many slabs, so this is how the
    robot should meet it: a single long strip, walked once, rather than N
    short strips with a stage rebuild between each. It also removes almost
    all the per-run setup cost.

    SEAM HANDLING is the whole difficulty. Butting two independently
    generated surfaces together would leave an artificial step at every join
    -- geometry present in neither surface's labels, and exactly the kind of
    join-between-objects artefact that already contaminated the city-scene
    runs. So each surface is RE-LEVELLED: its own best-fit plane is removed,
    leaving only relief, and one global grade and cross slope are applied to
    the finished walkway.

    Dropping per-surface grade costs nothing, because the labels are
    grade-invariant by construction -- `_detrended_metrics` removes a linear
    fit along each lane, and cross slope shifts a lane's profile by a
    constant. A discontinuous grade at every seam, by contrast, is
    unphysical: real sidewalks change slope gradually.

    What remains at a seam is the difference between two independent
    realisations of surface texture, on the order of the spalling RMS (~1-2
    mm) -- below the 6.4 mm ADA threshold. Windows spanning a seam are still
    flagged, since they mix two parameter draws and would muddy provenance
    when splitting train from test.

    Returns the combined surface and a list of segment records.
    """
    if not surfaces:
        raise ValueError("no surfaces to concatenate")

    first = surfaces[0]
    cell = first.params.cell_m
    ny = first.height_mm.shape[1]
    for s in surfaces[1:]:
        if abs(s.params.cell_m - cell) > 1e-9:
            raise ValueError("all surfaces must share cell_m")
        if s.height_mm.shape[1] != ny:
            raise ValueError("all surfaces must share width")

    blend = max(2, int(round(blend_m / cell)))
    segments: List[Dict] = []
    height: Optional[np.ndarray] = None

    for i, surf in enumerate(surfaces):
        relief = surf.height_mm - _fit_plane(surf.height_mm,
                                             surf.x_m, surf.y_m)[0]

        if height is None:
            height = relief.copy()
            x_start = 0.0
        else:
            # CROSS-FADE, not butt-join. Removing each surface's plane makes
            # the seam continuous in slope, but the relief on either side is
            # an independent realisation -- and near a slab's ends that
            # relief includes curl, heaves and heavy spalling. Butted
            # together, two surfaces fabricated steps up to ~10 mm, which is
            # bevel-required class: a fault invented by the joining process
            # rather than drawn from the parameter distribution. That breaks
            # the guarantee the whole approach rests on, that the parameters
            # ARE the ground truth.
            #
            # Overlapping the tail of one surface with the head of the next
            # and fading linearly between them makes the join continuous by
            # construction while keeping realistic texture throughout. The
            # cost is `blend_m` of overlap consumed per seam.
            tail = height[-blend:, :]
            head = relief[:blend, :]
            # Match levels over the overlap before fading, so the fade
            # interpolates texture rather than a step.
            relief = relief + float(tail.mean() - head.mean())
            head = relief[:blend, :]

            w = np.linspace(0.0, 1.0, blend)[:, None]
            height[-blend:, :] = tail * (1.0 - w) + head * w
            x_start = (height.shape[0] - blend) * cell
            height = np.concatenate([height, relief[blend:, :]], axis=0)

        x_end = height.shape[0] * cell

        # Shift each authored feature into walkway coordinates, so a fault
        # can still be traced back to the parameters that produced it.
        shifted = []
        for f in surf.features:
            g = dict(f)
            if "x_m" in g:
                g["x_m"] = float(g["x_m"] + x_start)
            shifted.append(g)

        segments.append({
            "segment_id": f"seg_{i:03d}",
            "source_seed": surf.seed,
            "x_start_m": float(x_start),
            "x_end_m": float(x_end),
            "params": surf.params.to_dict(),
            "features": shifted,
        })

    nx = height.shape[0]
    x = np.arange(nx) * cell
    y = first.y_m.copy()

    # One grade and cross slope for the whole walkway.
    X, Y = np.meshgrid(x, y, indexing="ij")
    height = height + X * (running_grade_pct / 100.0) * 1000.0
    height = height + Y * (cross_slope_pct / 100.0) * 1000.0

    combined_params = SidewalkParams(
        length_m=float(x[-1]),
        width_m=first.params.width_m,
        cell_m=cell,
        running_grade_pct=running_grade_pct,
        cross_slope_pct=cross_slope_pct,
    )

    all_features: List[Dict] = []
    for seg in segments:
        all_features.extend(seg["features"])

    walkway = SidewalkSurface(
        height_mm=height,
        x_m=x,
        y_m=y,
        params=combined_params,
        features=all_features,
        seed=first.seed,
    )
    return walkway, segments


def _step_at_index(profile_mm: np.ndarray,
                   cell_m: float,
                   index: int,
                   baseline_m: float = 0.15,
                   gap_m: float = 0.03) -> float:
    """Extrapolated level change at ONE specific position.

    Same estimator as `_max_discontinuity_mm` -- fit a line either side,
    extrapolate both to the boundary, difference them -- but evaluated at a
    single index instead of maximised over a window.
    """
    k = max(2, int(round(baseline_m / cell_m)))
    g = max(1, int(round(gap_m / cell_m)))
    lo, hi = index - g - k, index + g + k
    if lo < 0 or hi > profile_mm.size:
        return 0.0
    wl = _extrapolation_weights(np.arange(-(g + k), -g))
    wr = _extrapolation_weights(np.arange(g, g + k))
    left = float(profile_mm[index - g - k:index - g] @ wl)
    right = float(profile_mm[index + g:index + g + k] @ wr)
    return abs(right - left)


def seam_steps_mm(walkway: SidewalkSurface,
                  segments: Sequence[Dict],
                  lane_offsets_m: Sequence[float] = (-0.25, 0.0, 0.25),
                  ) -> List[float]:
    """Level change introduced AT each join.

    Evaluated exactly at the seam index, not maximised over a window around
    it. That distinction is the difference between measuring the join and
    measuring the terrain near the join: slabs sit ~1.35 m apart and faults
    reach 30 mm, so any window wide enough to be robust usually contains a
    genuine authored fault, and reports it as a seam defect.

    Two earlier versions of this function measured the wrong thing -- first
    by differencing strip means (which reads walkway grade as a step), then
    by maximising over a +/-0.5 m window (which reads nearby real faults as
    seam artefacts). Both made concatenation look broken when it was not.

    Judge the output against a CONTROL, never against an absolute number:
    run the same estimator at random non-seam positions. Measured on a
    12-segment, 140 m walkway, seams came out at mean 2.74 / median 2.09 mm
    against random positions at mean 3.68 / median 2.69 mm (max 23.4, where
    real faults are). Seams are quieter than ordinary terrain, which is the
    result that matters -- a few millimetres sounds alarming until you learn
    the surface itself does more than that everywhere.
    """
    cell = walkway.params.cell_m
    steps = []
    for seg in segments[1:]:
        i = int(round(seg["x_start_m"] / cell))
        worst = 0.0
        for off in lane_offsets_m:
            worst = max(worst, _step_at_index(walkway.lane_profile(off),
                                              cell, i))
        steps.append(float(worst))
    return steps


def tag_windows(records: List[Dict],
                segments: Sequence[Dict]) -> List[Dict]:
    """Attach segment provenance to each label window.

    `segment_id` is the train/test split key: two windows from the same
    segment share a parameter draw and the same authored faults, so they are
    not independent however far apart they sit. `spans_seam` marks windows
    straddling a join, which mix two draws.
    """
    for r in records:
        lo, hi = r["window_start_m"], r["window_end_m"]
        hits = [s for s in segments
                if s["x_start_m"] < hi and s["x_end_m"] > lo]
        r["segment_id"] = hits[0]["segment_id"] if hits else None
        r["spans_seam"] = int(len(hits) > 1)
    return records


# =====================================================================
# LABELS
# =====================================================================

def _detrended_metrics(profile_mm: np.ndarray,
                       station_m: np.ndarray) -> Tuple[float, float]:
    """Peak-to-peak and RMS after removing best-fit grade.

    Mirrors _detrended_height_metrics() in
    spot_walk_with_evenness_autosave.py so sim labels and the raycast
    profiler speak the same language.
    """
    if profile_mm.size < 2:
        return 0.0, 0.0
    coeff = np.polyfit(station_m, profile_mm, 1)
    residual = profile_mm - np.polyval(coeff, station_m)
    return float(np.ptp(residual)), float(np.sqrt(np.mean(residual ** 2)))


def _extrapolation_weights(offsets: np.ndarray) -> np.ndarray:
    """Weights that least-squares fit a line to samples at `offsets` and
    evaluate it at offset 0. Fixed for a given window geometry, so the whole
    profile can be processed with two dot products."""
    d = offsets.astype(float)
    dbar = d.mean()
    dev = d - dbar
    denom = float(np.sum(dev ** 2))
    slope_w = dev / denom if denom > 1e-12 else np.zeros_like(dev)
    # value at 0 = mean(y) - slope * mean(d)
    return (np.ones_like(d) / d.size) - dbar * slope_w


def _max_discontinuity_mm(profile_mm: np.ndarray,
                          cell_m: float,
                          baseline_m: float = 0.15,
                          gap_m: float = 0.03,
                          notch_fill_m: float = 0.10) -> float:
    """Largest change in level between adjacent walking surfaces.

    This is the ADA-regulated quantity, and getting it right requires
    rejecting two things that are NOT level changes:

    1. **The tooled joint groove.** Every sidewalk has scored control joints.
       Their walls are abrupt, but the surface returns to the same level on
       the far side, so a groove is a void, not a change in level. A naive
       before/after difference reports the groove depth at every joint --
       which would label every normal sidewalk joint as a hazard. Narrow
       notches (grooves, cracks) are filled by a grey-scale morphological
       closing before anything is measured; the structuring element is wider
       than any authored notch and narrower than any real surface feature, so
       slabs, domes and depressions pass through untouched. A step is
       invariant under closing, so a faulted joint keeps its full height even
       though its groove is filled.

    2. **Smooth relief.** A root heave is steep but continuous. Fitting a
       line to each side and extrapolating both to the boundary means a
       smooth surface predicts itself across the gap and cancels, while a
       genuine fault leaves the two extrapolations offset by the step.

    What survives is slab-to-slab faulting -- the thing that trips people.
    Smooth relief is still captured, as roughness, by the detrended metrics.
    """
    n = profile_mm.size
    k = max(2, int(round(baseline_m / cell_m)))
    g = max(1, int(round(gap_m / cell_m)))
    if n < 2 * (k + g) + 1:
        return 0.0

    fill = max(3, int(round(notch_fill_m / cell_m)))
    profile_mm = grey_closing(profile_mm, size=fill, mode="nearest")

    wl = _extrapolation_weights(np.arange(-(g + k), -g))
    wr = _extrapolation_weights(np.arange(g, g + k))

    windows = np.lib.stride_tricks.sliding_window_view(profile_mm, k)
    boundaries = np.arange(g + k, n - g - k + 1)
    if boundaries.size == 0:
        return 0.0

    left = windows[boundaries - g - k] @ wl
    right = windows[boundaries + g] @ wr
    return float(np.max(np.abs(right - left)))


def compute_labels(surface: SidewalkSurface,
                   window_m: float = 1.0,
                   stride_m: float = 0.10,
                   lane_offsets_m: Sequence[float] = (-0.25, 0.0, 0.25),
                   ) -> List[Dict]:
    """Rolling-window ground truth along the walk.

    One record per station. Roughness metrics are computed per lane and
    reduced across lanes, because Spot's feet sample a corridor rather than
    a single line. Vertical discontinuity is reported as the worst case
    across lanes -- a step only one foot track crosses is still a trip
    hazard.
    """
    cell = surface.cell_m
    win_cells = max(2, int(round(window_m / cell)))
    stride_cells = max(1, int(round(stride_m / cell)))
    x = surface.x_m
    nx = x.size

    lanes = [surface.lane_profile(off) for off in lane_offsets_m]

    records: List[Dict] = []
    for start in range(0, max(1, nx - win_cells + 1), stride_cells):
        stop = start + win_cells
        station = x[start:stop]

        ptps, rmss, discs = [], [], []
        for prof in lanes:
            seg = prof[start:stop]
            ptp, rms = _detrended_metrics(seg, station)
            ptps.append(ptp)
            rmss.append(rms)
            discs.append(_max_discontinuity_mm(seg, cell))

        worst_disc = float(np.max(discs))
        records.append({
            "window_start_m": float(station[0]),
            "window_end_m": float(station[-1]),
            "window_centre_m": float(0.5 * (station[0] + station[-1])),

            # Roughness -- matches the raycast profiler's definition.
            "ptp_mm_mean": float(np.mean(ptps)),
            "ptp_mm_max": float(np.max(ptps)),
            "rms_mm_mean": float(np.mean(rmss)),
            "rms_mm_max": float(np.max(rmss)),

            # Discontinuity -- the ADA-regulated quantity.
            "discontinuity_mm": worst_disc,
            "discontinuity_mm_per_lane": [float(d) for d in discs],
            "ada_class": ada_class(worst_disc),
        })

    return records


def label_summary(records: Sequence[Dict]) -> Dict:
    """Aggregate a label set -- used for dataset-level class balance."""
    if not records:
        return {"n_windows": 0}
    counts = {c: 0 for c in ADA_CLASSES}
    for r in records:
        counts[r["ada_class"]] += 1
    disc = np.asarray([r["discontinuity_mm"] for r in records])
    ptp = np.asarray([r["ptp_mm_max"] for r in records])
    return {
        "n_windows": len(records),
        "ada_class_counts": counts,
        "discontinuity_mm": {
            "min": float(disc.min()), "max": float(disc.max()),
            "mean": float(disc.mean()), "p95": float(np.percentile(disc, 95)),
        },
        "ptp_mm": {
            "min": float(ptp.min()), "max": float(ptp.max()),
            "mean": float(ptp.mean()), "p95": float(np.percentile(ptp, 95)),
        },
    }


# =====================================================================
# MESH EXPORT
# =====================================================================

def write_obj(surface: SidewalkSurface, path: str, stride: int = 1) -> Dict:
    """Write the heightfield as a triangulated OBJ.

    OBJ rather than USD so this module stays Isaac-free -- `pxr` only exists
    inside the Isaac Python environment. import_terrain_isaac.py brings the
    mesh in and attaches collision.

    `stride` decimates the exported mesh without touching the label grid: a
    1 cm label resolution with a 2 cm mesh keeps files manageable while the
    ground truth stays fine.
    """
    h = surface.height_mm[::stride, ::stride] / 1000.0
    xs = surface.x_m[::stride]
    ys = surface.y_m[::stride]
    nx, ny = h.shape

    lines = [
        "# SidewalkAtlas parametric sidewalk",
        f"# seed={surface.seed} cells={nx}x{ny} stride={stride}",
        "",
    ]
    # Z-up to match the Isaac scene.
    for i in range(nx):
        xi = xs[i]
        for j in range(ny):
            lines.append(f"v {xi:.5f} {ys[j]:.5f} {h[i, j]:.6f}")

    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j + 1
            b = a + 1
            c = a + ny
            d = c + 1
            lines.append(f"f {a} {c} {b}")
            lines.append(f"f {b} {c} {d}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")

    return {
        "path": path,
        "vertices": int(nx * ny),
        "faces": int(2 * (nx - 1) * (ny - 1)),
        "stride": int(stride),
    }


# =====================================================================
# SELF TEST
# =====================================================================

def _self_test() -> None:
    """Verify that authored features come back out of the labels.

    The whole premise is that the parameter IS the ground truth, so the one
    thing worth asserting is that a step we author is a step the labeller
    recovers.
    """
    rng = np.random.default_rng(0)

    def clean(**over) -> SidewalkParams:
        """A pristine surface; override one thing at a time."""
        base = dict(
            length_m=6.0, width_m=1.5, cell_m=0.01,
            running_grade_pct=0.0, cross_slope_pct=0.0,
            slab_length_m=2.0, slab_length_jitter_m=0.0,
            joint_width_m=0.02, joint_groove_depth_mm=0.0,
            joint_fault_prob=0.0, joint_fault_mm_range=(0.0, 0.0),
            n_heaves=0, n_depressions=0, curl_mm_range=(0.0, 0.0),
            n_cracks=0, spall_rms_mm=0.0,
        )
        base.update(over)
        return SidewalkParams(**base)

    # 1. An authored fault must come back out at full height.
    surf = generate_surface(
        clean(joint_fault_prob=1.0, joint_fault_mm_range=(12.0, 12.0)),
        rng, seed=0)
    faults = [f for f in surf.features if f["type"] == "joint_fault"]
    assert faults, "expected at least one authored fault"
    authored = max(f["step_mm"] for f in faults)
    recovered = max(r["discontinuity_mm"]
                    for r in compute_labels(surf, window_m=1.0, stride_m=0.10))
    err = abs(recovered - authored)
    assert err < 0.5, f"authored {authored:.2f} mm, recovered {recovered:.2f} mm"

    # 2. Running grade and cross slope must detrend away, not read as relief.
    flat = generate_surface(
        clean(running_grade_pct=5.0, cross_slope_pct=2.0),
        np.random.default_rng(1), seed=1)
    worst_ptp = max(r["ptp_mm_max"]
                    for r in compute_labels(flat, window_m=1.0, stride_m=0.25))
    assert worst_ptp < 0.5, f"grade leaked into roughness: {worst_ptp:.3f} mm"

    # 3. A tooled joint groove is NOT a change in level. Unfaulted joints
    #    with deep grooves must stay well under the ADA bevel threshold.
    grooved = generate_surface(
        clean(joint_groove_depth_mm=8.0, joint_fault_prob=0.0),
        np.random.default_rng(2), seed=2)
    groove_disc = max(r["discontinuity_mm"]
                      for r in compute_labels(grooved, window_m=1.0, stride_m=0.10))
    assert groove_disc < ADA_BEVEL_MM, (
        f"8 mm joint groove misread as a {groove_disc:.2f} mm level change")

    # 4. A smooth root heave is roughness, not a discontinuity. It must show
    #    up in ptp and stay out of the discontinuity channel.
    heaved = generate_surface(
        clean(n_heaves=1, heave_amp_mm_range=(40.0, 40.0),
              heave_radius_m_range=(0.35, 0.35)),
        np.random.default_rng(7), seed=7)
    hrecs = compute_labels(heaved, window_m=1.0, stride_m=0.10)
    heave_disc = max(r["discontinuity_mm"] for r in hrecs)
    heave_ptp = max(r["ptp_mm_max"] for r in hrecs)
    assert heave_ptp > 10.0, f"40 mm heave barely registered as relief: {heave_ptp:.1f} mm"
    assert heave_disc < ADA_BEVEL_MM, (
        f"smooth heave misread as a {heave_disc:.2f} mm level change")

    # 5. Concatenation must not invent a defect at the seams. Surfaces with
    #    deliberately clashing grades are the hard case: butted together
    #    naively they would leave a large step and a slope kink at every join.
    parts = []
    for k, (grade, cross) in enumerate([(5.0, 2.0), (-4.0, -1.5), (3.0, 0.5)]):
        parts.append(generate_surface(
            clean(length_m=4.0, running_grade_pct=grade,
                  cross_slope_pct=cross, spall_rms_mm=0.5),
            np.random.default_rng(20 + k), seed=20 + k))
    walkway, segments = concatenate_surfaces(
        parts, running_grade_pct=2.0, cross_slope_pct=1.0)
    steps = seam_steps_mm(walkway, segments)
    worst_seam = max(steps) if steps else 0.0
    assert len(segments) == 3, f"expected 3 segments, got {len(segments)}"
    assert worst_seam < ADA_BEVEL_MM, (
        f"concatenation created a {worst_seam:.2f} mm step at a seam")

    wrecs = tag_windows(
        compute_labels(walkway, window_m=1.0, stride_m=0.10), segments)
    seam_windows = sum(r["spans_seam"] for r in wrecs)
    seg_ids = {r["segment_id"] for r in wrecs}
    assert seg_ids == {"seg_000", "seg_001", "seg_002"}, seg_ids
    assert seam_windows > 0, "no windows flagged as spanning a seam"

    print(f"  fault recovery:   authored {authored:.2f} mm -> "
          f"recovered {recovered:.2f} mm (err {err:.3f} mm)")
    print(f"  grade rejection:  worst detrended ptp on a 5%/2% ramp = "
          f"{worst_ptp:.4f} mm")
    print(f"  groove rejection: 8 mm tooled joint -> "
          f"{groove_disc:.2f} mm discontinuity (< {ADA_BEVEL_MM})")
    print(f"  heave separation: 40 mm dome -> ptp {heave_ptp:.1f} mm, "
          f"discontinuity {heave_disc:.2f} mm")
    print(f"  seam continuity:  3 surfaces (+5%/-4%/+3% grades) joined, "
          f"worst seam step {worst_seam:.3f} mm (< {ADA_BEVEL_MM})")
    print(f"                    {len(wrecs)} windows, {seam_windows} flagged "
          f"spans_seam, segments {sorted(seg_ids)}")
    print("self-test OK")


if __name__ == "__main__":
    _self_test()
