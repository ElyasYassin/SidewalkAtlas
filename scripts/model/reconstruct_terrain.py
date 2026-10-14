"""
reconstruct_terrain.py

Measure a sidewalk's profile from where Spot's feet land, score it against ADA
thresholds, and plot it against the authored truth.

No model, no training: a foot in contact IS on the surface, so every stance is
a direct sample of the terrain. The robot's legs are the measuring instrument.

Runs OFFLINE on a walk CSV from spot_walk_with_evenness_autosave.py.

WHY MEASUREMENT RATHER THAN INFERENCE
-------------------------------------
The goal is an inspection record: "at 23.4 m the vertical discontinuity is
14 mm, above the 13 mm ADA barrier threshold." That is a measurement with a
stated accuracy. A learned model gives an estimate with an error distribution,
which is weaker evidence for compliance and must be retrained for every new
robot, gait and surface.

The simulator's role changes accordingly: not a source of training data, but
the CALIBRATION RIG. The terrain was authored, so the truth is known exactly
and the method's accuracy can be stated -- a claim unavailable on real
sidewalks without a survey crew.

HOW IT WORKS
------------
1. Mid-stance sampling. Each contiguous run of foot contact is a stance; its
   first and last quarter are discarded. At touchdown the foot is still
   arriving and the impact is resolving; at lift-off it is unloading. The
   middle is where the foot is planted and genuinely reports the surface.

2. Fixed frame. Body-relative foot height is nearly useless, because the body
   height tracks the terrain and the signal largely cancels. Foot positions
   are rotated into world coordinates using the logged body heading.

3. All four feet, cross-slope removed. Left and right feet sit ~0.3 m apart,
   so on a 2% cross slope they differ by ~6 mm for reasons that have nothing
   to do with roughness. Fitting and removing a PLANE in (x, y) -- not just a
   line in x -- lets all four feet contribute to one profile, doubling sample
   density. Density is what limits how sharply a step can be resolved.

4. Regular resampling. Scattered footfalls are binned along-track to a regular
   profile, so the generator's own estimators apply unchanged and measured
   numbers are directly comparable to authored ones.

The foot prim's origin sits some distance above the true contact point. That
offset is constant and cancels: both metrics work on a detrended profile, and
the ADA discontinuity is a difference of two extrapolated levels. No foot
geometry calibration is needed, in sim or on hardware.

KNOWN LIMIT
-----------
Feet land roughly every 0.5 m. A step falling between two footfalls is
reconstructed as a ramp, so sharp discontinuities are under-measured. Roughness
over a window is unaffected. Combining feet and walking a surface twice in
opposite directions both help; nothing removes the limit entirely.

Usage:
    python scripts/model/reconstruct_terrain.py --run <run_dir> \
        --walkway <walkway_sample_dir> --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "terrain"))
from sidewalk_terrain import (  # noqa: E402
    ADA_BARRIER_MM, ADA_BEVEL_MM, ada_class,
    _max_discontinuity_mm, _detrended_metrics,
)

FEET = ("fl_foot", "fr_foot", "hl_foot", "hr_foot")


# =====================================================================
# LOADING
# =====================================================================

def load_run(path: str) -> Dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"empty CSV: {path}")
    out: Dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        if key == "collider":
            out[key] = np.array([r.get(key, "") for r in rows], dtype=object)
            continue
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(key, "")))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[key] = np.asarray(vals, dtype=float)
    return out


def heading(run: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, str]:
    """Body forward unit vector, logged if the run has it.

    Older runs predate `body_fwd_*`; the fallback uses direction of travel,
    which is NOT the same thing for a legged robot that can crab sideways --
    and the walker's coverage weave makes it do exactly that. The fallback is
    reported rather than used silently, because every foot position depends
    on it.
    """
    if "body_fwd_x" in run and np.isfinite(run["body_fwd_x"]).any():
        fx, fy = run["body_fwd_x"].copy(), run["body_fwd_y"].copy()
        norm = np.hypot(fx, fy)
        norm[norm < 1e-9] = 1.0
        return fx / norm, fy / norm, "logged body_fwd (exact)"

    bx, by = run["body_x_m"], run["body_y_m"]
    dx, dy = np.gradient(bx), np.gradient(by)
    norm = np.hypot(dx, dy)
    norm[norm < 1e-9] = 1.0
    return dx / norm, dy / norm, "APPROXIMATED from travel direction"


# =====================================================================
# STANCE SAMPLING
# =====================================================================

def stance_samples(run: Dict[str, np.ndarray],
                   feet: Sequence[str] = FEET,
                   min_stance: int = 4,
                   trim_frac: float = 0.25,
                   on_sidewalk_only: bool = True) -> np.ndarray:
    """Mid-stance foot samples. Returns (N, 5):

        along_m, lateral_m, world_z, world_x, world_y

    ALONG-TRACK IS PATH DISTANCE, not world X. A sidewalk does not have to run
    along a world axis, and a real one bends. Binning by world X silently
    folds a curving path back on itself -- on the city scene it compressed a
    24 m walk into 10.6 m of apparent extent and reported a wildly uneven
    surface that was really the same ground counted several times. Path
    distance is already logged, so the fix is free and general.

    Lateral is the foot's own body-relative offset, which is the perpendicular
    distance from the path by construction.

    World x/y are carried along only so authored truth can be looked up.
    """
    fx, fy, _ = heading(run)
    bx, by = run["body_x_m"], run["body_y_m"]
    dist = run["distance_m"]

    keep_mask = None
    if on_sidewalk_only and "on_sidewalk" in run:
        keep_mask = np.nan_to_num(run["on_sidewalk"], nan=1.0) > 0.5

    out: List[Tuple[float, float, float, float, float]] = []
    for foot in feet:
        contact = run.get(f"foot_{foot}_in_contact")
        fwd = run.get(f"foot_{foot}_x_m")
        lat = run.get(f"foot_{foot}_y_m")
        wz = run.get(f"foot_{foot}_world_z_m")
        if contact is None or wz is None or fwd is None:
            continue

        down = np.nan_to_num(contact) > 0.5
        edges = np.diff(down.astype(int))
        starts = list(np.where(edges == 1)[0] + 1)
        stops = list(np.where(edges == -1)[0] + 1)
        if down[0]:
            starts.insert(0, 0)
        if down[-1]:
            stops.append(len(down))

        for a, b in zip(starts, stops):
            n = b - a
            if n < min_stance:
                continue
            trim = int(n * trim_frac)
            for i in range(a + trim, b - trim):
                if not (np.isfinite(wz[i]) and np.isfinite(fwd[i])
                        and np.isfinite(lat[i]) and np.isfinite(dist[i])):
                    continue
                if keep_mask is not None and not keep_mask[i]:
                    continue
                lx, ly = -fy[i], fx[i]          # left = +90 deg from forward
                out.append((dist[i] + fwd[i],            # along the path
                            lat[i],                       # perpendicular
                            wz[i],
                            bx[i] + fwd[i] * fx[i] + lat[i] * lx,
                            by[i] + fwd[i] * fy[i] + lat[i] * ly))

    return np.asarray(out, dtype=float) if out else np.zeros((0, 5))


# =====================================================================
# PROFILE
# =====================================================================

def build_profile(samples: np.ndarray, bin_m: float = 0.05
                  ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Scattered footfalls -> a regular 1D relief profile along the walk.

    A plane is fitted in (x, y) and removed, so left and right feet can share
    one profile: without that, the ~0.3 m lateral separation shows up as
    several millimetres of false relief on any cross slope. Removing grade
    too is harmless -- every metric detrends anyway.
    """
    if samples.shape[0] < 10:
        return np.zeros(0), np.zeros(0), {}

    x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
    A = np.column_stack([x, y, np.ones(x.size)])
    coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
    relief = z - A @ coeff

    lo, hi = x.min(), x.max()
    edges = np.arange(lo, hi + bin_m, bin_m)
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)

    xs, zs = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(0.5 * (edges[b] + edges[b + 1]))
        zs.append(float(np.median(relief[m])))

    info = {
        "grade_pct": float(coeff[0] * 100.0),
        "cross_slope_pct": float(coeff[1] * 100.0),
        "n_samples": int(samples.shape[0]),
        "n_bins": len(xs),
        "bin_m": bin_m,
    }
    return np.asarray(xs), np.asarray(zs), info


def profile_report(x: np.ndarray, z_m: np.ndarray, cell_m: float,
                   window_m: float = 1.0, stride_m: float = 0.25
                   ) -> List[Dict]:
    """Windowed ADA metrics, using the generator's own estimators so measured
    and authored numbers mean the same thing."""
    if x.size < 10:
        return []
    z_mm = z_m * 1000.0
    win = max(4, int(round(window_m / cell_m)))
    step = max(1, int(round(stride_m / cell_m)))

    records = []
    for start in range(0, max(1, x.size - win + 1), step):
        stop = start + win
        seg_x, seg_z = x[start:stop], z_mm[start:stop]
        if seg_x.size < win:
            break
        ptp, rms = _detrended_metrics(seg_z, seg_x)
        disc = _max_discontinuity_mm(seg_z, cell_m)
        records.append({
            "centre_m": float(0.5 * (seg_x[0] + seg_x[-1])),
            "ptp_mm": float(ptp),
            "rms_mm": float(rms),
            "discontinuity_mm": float(disc),
            "ada_class": ada_class(disc),
        })
    return records


def authored_profile(walkway_dir: str, origin_x: float, origin_y: float,
                     samples: np.ndarray, bin_m: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """The authored heightfield sampled at exactly the positions the feet
    visited, then put through the identical pipeline. Anything else would
    compare two different things."""
    height = np.load(os.path.join(walkway_dir, "height.npy"))
    with open(os.path.join(walkway_dir, "labels.json"), encoding="utf-8") as fh:
        cell = float(json.load(fh)["params"]["cell_m"])

    gi = np.round((samples[:, 3] - origin_x) / cell).astype(int)
    gj = np.round((samples[:, 4] - origin_y) / cell).astype(int)
    ok = ((gi >= 0) & (gi < height.shape[0])
          & (gj >= 0) & (gj < height.shape[1]))

    truth = np.column_stack([samples[ok, 0], samples[ok, 1],
                             height[gi[ok], gj[ok]] / 1000.0,
                             samples[ok, 3], samples[ok, 4]])
    x, z, _ = build_profile(truth, bin_m)
    return x, z


# =====================================================================
# PLOT
# =====================================================================

def write_plot(path: str,
               mx: np.ndarray, mz: np.ndarray,
               tx: Optional[np.ndarray], tz: Optional[np.ndarray],
               measured: List[Dict], truth: Optional[List[Dict]],
               samples: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [2, 1, 1]})

    ax = axes[0]
    if tx is not None and tx.size:
        ax.plot(tx, tz * 1000.0, lw=1.6, color="#1565c0",
                label="authored truth", zorder=2)
    ax.plot(mx, mz * 1000.0, lw=1.2, color="#c62828",
            label="measured from footfalls", zorder=3)
    ax.set_title("Sidewalk profile: measured by Spot's feet vs authored truth")
    ax.set_ylabel("relief (mm)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot([r["centre_m"] for r in measured],
            [r["ptp_mm"] for r in measured],
            lw=1.4, color="#c62828", label="measured")
    if truth:
        ax.plot([r["centre_m"] for r in truth],
                [r["ptp_mm"] for r in truth],
                lw=1.4, color="#1565c0", label="truth")
    ax.set_title("Roughness: detrended peak-to-peak per 1 m window")
    ax.set_ylabel("mm")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    top = max([r["discontinuity_mm"] for r in measured] +
              ([r["discontinuity_mm"] for r in truth] if truth else [0]) +
              [ADA_BARRIER_MM]) * 1.15
    ax.axhspan(0, ADA_BEVEL_MM, color="#2e7d32", alpha=0.12, label="compliant")
    ax.axhspan(ADA_BEVEL_MM, ADA_BARRIER_MM, color="#f9a825", alpha=0.15,
               label="bevel required")
    ax.axhspan(ADA_BARRIER_MM, top, color="#c62828", alpha=0.12,
               label="barrier")
    ax.plot([r["centre_m"] for r in measured],
            [r["discontinuity_mm"] for r in measured],
            lw=1.6, color="#111", label="measured")
    if truth:
        ax.plot([r["centre_m"] for r in truth],
                [r["discontinuity_mm"] for r in truth],
                lw=1.4, color="#1565c0", ls="--", label="truth")
    ax.set_ylim(0, top)
    ax.set_title("Vertical discontinuity vs ADA thresholds")
    ax.set_xlabel("along walk (m)")
    ax.set_ylabel("mm")
    ax.legend(loc="upper right", fontsize=8, ncol=5)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)


# =====================================================================
# SURFACE MAP
# =====================================================================

def grid_surface(samples: np.ndarray, bin_x: float, bin_y: float
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin scattered footfalls into a 2D relief map.

    Cells with no footfall stay NaN rather than being interpolated over.
    That is deliberate: an inspection record must distinguish "surveyed and
    compliant" from "never walked on". Spot covers the width it happens to
    tread, not the whole slab.
    """
    x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
    A = np.column_stack([x, y, np.ones(x.size)])
    coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
    relief = (z - A @ coeff) * 1000.0

    xe = np.arange(x.min(), x.max() + bin_x, bin_x)
    ye = np.arange(y.min(), y.max() + bin_y, bin_y)
    gi = np.clip(np.digitize(x, xe) - 1, 0, len(xe) - 2)
    gj = np.clip(np.digitize(y, ye) - 1, 0, len(ye) - 2)

    grid = np.full((len(xe) - 1, len(ye) - 1), np.nan)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            m = (gi == i) & (gj == j)
            if m.any():
                grid[i, j] = np.median(relief[m])
    return grid, xe, ye, (gi, gj)


def write_surface_plot(path: str, samples: np.ndarray,
                       walkway_dir: Optional[str],
                       origin_x: float, origin_y: float,
                       measured: List[Dict],
                       bin_x: float = 0.10, bin_y: float = 0.20) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid, xe, ye, (gi_all, gj_all) = grid_surface(samples, bin_x, bin_y)
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    lim = float(np.nanpercentile(np.abs(grid), 98)) or 1.0

    truth_grid = None
    if walkway_dir:
        height = np.load(os.path.join(walkway_dir, "height.npy"))
        with open(os.path.join(walkway_dir, "labels.json"),
                  encoding="utf-8") as fh:
            cell = float(json.load(fh)["params"]["cell_m"])
        tg = np.full_like(grid, np.nan)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isfinite(grid[i, j]):
                    continue
                m = (gi_all == i) & (gj_all == j)
                if not m.any():
                    continue
                a = int(round((np.median(samples[m, 3]) - origin_x) / cell))
                b = int(round((np.median(samples[m, 4]) - origin_y) / cell))
                if 0 <= a < height.shape[0] and 0 <= b < height.shape[1]:
                    tg[i, j] = height[a, b]
        # Same plane removal, so the two maps are on one scale.
        ii, jj = np.where(np.isfinite(tg))
        cxs = 0.5 * (xe[ii] + xe[ii + 1])
        cys = 0.5 * (ye[jj] + ye[jj + 1])
        A = np.column_stack([cxs, cys, np.ones(cxs.size)])
        tc, *_ = np.linalg.lstsq(A, tg[ii, jj], rcond=None)
        truth_grid = np.full_like(tg, np.nan)
        truth_grid[ii, jj] = tg[ii, jj] - A @ tc

    rows = 4 if truth_grid is not None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(15, 2.4 * rows + 2))

    def show(ax, data, title, cmap="RdBu_r", vlim=None):
        v = vlim or lim
        im = ax.imshow(data.T, origin="lower", aspect="auto", cmap=cmap,
                       vmin=-v, vmax=v, extent=extent, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("cross (m)")
        fig.colorbar(im, ax=ax, label="mm", fraction=0.025, pad=0.01)

    show(axes[0], grid, "MEASURED surface from footfalls "
                        "(white = never walked on)")
    k = 1
    if truth_grid is not None:
        show(axes[k], truth_grid,
             "AUTHORED truth, same cells"); k += 1
        diff = grid - truth_grid
        show(axes[k], diff, f"ERROR (measured - truth), "
                            f"std {np.nanstd(diff):.2f} mm",
             vlim=max(3.0, float(np.nanpercentile(np.abs(diff), 95)))); k += 1

    ax = axes[k]
    cx = [r["centre_m"] for r in measured]
    dv = [r["discontinuity_mm"] for r in measured]
    top = max(max(dv, default=0), ADA_BARRIER_MM) * 1.15
    ax.axhspan(0, ADA_BEVEL_MM, color="#2e7d32", alpha=0.12)
    ax.axhspan(ADA_BEVEL_MM, ADA_BARRIER_MM, color="#f9a825", alpha=0.15)
    ax.axhspan(ADA_BARRIER_MM, top, color="#c62828", alpha=0.12)
    ax.plot(cx, dv, lw=1.5, color="#111")
    bar = [(c, v) for c, v in zip(cx, dv) if v >= ADA_BARRIER_MM]
    if bar:
        ax.scatter([b[0] for b in bar], [b[1] for b in bar], s=26,
                   color="#c62828", zorder=5,
                   label=f"{len(bar)} barrier-class windows")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, top)
    ax.set_xlim(extent[0], extent[1])
    ax.set_title("Hazard map: vertical discontinuity vs ADA thresholds",
                 fontsize=10)
    ax.set_xlabel("along walk (m)")
    ax.set_ylabel("mm")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)


def per_segment_accuracy(samples: np.ndarray, walkway_dir: str,
                         origin_x: float, bin_m: float,
                         window_m: float, stride_m: float,
                         origin_y: float) -> None:
    """Accuracy broken down by generated segment.

    Each segment is one parameter draw at one severity, so this is an
    accuracy-versus-severity curve from a single walk -- the beginning of
    characterising the instrument rather than quoting one number.
    """
    with open(os.path.join(walkway_dir, "labels.json"), encoding="utf-8") as fh:
        labels = json.load(fh)
    segments = labels["segments"]

    print(f"\n{'segment':10s} {'severity':>9s} {'n_win':>6s} "
          f"{'disc MAE':>9s} {'ptp MAE':>8s} {'profile std':>12s}")
    for seg in segments:
        lo = origin_x + seg["x_start_m"]
        hi = origin_x + seg["x_end_m"]
        m = (samples[:, 3] >= lo) & (samples[:, 3] < hi)
        if m.sum() < 200:
            continue
        sub = samples[m]
        mx, mz, _ = build_profile(sub, bin_m)
        tx, tz = authored_profile(walkway_dir, origin_x, origin_y, sub, bin_m)
        meas = profile_report(mx, mz, bin_m, window_m, stride_m)
        true = profile_report(tx, tz, bin_m, window_m, stride_m)
        n = min(len(meas), len(true))
        if n < 3:
            continue
        dm = np.abs([meas[i]["discontinuity_mm"] - true[i]["discontinuity_mm"]
                     for i in range(n)]).mean()
        pm = np.abs([meas[i]["ptp_mm"] - true[i]["ptp_mm"]
                     for i in range(n)]).mean()
        k = min(mx.size, tx.size)
        resid = (mz[:k] - tz[:k]) * 1000.0
        resid -= np.median(resid)
        sev = seg["params"].get("joint_fault_prob", float("nan"))
        print(f"{seg['segment_id']:10s} {sev:9.2f} {n:6d} "
              f"{dm:8.2f}  {pm:7.2f}  {resid.std():11.2f}")


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--walkway", default=None)
    ap.add_argument("--origin-x", type=float, default=1100.0)
    ap.add_argument("--origin-y", type=float, default=-546.0)
    ap.add_argument("--bin", type=float, default=0.05, help="profile bin, m")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--stride", type=float, default=0.25)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--surface", action="store_true",
                    help="2D surface map + hazard map")
    ap.add_argument("--by-segment", action="store_true",
                    help="accuracy per generated segment")
    args = ap.parse_args()

    run = load_run(os.path.join(args.run, "sidewalk_evenness_timeseries.csv"))
    _, _, how = heading(run)
    print(f"heading source: {how}")

    samples = stance_samples(run)
    if samples.shape[0] < 50:
        print("too few mid-stance samples")
        return 1

    mx, mz, info = build_profile(samples, args.bin)
    span = float(samples[:, 0].max() - samples[:, 0].min())
    print(f"\n{info['n_samples']} mid-stance samples over {span:.1f} m "
          f"-> {info['n_bins']} bins of {args.bin * 100:.0f} cm")
    print(f"removed plane: grade {info['grade_pct']:+.2f}%, "
          f"cross slope {info['cross_slope_pct']:+.2f}%")

    measured = profile_report(mx, mz, args.bin, args.window, args.stride)
    md = np.array([r["discontinuity_mm"] for r in measured])
    mp = np.array([r["ptp_mm"] for r in measured])
    print(f"\nMEASURED over {len(measured)} windows:")
    print(f"  discontinuity  median {np.median(md):5.2f}  "
          f"p90 {np.percentile(md, 90):5.2f}  max {md.max():5.2f} mm")
    print(f"  roughness ptp  median {np.median(mp):5.2f}  "
          f"p90 {np.percentile(mp, 90):5.2f}  max {mp.max():5.2f} mm")
    counts: Dict[str, int] = {}
    for r in measured:
        counts[r["ada_class"]] = counts.get(r["ada_class"], 0) + 1
    print(f"  ADA classes: {counts}")

    truth = None
    tx = tz = None
    if args.walkway:
        tx, tz = authored_profile(args.walkway, args.origin_x, args.origin_y,
                                  samples, args.bin)
        truth = profile_report(tx, tz, args.bin, args.window, args.stride)
        n = min(len(measured), len(truth))
        if n >= 5:
            print(f"\nvs AUTHORED truth, {n} windows:")
            for name, key in (("discontinuity", "discontinuity_mm"),
                              ("roughness ptp", "ptp_mm")):
                a = np.array([r[key] for r in measured[:n]])
                b = np.array([r[key] for r in truth[:n]])
                err = a - b
                corr = (np.corrcoef(a, b)[0, 1]
                        if a.std() > 1e-9 and b.std() > 1e-9 else np.nan)
                print(f"  {name}  bias {err.mean():+6.2f} mm   "
                      f"MAE {np.abs(err).mean():5.2f} mm   corr {corr:+.3f}")
            agree = sum(1 for i in range(n)
                        if measured[i]["ada_class"] == truth[i]["ada_class"])
            print(f"  ADA class agreement: {agree}/{n} "
                  f"({100 * agree / n:.0f}%)")

            # Profile-level accuracy, the number an inspector would quote.
            k = min(mx.size, tx.size)
            resid = (mz[:k] - tz[:k]) * 1000.0
            resid -= np.median(resid)   # constant foot offset cancels
            print(f"  profile residual: std {resid.std():.2f} mm   "
                  f"p90 |err| {np.percentile(np.abs(resid), 90):.2f} mm")

    if args.by_segment and args.walkway:
        per_segment_accuracy(samples, args.walkway, args.origin_x,
                             args.bin, args.window, args.stride,
                             args.origin_y)

    if args.surface:
        out = os.path.join(args.run, "terrain_surface.png")
        write_surface_plot(out, samples, args.walkway,
                           args.origin_x, args.origin_y, measured)
        print(f"surface map -> {out}")

    if args.plot:
        out = os.path.join(args.run, "terrain_reconstruction.png")
        write_plot(out, mx, mz, tx, tz, measured, truth, samples)
        print(f"\nplot -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
