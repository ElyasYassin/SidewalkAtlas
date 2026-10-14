"""
inspection_report.py

Turn a walk into an inspection document: the hazard profile measured from
Spot's feet, with the camera's view of the sidewalk at each flagged location.

Runs OFFLINE on a walk produced by spot_walk_with_evenness_autosave.py.

WHY LINK IMAGERY
----------------
A number alone is hard to act on. "14 mm vertical discontinuity at 23.4 m" is
a measurement; the same measurement beside a photograph of the slab is
something a maintenance crew can find and a reviewer can check. The photograph
is evidence, not input -- the measurement comes entirely from foot positions
and never looks at the camera.

The join is exact rather than approximate: capture filenames carry the physics
step (`rgb_000003_step_0004208.png`) and the evenness CSV carries step ->
path distance, so each frame maps to a point on the measured profile with no
interpolation or timestamp matching.

Usage:
    python scripts/model/inspection_report.py --run <run_dir>
    python scripts/model/inspection_report.py --run <run_dir> --top 8 --depth
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reconstruct_terrain import (  # noqa: E402
    ADA_BARRIER_MM, ADA_BEVEL_MM,
    build_profile, load_run, profile_report, stance_samples,
)

STEP_RE = re.compile(r"_step_(\d+)\.png$")


# =====================================================================
# FRAME INDEX
# =====================================================================

def frame_index(run_dir: str, sub: str) -> List[Tuple[int, str]]:
    """(physics step, path) for each captured frame, sorted by step."""
    folder = os.path.join(run_dir, sub)
    if not os.path.isdir(folder):
        return []
    out = []
    for name in os.listdir(folder):
        m = STEP_RE.search(name)
        if m:
            out.append((int(m.group(1)), os.path.join(folder, name)))
    return sorted(out)


def step_to_distance(run: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """The join key. Both columns come from the same CSV rows, so a frame's
    step maps to the exact path distance the robot had covered."""
    return run["step"], run["distance_m"]


def nearest_frame(frames: List[Tuple[int, str]],
                  steps: np.ndarray, dists: np.ndarray,
                  target_m: float) -> Optional[Tuple[str, float, float]]:
    """Frame whose capture point is closest to a distance along the walk.

    Returns (path, actual distance, error in metres). The error is reported
    because captures are ~2 s apart, so a frame can be up to about a metre
    from the hazard it is meant to illustrate -- worth showing rather than
    hiding.
    """
    if not frames:
        return None
    best = None
    for step, path in frames:
        k = int(np.argmin(np.abs(steps - step)))
        d = float(dists[k])
        err = abs(d - target_m)
        if best is None or err < best[2]:
            best = (path, d, err)
    return best


# =====================================================================
# REPORT
# =====================================================================

def write_report(out_path: str,
                 measured: List[Dict],
                 hazards: List[Dict],
                 run: Dict[str, np.ndarray],
                 rgb: List[Tuple[int, str]],
                 depth: List[Tuple[int, str]],
                 show_depth: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    steps, dists = step_to_distance(run)
    n = len(hazards)
    img_rows = 2 if (show_depth and depth) else 1
    cols = max(n, 1)

    fig = plt.figure(figsize=(3.1 * cols, 4.2 + 2.6 * img_rows))
    gs = fig.add_gridspec(1 + img_rows, cols, height_ratios=[2.2] +
                          [2.0] * img_rows, hspace=0.32, wspace=0.06)

    # --- hazard profile across the full width ---------------------
    ax = fig.add_subplot(gs[0, :])
    cx = [r["centre_m"] for r in measured]
    dv = [r["discontinuity_mm"] for r in measured]
    top = max(max(dv, default=0), ADA_BARRIER_MM) * 1.18
    ax.axhspan(0, ADA_BEVEL_MM, color="#2e7d32", alpha=0.12, label="compliant")
    ax.axhspan(ADA_BEVEL_MM, ADA_BARRIER_MM, color="#f9a825", alpha=0.15,
               label="bevel required")
    ax.axhspan(ADA_BARRIER_MM, top, color="#c62828", alpha=0.12,
               label="barrier")
    ax.plot(cx, dv, lw=1.5, color="#111")
    for i, h in enumerate(hazards):
        ax.axvline(h["centre_m"], color="#c62828", lw=1.0, ls="--", alpha=0.7)
        ax.annotate(f"{i + 1}", (h["centre_m"], h["discontinuity_mm"]),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=9, fontweight="bold",
                    color="#c62828")
    ax.set_ylim(0, top)
    ax.set_title("Measured vertical discontinuity, with worst locations marked",
                 fontsize=11)
    ax.set_xlabel("along walk (m)")
    ax.set_ylabel("mm")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    # --- one column of imagery per hazard --------------------------
    for i, h in enumerate(hazards):
        for row, (frames, label) in enumerate(
                [(rgb, "camera")] + ([(depth, "depth")] if img_rows == 2 else [])):
            axi = fig.add_subplot(gs[1 + row, i])
            axi.set_xticks([])
            axi.set_yticks([])
            hit = nearest_frame(frames, steps, dists, h["centre_m"])
            if hit is None:
                axi.text(0.5, 0.5, "no frame", ha="center", va="center",
                         fontsize=9, transform=axi.transAxes)
                continue
            path, actual, err = hit
            try:
                axi.imshow(mpimg.imread(path))
            except Exception as exc:
                axi.text(0.5, 0.5, f"unreadable\n{exc}", ha="center",
                         va="center", fontsize=7, transform=axi.transAxes)
                continue
            if row == 0:
                axi.set_title(
                    f"{i + 1}.  {h['discontinuity_mm']:.0f} mm  "
                    f"@ {h['centre_m']:.1f} m\n"
                    f"{label} at {actual:.1f} m  (Δ{err:.1f} m)",
                    fontsize=8.5)
            else:
                axi.set_xlabel(f"{label} @ {actual:.1f} m", fontsize=8)

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--top", type=int, default=6,
                    help="how many worst locations to illustrate")
    ap.add_argument("--bin", type=float, default=0.05)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--stride", type=float, default=0.25)
    ap.add_argument("--depth", action="store_true",
                    help="add a depth row beneath the camera row")
    ap.add_argument("--min-separation", type=float, default=1.5,
                    help="metres between reported hazards, so one bad slab "
                         "does not fill every column")
    args = ap.parse_args()

    run = load_run(os.path.join(args.run, "sidewalk_evenness_timeseries.csv"))
    samples = stance_samples(run)
    if samples.shape[0] < 50:
        print("too few mid-stance samples")
        return 1

    mx, mz, info = build_profile(samples, args.bin)
    measured = profile_report(mx, mz, args.bin, args.window, args.stride)
    if not measured:
        print("no windows")
        return 1

    rgb = frame_index(args.run, "rgb")
    depth = frame_index(args.run, "depth_vis")
    print(f"{info['n_samples']} stance samples, {len(measured)} windows, "
          f"{len(rgb)} camera frames, {len(depth)} depth frames")

    # Worst-first, but spread out: adjacent windows overlap 75%, so without a
    # separation rule the same slab would occupy every column of the report.
    hazards: List[Dict] = []
    for r in sorted(measured, key=lambda r: -r["discontinuity_mm"]):
        if r["discontinuity_mm"] < ADA_BEVEL_MM:
            break
        if all(abs(r["centre_m"] - h["centre_m"]) >= args.min_separation
               for h in hazards):
            hazards.append(r)
        if len(hazards) >= args.top:
            break
    hazards.sort(key=lambda r: r["centre_m"])

    if not hazards:
        print("no locations above the ADA bevel threshold")
        return 0

    print(f"\n{'#':>2s}  {'at (m)':>7s}  {'disc (mm)':>10s}  {'class':<15s} frame")
    steps, dists = step_to_distance(run)
    for i, h in enumerate(hazards):
        hit = nearest_frame(rgb, steps, dists, h["centre_m"])
        frame = os.path.basename(hit[0]) if hit else "-"
        # ASCII only here: the Windows console is cp1252 and cannot encode the
        # delta sign. Matplotlib renders unicode fine, so the plot keeps it.
        off = f"  (off by {hit[2]:.1f} m)" if hit else ""
        print(f"{i + 1:2d}  {h['centre_m']:7.2f}  {h['discontinuity_mm']:10.1f}  "
              f"{h['ada_class']:<15s} {frame}{off}")

    out = os.path.join(args.run, "inspection_report.png")
    write_report(out, measured, hazards, run, rgb, depth, args.depth)
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
