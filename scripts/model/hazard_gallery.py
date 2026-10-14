"""
hazard_gallery.py

Show the sidewalk at each location the leg-based measurement flagged, using
the onboard depth camera.

Runs OFFLINE on a walk from spot_walk_with_evenness_autosave.py.

WHY DEPTH RATHER THAN RGB
-------------------------
RGB captures render black whenever the stage has no camera-visible light --
Isaac's viewport headlight is not a stage prim, so offscreen render products
see an unlit scene. Depth does not care: it is a geometric pass. So depth is
the reliable visual channel, and on this problem it is also the more
informative one, since the thing being inspected is shape, not appearance.

Raw depth is nearly useless to look at -- a sidewalk receding from the camera
is a smooth ramp, and a 15 mm fault is invisible against a 0.5-10 m range.
Two derived views expose it, the same pair `view_defects.py` uses:

  local deviation  depth minus a locally smoothed copy of itself, so the
                   overall ramp cancels and only relief survives. A fault
                   appears as a band of one sign against near-zero.
  depth gradient   magnitude of the spatial derivative, which peaks at edges.
                   Slab joints and fault lips light up; smooth heave does not.

The depth images are EVIDENCE, not input. Every number comes from foot
positions; the camera never enters the measurement.

Usage:
    python scripts/model/hazard_gallery.py --run <run_dir> --top 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reconstruct_terrain import (  # noqa: E402
    ADA_BARRIER_MM, ADA_BEVEL_MM,
    build_profile, load_run, profile_report, stance_samples,
)

STEP_RE = re.compile(r"_step_(\d+)\.npy$")


def depth_index(run_dir: str) -> List[Tuple[int, str]]:
    folder = os.path.join(run_dir, "depth_npy")
    if not os.path.isdir(folder):
        return []
    out = []
    for name in os.listdir(folder):
        m = STEP_RE.search(name)
        if m:
            out.append((int(m.group(1)), os.path.join(folder, name)))
    return sorted(out)


def nearest(frames, steps, dists, target_m):
    if not frames:
        return None
    best = None
    for step, path in frames:
        k = int(np.argmin(np.abs(steps - step)))
        err = abs(float(dists[k]) - target_m)
        if best is None or err < best[2]:
            best = (path, float(dists[k]), err)
    return best


# =====================================================================
# DERIVED VIEWS
# =====================================================================

def local_deviation_mm(depth_m: np.ndarray, window_px: int = 61) -> np.ndarray:
    """Depth minus a locally smoothed copy, in millimetres.

    The smoothing is weighted by validity so holes do not drag the local mean
    toward zero and manufacture a defect at every dropout.
    """
    valid = np.isfinite(depth_m) & (depth_m > 0)
    safe = np.where(valid, depth_m, 0.0)
    smooth = uniform_filter(safe, size=window_px)
    weight = uniform_filter(valid.astype(np.float32), size=window_px)
    smooth = np.where(weight > 0.1, smooth / np.maximum(weight, 1e-3), 0.0)
    out = (depth_m - smooth) * 1000.0
    out[~valid] = np.nan
    return out


def gradient_mm_per_px(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    safe = np.where(valid, depth_m, 0.0)
    gy, gx = np.gradient(safe)
    out = np.sqrt(gx ** 2 + gy ** 2) * 1000.0
    out[~valid] = np.nan
    return out


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--bin", type=float, default=0.05)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--stride", type=float, default=0.25)
    ap.add_argument("--min-separation", type=float, default=1.5)
    ap.add_argument("--lead", type=float, default=1.2,
                    help="metres the camera looks AHEAD of the body; a "
                         "hazard at X is framed from a capture at X-lead")
    ap.add_argument("--crop", type=float, default=0.55,
                    help="keep this bottom fraction of the image, where "
                         "the near-field ground is; 1.0 keeps all")
    ap.add_argument("--dev-clip", type=float, default=15.0,
                    help="local-deviation colour limit, mm")
    ap.add_argument("--grad-clip", type=float, default=25.0,
                    help="gradient colour limit, mm/px")
    args = ap.parse_args()

    run = load_run(os.path.join(args.run, "sidewalk_evenness_timeseries.csv"))
    samples = stance_samples(run)
    mx, mz, info = build_profile(samples, args.bin)
    measured = profile_report(mx, mz, args.bin, args.window, args.stride)
    frames = depth_index(args.run)
    print(f"{info['n_samples']} stance samples, {len(measured)} windows, "
          f"{len(frames)} depth frames")
    if not measured or not frames:
        print("nothing to show")
        return 1

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

    steps, dists = run["step"], run["distance_m"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(hazards)
    fig, axes = plt.subplots(3, n, figsize=(3.4 * n, 7.2), squeeze=False)

    print(f"\n{'#':>2s} {'at (m)':>7s} {'disc':>7s}  frame")
    for i, h in enumerate(hazards):
        # Frame the hazard, do not stand on it. The camera points forward,
        # so the capture taken AT a hazard shows the ground beyond it. The
        # one taken `lead` metres earlier is the one looking at it.
        hit = nearest(frames, steps, dists, h["centre_m"] - args.lead)
        if hit is None:
            continue
        path, actual, err = hit
        depth = np.load(path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if 0.1 < args.crop < 1.0:
            # The upper image is sky, buildings and far road, whose huge
            # depth range swamps the colour scale. The sidewalk being
            # inspected is in the near field at the bottom.
            depth = depth[int(depth.shape[0] * (1.0 - args.crop)):, :]
        print(f"{i + 1:2d} {h['centre_m']:7.2f} {h['discontinuity_mm']:6.1f}  "
              f"{os.path.basename(path)}  (off by {err:.1f} m)")

        valid = np.isfinite(depth) & (depth > 0)
        near = np.nanpercentile(depth[valid], 2) if valid.any() else 0.0
        far = np.nanpercentile(depth[valid], 60) if valid.any() else 5.0

        ax = axes[0][i]
        ax.imshow(np.where(valid, depth, np.nan), cmap="viridis",
                  vmin=near, vmax=far)
        ax.set_title(f"{i + 1}.  {h['discontinuity_mm']:.0f} mm "
                     f"@ {h['centre_m']:.1f} m\ndepth "
                     f"({near:.1f}-{far:.1f} m)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[1][i]
        im = ax.imshow(local_deviation_mm(depth), cmap="RdBu_r",
                       vmin=-args.dev_clip, vmax=args.dev_clip)
        ax.set_title(f"local deviation  ±{args.dev_clip:.0f} mm", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if i == n - 1:
            fig.colorbar(im, ax=ax, fraction=0.035, label="mm")

        ax = axes[2][i]
        im = ax.imshow(gradient_mm_per_px(depth), cmap="hot",
                       vmin=0, vmax=args.grad_clip)
        ax.set_title(f"depth gradient  (edges)", fontsize=9)
        ax.set_xlabel(f"camera at {actual:.1f} m, looking at "
                      f"~{h['centre_m']:.1f} m", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if i == n - 1:
            fig.colorbar(im, ax=ax, fraction=0.035, label="mm/px")

    fig.suptitle("Sidewalk at each location flagged by the leg-based "
                 "measurement (depth camera)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(args.run, "hazard_gallery.png")
    fig.savefig(out, dpi=105)
    plt.close(fig)
    print(f"\ngallery -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
