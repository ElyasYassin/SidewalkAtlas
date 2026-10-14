"""
paint_survey.py

Emit an Isaac Sim snippet that paints the whole surveyed corridor with the
leg-based measurement, ADA-coloured, following the terrain it measured.

Runs OFFLINE on a walk from spot_walk_with_evenness_autosave.py.

WHY A CONTINUOUS RIBBON
-----------------------
Markers on the worst few spots answer "where is the worst damage". An
inspection needs the complementary question answered too: "what did you
survey, and what did you find everywhere else". A continuous stripe shows
compliant stretches as surveyed-and-fine rather than as absence of evidence,
and it makes the extent of the survey visible -- the ribbon covers exactly the
corridor the feet actually crossed, and stops where they stopped.

The ribbon follows the measured terrain height and floats a couple of
centimetres above it, so it reads as paint on the sidewalk rather than a
floating chart, and it bends with the path instead of assuming a straight walk.

Colour is keyed to the ADA thresholds:
    green  -> compliant      (< 6.4 mm)
    amber  -> bevel required (6.4-13 mm)
    red    -> barrier        (>= 13 mm), deepening with severity

Usage:
    python scripts/model/paint_survey.py --run <run_dir> --out survey.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reconstruct_terrain import (  # noqa: E402
    ADA_BARRIER_MM, ADA_BEVEL_MM,
    build_profile, load_run, profile_report, stance_samples,
)


ISAAC_TEMPLATE = '''# Paste into the Isaac Sim Script Editor.
#
# Paints the corridor Spot surveyed with its own measurements.
#   green -> compliant (< 6.4 mm)   amber -> bevel required   red -> barrier
#
# Visual only: no collision, no physics. Delete /World/survey_overlay to clear.

import omni.usd
from pxr import UsdGeom, Gf, Vt

ROOT = "/World/survey_overlay"
LIFT_M = {lift}          # float above the surface so it reads as paint
HALF_WIDTH_M = {half_w}  # half the corridor the feet actually covered

# station: world x, y, z, measured discontinuity in mm
STATIONS = [
{rows}
]

stage = omni.usd.get_context().get_stage()
if stage.GetPrimAtPath(ROOT):
    stage.RemovePrim(ROOT)
UsdGeom.Xform.Define(stage, ROOT)

pts, colours = [], []
n = len(STATIONS)
for i, s in enumerate(STATIONS):
    # Local path direction, from neighbours, so the ribbon bends with the walk
    # instead of assuming it ran along an axis.
    a = STATIONS[max(i - 1, 0)]
    b = STATIONS[min(i + 1, n - 1)]
    dx, dy = b[0] - a[0], b[1] - a[1]
    m = (dx * dx + dy * dy) ** 0.5 or 1.0
    px, py = -dy / m, dx / m          # unit perpendicular
    z = s[2] + LIFT_M
    pts.append(Gf.Vec3f(s[0] - px * HALF_WIDTH_M, s[1] - py * HALF_WIDTH_M, z))
    pts.append(Gf.Vec3f(s[0] + px * HALF_WIDTH_M, s[1] + py * HALF_WIDTH_M, z))

def ada_colour(mm):
    if mm < 6.4:
        t = max(mm, 0.0) / 6.4
        return (0.15 + 0.55 * t, 0.70, 0.20)          # green -> yellow-green
    if mm < 13.0:
        t = (mm - 6.4) / 6.6
        return (1.0, 0.72 - 0.30 * t, 0.05)           # amber -> orange
    t = min((mm - 13.0) / 25.0, 1.0)
    return (0.90 - 0.35 * t, 0.10 - 0.08 * t, 0.10 - 0.08 * t)  # red -> dark

idx, counts = [], []
for i in range(n - 1):
    a = 2 * i
    idx.extend([a, a + 1, a + 3, a + 2])
    counts.append(4)
    colours.append(Gf.Vec3f(*ada_colour(0.5 * (STATIONS[i][3] + STATIONS[i + 1][3]))))

mesh = UsdGeom.Mesh.Define(stage, ROOT + "/ribbon")
mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
mesh.CreateDoubleSidedAttr(True)

# One colour per face, not per vertex: a fault is a discrete finding, and
# interpolating it into a gradient would blur the compliance boundary that
# the whole record turns on.
cp = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform)
cp.Set(Vt.Vec3fArray(colours))

vals = [s[3] for s in STATIONS]
bad = sum(1 for v in vals if v >= 13.0)
mid = sum(1 for v in vals if 6.4 <= v < 13.0)
print(f"painted {{len(counts)}} segments over "
      f"{{sum(((STATIONS[i+1][0]-STATIONS[i][0])**2 + (STATIONS[i+1][1]-STATIONS[i][1])**2)**0.5 for i in range(n-1)):.1f}} m")
print(f"  barrier {{bad}}  bevel {{mid}}  compliant {{len(vals)-bad-mid}}  "
      f"(of {{len(vals)}} stations)")
print(f"  worst {{max(vals):.1f}} mm   median {{sorted(vals)[len(vals)//2]:.1f}} mm")
print(f"select {{ROOT}} in the Stage tree and press F to frame the survey")
'''


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--bin", type=float, default=0.05)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--stride", type=float, default=0.25)
    ap.add_argument("--station", type=float, default=0.20,
                    help="spacing of painted segments, metres")
    ap.add_argument("--half-width", type=float, default=0.35,
                    help="half the painted corridor width, metres")
    ap.add_argument("--lift", type=float, default=0.03,
                    help="float above the surface, metres")
    ap.add_argument("--out", default=None)
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

    # Measurement is per 1 m window at 0.25 m stride; the ribbon wants a value
    # at every station, so interpolate the window centres onto them.
    wc = np.array([r["centre_m"] for r in measured])
    wv = np.array([r["discontinuity_mm"] for r in measured])

    along = samples[:, 0]
    lo, hi = float(along.min()), float(along.max())
    stations = np.arange(lo, hi, args.station)

    rows: List[str] = []
    values: List[float] = []
    for s in stations:
        m = np.abs(along - s) <= args.station
        if m.sum() < 2:
            continue
        x = float(np.median(samples[m, 3]))
        y = float(np.median(samples[m, 4]))
        z = float(np.median(samples[m, 2]))
        v = float(np.interp(s, wc, wv))
        values.append(v)
        rows.append(f"    ({x:.4f}, {y:.4f}, {z:.4f}, {v:.2f}),")

    if len(rows) < 4:
        print("not enough stations to paint")
        return 1

    vals = np.array(values)
    print(f"{info['n_samples']} stance samples -> {len(rows)} painted stations "
          f"every {args.station * 100:.0f} cm over {hi - lo:.1f} m of walk")
    print(f"  barrier   (>= 13.0 mm): {int((vals >= 13).sum()):4d} "
          f"({100 * (vals >= 13).mean():.0f}%)")
    print(f"  bevel     (6.4-13 mm) : {int(((vals >= 6.4) & (vals < 13)).sum()):4d} "
          f"({100 * ((vals >= 6.4) & (vals < 13)).mean():.0f}%)")
    print(f"  compliant (< 6.4 mm)  : {int((vals < 6.4).sum()):4d} "
          f"({100 * (vals < 6.4).mean():.0f}%)")
    print(f"  worst {vals.max():.1f} mm, median {np.median(vals):.1f} mm")

    snippet = ISAAC_TEMPLATE.format(
        rows="\n".join(rows),
        lift=args.lift,
        half_w=args.half_width,
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(snippet)
        print(f"\nIsaac snippet -> {args.out}")
        print("In the Script Editor:")
        print(f'    exec(open(r"{os.path.abspath(args.out)}").read())')
    else:
        print("\n" + snippet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
