"""
locate_hazards.py

Report the WORLD coordinates of each hazard the leg-based measurement flagged,
and emit a ready-to-paste Isaac Sim snippet that drops a marker at each one and
flies the viewport camera to the worst.

Runs OFFLINE on a walk from spot_walk_with_evenness_autosave.py.

The measurement works in path coordinates (distance along the walk), which is
what an inspection record wants -- "14 m from the corner". To actually look at
a spot in the simulator you need world XYZ, which the stance samples carry
alongside. This converts one to the other by taking the median world position
of the footfalls inside each flagged window, so the coordinate is where feet
actually were, not an interpolation.

Usage:
    python scripts/model/locate_hazards.py --run <run_dir> --top 5
    python scripts/model/locate_hazards.py --run <run_dir> --out markers.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reconstruct_terrain import (  # noqa: E402
    ADA_BARRIER_MM, ADA_BEVEL_MM,
    build_profile, load_run, profile_report, stance_samples,
)


ISAAC_TEMPLATE = '''# Paste into the Isaac Sim Script Editor.
# Drops a coloured marker at each flagged location and frames the worst one.
#
#   red    = barrier        (>= 13.0 mm, ADA repair required)
#   orange = bevel required (>= 6.4 mm)
#
# Markers are visual only -- no collision, no physics, nothing the robot can
# interact with. Delete /World/hazard_markers to clear them.

import omni.usd
from pxr import UsdGeom, Gf, Sdf, UsdPhysics

HAZARDS = [
{rows}
]

ROOT = "/World/hazard_markers"
MARKER_RADIUS = 0.08

stage = omni.usd.get_context().get_stage()
if stage.GetPrimAtPath(ROOT):
    stage.RemovePrim(ROOT)
UsdGeom.Xform.Define(stage, ROOT)

for i, h in enumerate(HAZARDS):
    path = f"{{ROOT}}/hazard_{{i:02d}}"
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(MARKER_RADIUS)
    # Lift the marker so it floats above the surface and stays visible
    # instead of being buried in the slab it is marking.
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(
        Gf.Vec3d(h["x"], h["y"], h["z"] + 0.25))
    colour = (1.0, 0.1, 0.1) if h["mm"] >= 13.0 else (1.0, 0.6, 0.0)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    print(f'  hazard {{i}}: {{h["mm"]:.0f}} mm at '
          f'({{h["x"]:.2f}}, {{h["y"]:.2f}}, {{h["z"]:.2f}})  [{{h["at_m"]:.1f}} m along walk]')

print(f"placed {{len(HAZARDS)}} markers under {{ROOT}}")

# Fly the perspective camera to the worst one.
worst = max(HAZARDS, key=lambda h: h["mm"])
try:
    from omni.kit.viewport.utility import get_active_viewport
    from omni.kit.viewport.utility.camera_state import ViewportCameraState
    vp = get_active_viewport()
    cam = ViewportCameraState(vp.camera_path, vp)
    target = Gf.Vec3d(worst["x"], worst["y"], worst["z"])
    cam.set_target_world(target, True)
    cam.set_position_world(
        Gf.Vec3d(worst["x"] - 1.8, worst["y"] - 1.8, worst["z"] + 1.4), True)
    print(f'camera moved to the worst hazard: {{worst["mm"]:.0f}} mm '
          f'at ({{worst["x"]:.2f}}, {{worst["y"]:.2f}}, {{worst["z"]:.2f}})')
except Exception as exc:
    print(f"could not move the viewport camera ({{exc}})")
    print("select a marker in the Stage tree and press F to frame it instead")
'''


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
    ap.add_argument("--out", default=None,
                    help="write the Isaac snippet here as well as printing it")
    args = ap.parse_args()

    run = load_run(os.path.join(args.run, "sidewalk_evenness_timeseries.csv"))
    samples = stance_samples(run)
    if samples.shape[0] < 50:
        print("too few mid-stance samples")
        return 1

    mx, mz, _ = build_profile(samples, args.bin)
    measured = profile_report(mx, mz, args.bin, args.window, args.stride)
    if not measured:
        print("no windows")
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

    along = samples[:, 0]
    located = []
    for h in hazards:
        m = np.abs(along - h["centre_m"]) <= args.window / 2.0
        if m.sum() < 3:
            continue
        located.append({
            "at_m": h["centre_m"],
            "mm": h["discontinuity_mm"],
            "cls": h["ada_class"],
            # Median over the window's footfalls: a position feet actually
            # occupied, not an interpolated guess.
            "x": float(np.median(samples[m, 3])),
            "y": float(np.median(samples[m, 4])),
            "z": float(np.median(samples[m, 2])),
            "n": int(m.sum()),
        })

    print(f"\n{'#':>2s} {'along (m)':>10s} {'disc (mm)':>10s} "
          f"{'class':<15s} {'world X':>10s} {'world Y':>10s} {'world Z':>8s}")
    for i, h in enumerate(located):
        print(f"{i + 1:2d} {h['at_m']:10.2f} {h['mm']:10.1f} {h['cls']:<15s} "
              f"{h['x']:10.3f} {h['y']:10.3f} {h['z']:8.3f}")

    rows = "\n".join(
        f'    dict(x={h["x"]:.4f}, y={h["y"]:.4f}, z={h["z"]:.4f}, '
        f'mm={h["mm"]:.1f}, at_m={h["at_m"]:.2f}),'
        for h in located)
    snippet = ISAAC_TEMPLATE.format(rows=rows)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(snippet)
        print(f"\nIsaac snippet written to {args.out}")
        print("In the Script Editor:")
        print(f'    exec(open(r"{os.path.abspath(args.out)}").read())')
    else:
        print("\n" + "=" * 70)
        print(snippet)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
