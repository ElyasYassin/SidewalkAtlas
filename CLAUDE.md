# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SidewalkAtlas — sim-to-real sidewalk unevenness detection. Train in Isaac Sim
on parametrically generated terrain with exact labels; deploy on a real Spot.

See `docs/SIM2REAL_PLAN.md` for the research plan, locked design decisions and
phase status. See `scripts/Isaac_sim/README.md` for the simulation scripts.

## Commands

Two execution environments — do not mix them up.

**Offline** (plain Python; numpy/scipy/matplotlib/PIL/cv2/torch):

```bash
python scripts/terrain/sidewalk_terrain.py                    # generator self-test
python scripts/terrain/generate_dataset.py --n 200 --out data/terrain
python scripts/Isaac_sim/export_yolo_dataset.py               # capture -> YOLO-seg
python scripts/Isaac_sim/depth_video.py                       # depth-gradient video
```

**Inside Isaac Sim** (Script Editor; needs `omni.*`, `pxr`, an open stage):
`walking.py`, `spot_walk_with_evenness_autosave.py`,
`capture_crack_dataset.py`, `crack_detection/*`,
`terrain/import_terrain_isaac.py`.

There is no build, lint or test runner. `sidewalk_terrain.py` self-tests when
run directly; keep it passing.

## Architecture

- `scripts/terrain/` — parametric sidewalk heightfield synthesis with exact
  ground truth. Pure numpy, Isaac-free, so it can be iterated without the
  simulator. Labels come from the generative parameters, not from measurement.
- `scripts/Isaac_sim/` — simulation: autonomous sidewalk walking, RGB/depth
  capture, evenness measurement by raycast, defect ground truth, dataset
  export. See that folder's README.
- `sidewalks/` — 20 Meshy-generated GLB defect assets. Superseded by
  `scripts/terrain/` for training data; a 17-shape library is too small a label
  distribution for sim-only training.

## Conventions

- `PHYSICS_RATE = 500` in the walkers is fixed by the trained policy
  (`sim.dt = 0.002` x `decimation = 10`). Do not lower it.
- `INVERT_TURN_COMMAND = True` compensates for the Spot policy's yaw sign. Not
  a bug.
- Thresholds follow ADA/PROWAG: 6.4 mm (1/4") requires beveling, 13 mm (1/2")
  is a barrier.
- Vertical discontinuity and roughness are separate labels and must stay
  separate — different hazards, different detectability.
- Output paths are absolute and machine-specific, in a `CONFIG` block at the
  top of each script.
