"""
Standalone Isaac Sim script — places sidewalk defect assets into the AECO city scene.

Run with:
    C:\isaac-sim\python.bat scripts\place_defects.py

Workflow:
  1. Convert GLB defect meshes → USD (skips if already converted)
  2. Load the city stage
  3. Discover Street_Modern_Standard_Curbs prims (the sidewalk surfaces)
  4. Randomly place defect references on those surfaces
"""

import os
import random
import asyncio
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------

CITY_USD_PATH = (
    "D:/Downloads/AECO_CityDemoPack_NVD@10011/Demos/AEC"
    "/TowerDemo/CityDemopack/World_CityDemopack.usd"
)

DEFECTS_DIR = Path(
    r"C:\Users\Elyas\OneDrive - The University of Colorado Denver"
    r"\Desktop\projects\SidewalkAtlas\sidewalks"
)

# Sidewalk prim name in the AECO asset — each Shape_N has one of these.
# Street_Modern_Standard_Pavement = road surface (avoid)
# Street_Modern_Standard_Curbs    = curb + sidewalk area (target)
SIDEWALK_PRIM_NAME = "Street_Modern_Standard_Curbs"

# Override with explicit paths if auto-discovery misses any.
SIDEWALK_PRIM_PATHS: list[str] = []

# Where converted USD defect assets will be saved
USD_CACHE_DIR = DEFECTS_DIR / "usd_cache"

# Placement settings
NUM_DEFECTS = 20          # total defect instances to place
MAX_ROTATION_DEG = 360    # random yaw range
SCALE_RANGE = (0.9, 1.1)  # slight scale jitter

# Defect category weights — adjust to control class balance
CATEGORY_WEIGHTS = {
    "crack":     0.40,
    "collapsed": 0.20,
    "pothole":   0.20,
    "uneven":    0.20,
}

# ---------------------------------------------------------------------------
# Launch Isaac Sim
# ---------------------------------------------------------------------------

from isaacsim import SimulationApp

app = SimulationApp({"headless": False})

import omni.kit.asset_converter as converter
import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf
import omni.isaac.core.utils.stage as stage_utils
import carb

# ---------------------------------------------------------------------------
# Step 1 — Convert GLB → USD
# ---------------------------------------------------------------------------

def get_defect_assets() -> dict[str, list[Path]]:
    """Return {category: [glb_path, ...]} from the sidewalks directory."""
    categories: dict[str, list[Path]] = {}
    for glb in DEFECTS_DIR.glob("*.glb"):
        # filename pattern: <category>_<n>.glb
        category = glb.stem.rsplit("_", 1)[0]
        categories.setdefault(category, []).append(glb)
    return categories


async def convert_glb_to_usd(glb_path: Path, usd_path: Path) -> bool:
    if usd_path.exists():
        return True
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    task_manager = converter.get_instance()
    task = task_manager.create_converter_task(str(glb_path), str(usd_path))
    success = await task.wait_until_finished()
    if not success:
        carb.log_error(f"Conversion failed: {glb_path}")
    return success


def convert_all_assets(assets: dict[str, list[Path]]) -> dict[str, list[Path]]:
    """Returns {category: [usd_path, ...]} after converting all GLBs."""
    usd_assets: dict[str, list[Path]] = {}
    loop = asyncio.get_event_loop()
    for category, glb_paths in assets.items():
        usd_paths = []
        for glb in glb_paths:
            usd_out = USD_CACHE_DIR / f"{glb.stem}.usd"
            ok = loop.run_until_complete(convert_glb_to_usd(glb, usd_out))
            if ok:
                usd_paths.append(usd_out)
        usd_assets[category] = usd_paths
        print(f"[convert] {category}: {len(usd_paths)}/{len(glb_paths)} ready")
    return usd_assets


# ---------------------------------------------------------------------------
# Step 2 — Discover sidewalk prims
# ---------------------------------------------------------------------------

def discover_sidewalk_prims(stage: Usd.Stage) -> list[str]:
    """Return prim paths for all Street_Modern_Standard_Curbs Xforms in the stage."""
    found = []
    for prim in stage.Traverse():
        if prim.GetName() == SIDEWALK_PRIM_NAME:
            found.append(str(prim.GetPath()))
    if not found:
        carb.log_warn(
            f"No prims named '{SIDEWALK_PRIM_NAME}' found. "
            "Check SIDEWALK_PRIM_NAME or set SIDEWALK_PRIM_PATHS manually."
        )
    else:
        print(f"[discover] Found {len(found)} sidewalk prims ('{SIDEWALK_PRIM_NAME}'):")
        for p in found:
            print(f"  {p}")
    return found


def get_prim_world_bounds(stage: Usd.Stage, prim_path: str) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    """Return (min, max) world-space bounding box, including all children."""
    prim = stage.GetPrimAtPath(prim_path)
    # UsdGeom.BBoxCache handles both Mesh and Xform (composites children)
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    bbox = bbox_cache.ComputeWorldBound(prim)
    r = bbox.GetRange()
    return r.GetMin(), r.GetMax()


# ---------------------------------------------------------------------------
# Step 3 — Place defect assets
# ---------------------------------------------------------------------------

def sample_position_on_prim(
    stage: Usd.Stage, prim_path: str
) -> Gf.Vec3d:
    """Sample a random XZ position within the prim's bounding box, at surface Y."""
    mn, mx = get_prim_world_bounds(stage, prim_path)
    x = random.uniform(mn[0], mx[0])
    z = random.uniform(mn[2], mx[2])
    y = mx[1]  # top surface
    return Gf.Vec3d(x, y, z)


def place_defect(
    stage: Usd.Stage,
    usd_asset_path: Path,
    position: Gf.Vec3d,
    instance_name: str,
    parent_path: str = "/World/Defects",
) -> str:
    """Add a USD reference for the defect asset at the given world position."""
    prim_path = f"{parent_path}/{instance_name}"
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(str(usd_asset_path))

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()

    # Translation
    xform.AddTranslateOp().Set(position)

    # Random yaw rotation
    yaw = random.uniform(0, MAX_ROTATION_DEG)
    xform.AddRotateYOp().Set(yaw)

    # Scale jitter
    scale = random.uniform(*SCALE_RANGE)
    xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

    return prim_path


def weighted_category_choice(
    usd_assets: dict[str, list[Path]],
) -> tuple[str, Path]:
    """Pick a category by weight, then a random asset from that category."""
    available = {k: v for k, v in usd_assets.items() if v}
    categories = list(available.keys())
    weights = [CATEGORY_WEIGHTS.get(c, 1.0) for c in categories]
    category = random.choices(categories, weights=weights, k=1)[0]
    asset = random.choice(available[category])
    return category, asset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Convert assets
    raw_assets = get_defect_assets()
    print(f"[assets] Found categories: {list(raw_assets.keys())}")
    usd_assets = convert_all_assets(raw_assets)

    # Load city stage
    print(f"[stage] Opening {CITY_USD_PATH}")
    stage_utils.open_stage(CITY_USD_PATH)
    stage = omni.usd.get_context().get_stage()

    # Resolve sidewalk prim paths
    sidewalk_prims = SIDEWALK_PRIM_PATHS if SIDEWALK_PRIM_PATHS else discover_sidewalk_prims(stage)
    if not sidewalk_prims:
        print("[ERROR] No sidewalk prims to place defects on. Exiting.")
        return

    # Create defects parent prim
    stage.DefinePrim("/World/Defects", "Xform")

    # Place defects
    placed = 0
    for i in range(NUM_DEFECTS):
        sidewalk_prim = random.choice(sidewalk_prims)
        category, asset_path = weighted_category_choice(usd_assets)
        position = sample_position_on_prim(stage, sidewalk_prim)
        instance_name = f"{category}_{i:03d}"
        prim_path = place_defect(stage, asset_path, position, instance_name)
        print(f"[place] {instance_name} → {prim_path} at {tuple(position):.2f}")
        placed += 1

    # Save modified stage
    out_path = CITY_USD_PATH.replace(".usd", "_with_defects.usd")
    stage.Export(out_path)
    print(f"\n[done] Placed {placed} defects. Saved to: {out_path}")


main()
app.close()
