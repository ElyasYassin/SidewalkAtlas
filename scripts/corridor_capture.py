"""
Spot corridor walk with camera capture in Isaac Sim.

Spot follows a list of waypoints along a sidewalk corridor.
A front-facing RGB camera is attached to Spot's body and captures
images at a set interval, saved to images/corridor_capture/.

Run with:
    C:\isaac-sim\python.bat scripts\corridor_capture.py

Workflow:
  1. Load city world
  2. Spawn Spot at corridor start
  3. Attach camera to Spot
  4. Step Spot toward each waypoint using velocity commands
  5. Capture image every CAPTURE_EVERY_N_FRAMES frames
"""

from pathlib import Path
import numpy as np
from isaacsim import SimulationApp

app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import carb
import omni.usd
import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from pxr import Gf, UsdGeom, Usd
import omni.isaac.core.utils.prims as prim_utils

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CITY_USD_PATH = (
    "D:/Downloads/AECO_CityDemoPack_NVD@10011/Demos/AEC"
    "/TowerDemo/CityDemopack/World_CityDemopack.usd"
)

# Spot USD — adjust to your Isaac Lab / Isaac Sim install path
SPOT_USD_PATH = "C:/isaac-sim/assets/Isaac/Robots/BostonDynamics/spot/spot.usd"

# Prim path where Spot will be created in the stage
SPOT_PRIM_PATH = "/World/Spot"

# ---------------------------------------------------------------------------
# Corridor waypoints — (x, y, z, yaw_degrees)
# Define these to match your sidewalk corridor in the AECO city scene.
# z should be the sidewalk surface height.
# ---------------------------------------------------------------------------
CORRIDOR_WAYPOINTS = [
    (  0.0,  0.0, 0.55,   0.0),
    (  5.0,  0.0, 0.55,   0.0),
    ( 10.0,  0.0, 0.55,   0.0),
    ( 15.0,  0.0, 0.55,   0.0),
    ( 20.0,  0.0, 0.55,   0.0),
    ( 20.0,  5.0, 0.55,  90.0),
    ( 20.0, 10.0, 0.55,  90.0),
    ( 20.0, 15.0, 0.55,  90.0),
]

# Navigation settings
ARRIVAL_RADIUS      = 0.3   # meters — how close to waypoint counts as arrived
LINEAR_SPEED        = 1.0   # m/s forward speed
ANGULAR_SPEED       = 1.5   # rad/s turning speed
YAW_TOLERANCE_DEG   = 5.0   # degrees — acceptable heading error before moving

# Camera settings — position/rotation relative to Spot's base prim
CAMERA_OFFSET_XYZ   = (0.5,  0.0, 0.4)   # front-center, slightly elevated
CAMERA_ORIENT_RPY   = (0.0, -15.0, 0.0)  # tilt 15° downward
IMAGE_WIDTH         = 1280
IMAGE_HEIGHT        = 720

# Capture settings
CAPTURE_EVERY_N_FRAMES = 15   # capture one image every N simulation frames
OUTPUT_DIR = Path(__file__).parent.parent / "images" / "corridor_capture"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def yaw_from_quat(quat_wxyz: np.ndarray) -> float:
    """Extract yaw (radians) from a wxyz quaternion."""
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def angle_diff(a: float, b: float) -> float:
    """Signed difference between two angles in radians, wrapped to [-pi, pi]."""
    d = a - b
    return float((d + np.pi) % (2 * np.pi) - np.pi)


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def setup_scene(world: World):
    """Load city USD and spawn Spot."""
    stage = omni.usd.get_context().get_stage()

    # Load city as a reference on a background prim
    city_prim_path = "/World/City"
    add_reference_to_stage(usd_path=CITY_USD_PATH, prim_path=city_prim_path)
    print(f"[scene] City loaded at {city_prim_path}")

    # Spawn Spot
    x, y, z, yaw = CORRIDOR_WAYPOINTS[0]
    orientation = euler_angles_to_quat(np.array([0.0, 0.0, np.deg2rad(yaw)]))
    add_reference_to_stage(usd_path=SPOT_USD_PATH, prim_path=SPOT_PRIM_PATH)

    spot_xform = UsdGeom.Xformable(stage.GetPrimAtPath(SPOT_PRIM_PATH))
    spot_xform.ClearXformOpOrder()
    spot_xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    spot_xform.AddOrientOp().Set(Gf.Quatd(*orientation))
    print(f"[scene] Spot spawned at ({x}, {y}, {z})")


# ---------------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------------

def attach_camera(world: World):
    """Attach an RGB camera to Spot's body and return a replicator render product."""
    camera_prim_path = f"{SPOT_PRIM_PATH}/body/front_camera"

    camera = rep.create.camera(
        position=CAMERA_OFFSET_XYZ,
        rotation=CAMERA_ORIENT_RPY,
        parent=f"{SPOT_PRIM_PATH}/body",
    )

    render_product = rep.create.render_product(
        camera,
        resolution=(IMAGE_WIDTH, IMAGE_HEIGHT),
    )
    print(f"[camera] Attached at {camera_prim_path}")
    return render_product


# ---------------------------------------------------------------------------
# Image capture
# ---------------------------------------------------------------------------

def setup_writer(render_product, output_dir: Path):
    """Set up a replicator BasicWriter to save RGB images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(output_dir),
        rgb=True,
        frame_padding=6,
    )
    writer.attach([render_product])
    print(f"[capture] Saving images to {output_dir}")
    return writer


# ---------------------------------------------------------------------------
# Navigation — velocity-based waypoint follower
# ---------------------------------------------------------------------------

def get_spot_pose(stage: Usd.Stage):
    """Return Spot's current (x, y, yaw_rad) from the stage."""
    prim = stage.GetPrimAtPath(SPOT_PRIM_PATH)
    xform = UsdGeom.Xformable(prim)
    xform_ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}

    translate = xform_ops.get("xformOp:translate")
    pos = translate.Get() if translate else Gf.Vec3d(0, 0, 0)

    orient = xform_ops.get("xformOp:orient")
    if orient:
        q = orient.Get()
        quat = np.array([q.GetReal(), *q.GetImaginary()])
    else:
        quat = np.array([1.0, 0.0, 0.0, 0.0])

    return float(pos[0]), float(pos[1]), yaw_from_quat(quat)


def apply_velocity_command(stage: Usd.Stage, vx: float, vz: float, omega: float, dt: float):
    """
    Move Spot kinematically by integrating velocity commands.
    In a full Isaac Lab setup this would send commands to the locomotion policy.
    Here we move the root prim directly for data-collection purposes.
    """
    x, y, yaw = get_spot_pose(stage)

    new_x   = x + vx * np.cos(yaw) * dt
    new_y   = y + vx * np.sin(yaw) * dt
    new_yaw = yaw + omega * dt

    orientation = euler_angles_to_quat(np.array([0.0, 0.0, new_yaw]))

    prim = stage.GetPrimAtPath(SPOT_PRIM_PATH)
    xform = UsdGeom.Xformable(prim)
    xform_ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}

    if "xformOp:translate" in xform_ops:
        xform_ops["xformOp:translate"].Set(Gf.Vec3d(new_x, new_y, CORRIDOR_WAYPOINTS[0][2]))
    if "xformOp:orient" in xform_ops:
        q = Gf.Quatd(*orientation)
        xform_ops["xformOp:orient"].Set(q)


def compute_velocity_command(
    x: float, y: float, yaw: float,
    target_x: float, target_y: float, target_yaw_deg: float,
) -> tuple[float, float, float]:
    """
    Return (vx, vz, omega) to move Spot toward a waypoint.
    Turns first, then moves forward once roughly aligned.
    """
    dx = target_x - x
    dy = target_y - y
    target_heading = np.arctan2(dy, dx)
    heading_err = angle_diff(target_heading, yaw)

    if abs(np.degrees(heading_err)) > YAW_TOLERANCE_DEG:
        # Rotate in place toward waypoint
        omega = np.clip(heading_err * 2.0, -ANGULAR_SPEED, ANGULAR_SPEED)
        return 0.0, 0.0, omega
    else:
        # Move forward
        return LINEAR_SPEED, 0.0, 0.0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    setup_scene(world)
    world.reset()

    render_product = attach_camera(world)
    setup_writer(render_product, OUTPUT_DIR)

    stage = omni.usd.get_context().get_stage()

    waypoint_idx = 0
    frame = 0
    captured = 0
    dt = 1.0 / 60.0

    print(f"\n[nav] Starting corridor walk — {len(CORRIDOR_WAYPOINTS)} waypoints")

    while waypoint_idx < len(CORRIDOR_WAYPOINTS):
        tx, ty, tz, t_yaw_deg = CORRIDOR_WAYPOINTS[waypoint_idx]
        x, y, yaw = get_spot_pose(stage)

        dist = np.hypot(tx - x, ty - y)

        if dist < ARRIVAL_RADIUS:
            print(f"[nav] Reached waypoint {waypoint_idx + 1}/{len(CORRIDOR_WAYPOINTS)}")
            waypoint_idx += 1
            continue

        vx, vz, omega = compute_velocity_command(x, y, yaw, tx, ty, t_yaw_deg)
        apply_velocity_command(stage, vx, vz, omega, dt)

        # Capture image every N frames
        if frame % CAPTURE_EVERY_N_FRAMES == 0:
            rep.orchestrator.step(rt_subframes=4)
            captured += 1
            if captured % 10 == 0:
                print(f"  [capture] {captured} images saved  (frame {frame})")
        else:
            world.step(render=True)

        frame += 1

    print(f"\n[done] Corridor complete. {captured} images saved to {OUTPUT_DIR}")
    app.close()


main()
