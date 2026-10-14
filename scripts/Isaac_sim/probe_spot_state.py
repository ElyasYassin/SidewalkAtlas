"""
probe_spot_state.py

One-off discovery script: report what proprioceptive state this Spot asset
actually exposes, so the sensor logging in the walker can be written against
real prim and joint names instead of guessed ones.

Runs INSIDE Isaac Sim (Script Editor), on a stage that contains Spot. No
simulation or policy needed -- this reads the USD, so it works with the
timeline stopped.

Prints:
  - rigid bodies (articulation links), with the foot candidates called out
  - articulated joints, in order
  - which contact-sensor and IMU APIs this Isaac build offers
  - whether any contact reporting is already configured

Usage (Script Editor):
    exec(open(r"...\\scripts\\Isaac_sim\\probe_spot_state.py").read())
"""

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema


SPOT_PRIM_PATH = globals().get("SPOT_PRIM_PATH", "/World/spot_03")

FOOT_HINTS = ("foot", "toe", "contact", "sole")
LEG_HINTS = ("leg", "hip", "knee", "thigh", "shank", "calf", "uleg", "lleg")


def _short(path, root):
    s = str(path)
    return s[len(root):].lstrip("/") if s.startswith(root) else s


def probe():
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(SPOT_PRIM_PATH)

    if not root or not root.IsValid():
        print(f"✗ No prim at {SPOT_PRIM_PATH}")
        print("  Open a stage containing Spot, or set SPOT_PRIM_PATH first.")
        return

    print(f"=== Spot state probe: {SPOT_PRIM_PATH} ===\n")

    bodies, joints, contact_reporters, imus = [], [], [], []

    for prim in Usd.PrimRange(root):
        path = prim.GetPath()
        name = prim.GetName()
        type_name = prim.GetTypeName()

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append((path, name))

        if type_name and "Joint" in str(type_name):
            joints.append((path, name, str(type_name)))

        try:
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                contact_reporters.append(path)
        except Exception:
            pass

        if "IsaacSensor" in str(type_name) or "Imu" in str(type_name):
            imus.append((path, str(type_name)))

    # --- bodies -----------------------------------------------------
    print(f"RIGID BODIES ({len(bodies)}):")
    feet, legs, others = [], [], []
    for path, name in bodies:
        low = name.lower()
        if any(h in low for h in FOOT_HINTS):
            feet.append((path, name))
        elif any(h in low for h in LEG_HINTS):
            legs.append((path, name))
        else:
            others.append((path, name))

    print(f"\n  FOOT CANDIDATES ({len(feet)}):")
    for path, name in feet:
        print(f"    {name:28s}  {_short(path, SPOT_PRIM_PATH)}")
    if not feet:
        print("    (none matched foot/toe/contact/sole -- check the leg list)")

    print(f"\n  LEG LINKS ({len(legs)}):")
    for path, name in legs:
        print(f"    {name:28s}  {_short(path, SPOT_PRIM_PATH)}")

    print(f"\n  OTHER BODIES ({len(others)}):")
    for path, name in others:
        print(f"    {name:28s}  {_short(path, SPOT_PRIM_PATH)}")

    # --- joints -----------------------------------------------------
    articulated = [j for j in joints if "Fixed" not in j[2]]
    print(f"\nJOINTS ({len(joints)} total, {len(articulated)} non-fixed):")
    for path, name, type_name in articulated:
        print(f"    {name:28s}  {type_name:22s}  {_short(path, SPOT_PRIM_PATH)}")

    # --- existing sensors -------------------------------------------
    print(f"\nEXISTING CONTACT REPORTERS: {len(contact_reporters)}")
    for path in contact_reporters:
        print(f"    {_short(path, SPOT_PRIM_PATH)}")

    print(f"EXISTING SENSOR PRIMS: {len(imus)}")
    for path, type_name in imus:
        print(f"    {type_name:24s}  {_short(path, SPOT_PRIM_PATH)}")

    # --- what this build offers -------------------------------------
    print("\nAVAILABLE APIs IN THIS BUILD:")
    for label, importer in [
        ("IMUSensor",
         lambda: __import__("isaacsim.sensors.physics", fromlist=["IMUSensor"]).IMUSensor),
        ("ContactSensor",
         lambda: __import__("isaacsim.sensors.physics", fromlist=["ContactSensor"]).ContactSensor),
        ("SingleArticulation",
         lambda: __import__("isaacsim.core.prims", fromlist=["SingleArticulation"]).SingleArticulation),
        ("Articulation (core.prims)",
         lambda: __import__("isaacsim.core.prims", fromlist=["Articulation"]).Articulation),
    ]:
        try:
            importer()
            print(f"    ok        {label}")
        except Exception as exc:
            print(f"    MISSING   {label}  ({type(exc).__name__})")

    # --- policy wrapper ---------------------------------------------
    print("\nPOLICY WRAPPER:")
    spot = None
    for key in ("_state", "spot"):
        obj = globals().get(key)
        if isinstance(obj, dict):
            spot = obj.get("spot")
        elif obj is not None:
            spot = obj
        if spot is not None:
            break

    if spot is None:
        print("    no live policy in this namespace -- run the walker first if")
        print("    you want the articulation's dof/body name lists too.")
    else:
        robot = getattr(spot, "robot", None)
        print(f"    policy: {type(spot).__name__}, robot: {type(robot).__name__}")
        for attr in ("dof_names", "body_names", "num_dof", "num_bodies"):
            if hasattr(robot, attr):
                print(f"    {attr}: {getattr(robot, attr)}")


probe()
