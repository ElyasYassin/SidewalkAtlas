"""
import_terrain_isaac.py

Bring a generated parametric sidewalk into Isaac Sim and verify that physics
reproduces the authored geometry.

Runs INSIDE Isaac Sim (Script Editor). Reads a sample directory produced by
scripts/terrain/generate_dataset.py and builds the surface directly as a
UsdGeom.Mesh from height.npy -- no OBJ round trip, no asset converter. (The
OBJ that generate_dataset.py also writes is for external tools; it is not used
here.)

CRITICAL: collision approximation must be "none" (exact triangle mesh).
Every other PhysX approximation -- convexHull, convexDecomposition, boundingCube
-- discards exactly the millimetre-scale relief this whole project measures. A
convex hull of a sidewalk slab is a flat box: every fault, crack and heave
disappears, the robot walks on a perfectly smooth surface, and the labels
describe geometry the feet never touch.

Usage (Script Editor):
    SAMPLE_DIR = ".../data/terrain/sw_00007"
    exec(open(".../scripts/terrain/import_terrain_isaac.py").read())
"""

import asyncio
import json
import os

import numpy as np

import omni.usd
import omni.timeline
import omni.kit.app
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, Vt


# =====================================================================
# CONFIG
# =====================================================================

# Any of these may be set in the Script Editor BEFORE exec'ing this file; a
# value already in the namespace wins over the default below. That way the
# sample can be switched without editing this file each time.
_g = globals()

SAMPLE_DIR = _g.get("SAMPLE_DIR",
                    r"C:\Users\Elyas\sidewalk_dataset\terrain\sw_00000")
PRIM_PATH = _g.get("PRIM_PATH", "/World/parametric_sidewalk")
ORIGIN_XYZ = _g.get("ORIGIN_XYZ", (0.0, 0.0, 0.0))  # placement in world space
SEMANTIC_CLASS = "sidewalk"      # matches WALKABLE_CLASSES in the walker

# Verification: cast rays down over a grid and compare with authored heights.
VERIFY = True
VERIFY_SAMPLES = 400
VERIFY_TOLERANCE_MM = 1.0
VERIFY_RAY_HEIGHT_M = 1.0
WARMUP_FRAMES = 10   # frames to let PhysX cook the new collider

# Long walkways are built as a row of chunk meshes rather than one giant
# collider. See build_chunks() for why, and for the shared-boundary detail.
CHUNK_LENGTH_M = 20.0

# Refuse to import a walkway big enough to lock the application.
#
# Learned the hard way: a 234 m walkway at 1 cm (4.2 M vertices, 8.4 M
# triangles) froze Isaac outright. Chunking solved the PhysX side, but the RTX
# viewport still draws every triangle, and the import waits on the app loop to
# tick -- so a viewport taking tens of seconds per frame hangs the whole thing.
# The limit is about RENDERING, not physics.
#
# Two ways past it: coarser cells (2 cm quarters the count and is still finer
# than Spot's ~4 cm foot), or RENDER_TERRAIN = False below.
MAX_TRIANGLES = 2_000_000
ALLOW_OVERSIZE = _g.get("ALLOW_OVERSIZE", False)

# Collision does not need a visible mesh. For physics-only collection runs --
# where Spot walks a known straight strip and needs no camera -- hiding the
# terrain removes the render cost entirely while leaving collision intact.
# Keep it True when the walker's vision-based navigation is in use.
RENDER_TERRAIN = _g.get("RENDER_TERRAIN", True)

# Stand Spot at the walkway start after import, so every run begins from
# the same pose relative to the terrain and runs stay comparable.
PLACE_SPOT = _g.get("PLACE_SPOT", True)
SPOT_PRIM_PATH = _g.get("SPOT_PRIM_PATH", "/World/spot_03")
SPOT_START_OFFSET_M = _g.get("SPOT_START_OFFSET_M", 0.6)

# Spawn height of the robot ROOT above the terrain directly beneath it.
#
# Too low and the feet start intersecting the ground; PhysX resolves that
# penetration by shoving the robot out, which reads as Spot falling over at
# spawn. Too high and it drops, which the policy usually absorbs.
# Err high: a small drop is recoverable, initial penetration often is not.
#
# The root prim origin is not necessarily the body origin, so this is not
# simply Spot's standing height (~0.52 m body above ground while walking).
# Tune by setting SPOT_STAND_HEIGHT_M in the Script Editor before exec.
SPOT_STAND_HEIGHT_M = _g.get("SPOT_STAND_HEIGHT_M", 0.80)

# Heading, degrees about +Z. The walkway runs along +X and Spot's body
# forward is its local +X, so 0 faces down the strip. Set 180 to walk it
# backwards from the far end.
SPOT_YAW_DEG = _g.get("SPOT_YAW_DEG", 0.0)


# =====================================================================
# MESH CONSTRUCTION
# =====================================================================

def load_sample(sample_dir):
    height_mm = np.load(os.path.join(sample_dir, "height.npy"))
    with open(os.path.join(sample_dir, "labels.json"), encoding="utf-8") as fh:
        labels = json.load(fh)
    return height_mm.astype(np.float64), labels


def build_mesh(stage, prim_path, height_mm, cell_m, origin):
    """Create a Z-up triangulated heightfield as a UsdGeom.Mesh."""
    nx, ny = height_mm.shape
    ox, oy, oz = origin

    xs = np.arange(nx) * cell_m + ox
    ys = np.arange(ny) * cell_m + oy
    zs = height_mm / 1000.0 + oz

    # Built with numpy and handed over via Vt.*.FromNumpy. A Python loop over
    # ~200k vertices and ~400k triangles locks the Script Editor for a long
    # time; this is effectively instant.
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([X, Y, zs], axis=-1).reshape(-1, 3).astype(np.float32)

    i = np.arange(nx - 1)[:, None]
    j = np.arange(ny - 1)[None, :]
    a = (i * ny + j)
    b, c, d = a + 1, a + ny, a + ny + 1
    tris = np.concatenate(
        [np.stack([a, c, b], axis=-1)[..., None, :],
         np.stack([b, c, d], axis=-1)[..., None, :]],
        axis=2,
    )
    indices = tris.reshape(-1).astype(np.int32)
    counts = np.full(2 * (nx - 1) * (ny - 1), 3, dtype=np.int32)

    if stage.GetPrimAtPath(prim_path):
        stage.RemovePrim(prim_path)

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    extent = UsdGeom.Boundable.ComputeExtentFromPlugins(
        mesh, Usd.TimeCode.Default()
    )
    if extent:
        mesh.CreateExtentAttr(extent)

    return mesh


def build_chunks(stage, root_path, height_mm, cell_m, origin, chunk_len_m):
    """Split a long walkway into several collider prims.

    A 234 m walkway at 1 cm is ~4.3 M vertices and ~8.6 M triangles -- about
    20x the 12 m surface that cooked cleanly. PhysX will not thank you for a
    single triangle mesh that size, either to cook or to hold, so the walkway
    is built as a row of chunk meshes under one Xform.

    Adjacent chunks SHARE their boundary column (note `start = stop`, not
    `stop + 1`). Without that the two meshes meet at coincident-but-distinct
    edges and a foot can catch in the hairline gap -- a contact artefact that
    would look exactly like a real defect in the sensor data, and one that
    appears nowhere in the labels.
    """
    nx = height_mm.shape[0]
    chunk_cells = max(2, int(round(chunk_len_m / cell_m)))
    ox, oy, oz = origin

    if stage.GetPrimAtPath(root_path):
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)

    paths = []
    start = 0
    while start < nx - 1:
        stop = min(start + chunk_cells, nx - 1)
        sub = height_mm[start:stop + 1, :]
        path = f"{root_path}/chunk_{len(paths):03d}"
        build_mesh(stage, path, sub, cell_m,
                   (ox + start * cell_m, oy, oz))
        paths.append(path)
        start = stop          # share the boundary column with the next chunk
    return paths


def add_exact_collision(prim):
    """Static triangle-mesh collider.

    Approximation must be 'none' -- see the warning at the top of the file.

    Deliberately NOT using PhysxSDFMeshCollisionAPI: signed-distance-field
    collision exists for dynamic, roughly convex bodies. On a static
    400k-triangle heightfield it is enormously expensive to cook and can
    leave the collider unusable, which shows up as a scene query that hits
    nothing at all. A static mesh with approximation 'none' is already exact
    triangle collision -- there is nothing for SDF to add here.

    No RigidBodyAPI, so the mesh is a static collider and will not fall.
    """
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision.CreateApproximationAttr().Set(UsdPhysics.Tokens.none)


def _set_pose(prim, x, y, z, yaw_deg):
    """Set a prim's position AND heading.

    Heading matters as much as position and is easier to forget. Placing Spot
    correctly on the walkway while leaving its old city heading meant the
    strip appeared off to the right of the camera: perception reported
    `L/C/R = 0/512/21224`, then nothing in the near band, and the robot spent
    the whole run backing up in recovery. It was standing in exactly the
    right spot, facing the wrong way.

    Three op-stack shapes, because USD allows all of them and Spot uses the
    one the obvious approaches do not handle:

    1. A single `xformOp:transform` matrix. XformCommonAPI REFUSES these --
       it returned False here and silently left the robot where it was.
       Rebuilt from identity, so the resulting basis is orthonormal by
       construction (an earlier attempt left one USD warned about).
    2. Separate translate / orient ops. Set each.
    3. Neither. Fall back to XformCommonAPI, then to adding ops.

    What NOT to do is `ClearXformOpOrder()` then `AddTranslateOp()`: that
    discards rotation entirely.
    """
    xform = UsdGeom.Xformable(prim)
    target = Gf.Vec3d(float(x), float(y), float(z))
    rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(yaw_deg))
    ops = xform.GetOrderedXformOps()

    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            matrix = Gf.Matrix4d(1.0)
            matrix.SetRotateOnly(rotation)
            matrix.SetTranslateOnly(target)
            op.Set(matrix)
            return True

    did_translate = False
    for op in ops:
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate:
            op.Set(target)
            did_translate = True
        elif op_type == UsdGeom.XformOp.TypeOrient:
            quat = rotation.GetQuat()
            if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                op.Set(Gf.Quatf(quat))
            else:
                op.Set(Gf.Quatd(quat))
        elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
            op.Set(Gf.Vec3d(0.0, 0.0, float(yaw_deg)))
    if did_translate:
        return True

    common = UsdGeom.XformCommonAPI(prim)
    if common.SetTranslate(target):
        common.SetRotate(Gf.Vec3f(0.0, 0.0, float(yaw_deg)))
        return True

    try:
        translate = xform.AddTranslateOp()
        translate.Set(target)
        xform.SetXformOpOrder([translate] + list(ops))
        return True
    except Exception as exc:
        print(f"  ({exc})")
        return False


def place_spot_at_start(stage, height_mm, cell_m, origin,
                        spot_path, start_offset_m, stand_height_m,
                        yaw_deg=0.0):
    """Stand Spot at the walkway's start, facing down its length.

    Saves hand-dragging the robot into place, and more importantly puts it
    somewhere reproducible: every run starts from the same pose relative to
    the terrain, so runs are comparable.

    Spot's body frame has +X forward (the walker reads forward from
    matrix[0, :3]), and the walkway runs along +X, so an identity rotation
    already points the robot the right way. Height is taken from the actual
    terrain under the start position rather than assumed, since the walkway
    carries a running grade.
    """
    spot = stage.GetPrimAtPath(spot_path)
    if not spot or not spot.IsValid():
        print(f"  no Spot at {spot_path} -- skipping placement")
        return False

    ox, oy, oz = origin
    ny = height_mm.shape[1]
    i = max(0, min(height_mm.shape[0] - 1, int(round(start_offset_m / cell_m))))
    j = ny // 2
    ground_z = float(height_mm[i, j]) / 1000.0 + oz

    x = ox + start_offset_m
    y = oy + (ny // 2) * cell_m
    z = ground_z + stand_height_m

    if not _set_pose(spot, x, y, z, yaw_deg):
        print("  ⚠ could not set Spot's pose -- place it by hand")
        return False

    print(f"✓ Spot placed at ({x:.2f}, {y:.2f}, {z:.2f}), yaw {yaw_deg:.0f}deg "
          f"({stand_height_m:.2f} m above terrain)")
    return True


def ensure_light(stage, path="/World/inspection_sun"):
    """Add a distant light if the stage has none that a camera can see.

    The Isaac viewport lights itself with a default headlight that is NOT a
    stage prim. Offscreen render products -- which is what the walker's camera
    is -- do not get it, so they render an unlit scene and every captured RGB
    frame comes back pure black. Semantic segmentation keeps working the whole
    time, because an ID pass needs no lighting, which is exactly why the
    problem hid: navigation ran fine on semantics while every photograph was
    empty.
    """
    from pxr import UsdLux

    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DistantLight) or prim.IsA(UsdLux.DomeLight):
            return False

    light = UsdLux.DistantLight.Define(stage, Sdf.Path(path))
    light.CreateIntensityAttr(3000.0)
    light.CreateAngleAttr(0.53)
    UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(
        Gf.Vec3f(-45.0, 0.0, 25.0))
    print(f"  created {path} (stage had no camera-visible light; "
          f"RGB captures would have been black)")
    return True


def ensure_physics_scene(stage):
    """A stage with no PhysicsScene has no PhysX scene behind it, so every
    scene query returns nothing -- which looks identical to 'the collision
    mesh is missing'. Create one rather than let verification fail opaquely."""
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            return prim.GetPath()
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    print("  created /physicsScene (stage had none)")
    return Sdf.Path("/physicsScene")


def tag_semantic(prim, class_name):
    try:
        from isaacsim.core.utils.semantics import add_update_semantics
        add_update_semantics(prim, class_name)
    except ImportError:
        from pxr import Semantics
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr("class")
        sem.CreateSemanticDataAttr(class_name)


# =====================================================================
# VERIFICATION -- Phase 1 exit criterion
# =====================================================================

def verify_collision_matches_authored(height_mm, cell_m, origin, n_samples,
                                      tolerance_mm, ray_height_m):
    """Cast rays at the physics surface and compare with what we authored.

    This is the whole point of the parametric approach: if the collision
    surface does not match the parameters, the labels are describing geometry
    the robot never touches. Run it once per scene configuration, not once
    per sample.
    """
    from omni.physx import get_physx_scene_query_interface

    sq = get_physx_scene_query_interface()

    def _raycast_ignoring_spot(origin_pt, max_dist):
        """Closest hit that is not part of the robot.

        Spot stands ON the surface being verified, so a sampled point near it
        casts a ray straight into the robot. That hit is legitimate physics
        and completely wrong for this measurement -- it produced a single
        1000 mm outlier among 399 exact rays, which averaged out to a
        plausible-looking 2.5 mm and failed the whole check.

        Same technique as the walker's evenness sampler: advance the ray past
        any hit belonging to the robot and try again.
        """
        origin_pt = list(origin_pt)
        remaining = float(max_dist)
        for _ in range(8):
            hit = sq.raycast_closest(origin_pt, [0.0, 0.0, -1.0], remaining)
            if not hit or not hit.get("hit"):
                return None
            collision = str(hit.get("collision", ""))
            if not collision.startswith(SPOT_PRIM_PATH):
                return hit
            advance = max(float(origin_pt[2] - hit["position"][2]) + 1e-4, 1e-4)
            origin_pt[2] -= advance
            remaining -= advance
            if remaining <= 0:
                return None
        return None
    nx, ny = height_mm.shape
    ox, oy, oz = origin

    rng = np.random.default_rng(0)
    # Stay off the border, where a ray can slip past the mesh edge.
    ii = rng.integers(2, nx - 2, n_samples)
    jj = rng.integers(2, ny - 2, n_samples)

    errors = []
    misses = 0
    for i, j in zip(ii, jj):
        x = float(i * cell_m + ox)
        y = float(j * cell_m + oy)
        expected_m = float(height_mm[i, j] / 1000.0 + oz)

        origin_pt = [x, y, expected_m + ray_height_m]
        hit = _raycast_ignoring_spot(origin_pt, ray_height_m * 2.0)
        if not hit or not hit.get("hit"):
            misses += 1
            continue
        got_m = float(hit["position"][2])
        errors.append((got_m - expected_m) * 1000.0)

    if not errors:
        print("✗ verification: no ray hits at all -- is collision enabled "
              "and the timeline playing?")
        return False

    errors = np.asarray(errors)
    max_abs = float(np.max(np.abs(errors)))
    # A couple of missed rays are expected -- a sample point can land under
    # the robot, whose body the filtered raycast steps past and then finds
    # nothing within range. What matters is the error on the rays that DID
    # hit; requiring zero misses failed a run whose 399 hits were all exact.
    miss_fraction = misses / max(n_samples, 1)
    ok = max_abs <= tolerance_mm and miss_fraction <= 0.02

    print(f"{'✓' if ok else '✗'} collision vs authored geometry:")
    print(f"    rays: {len(errors)} hit, {misses} missed")
    print(f"    error: mean {errors.mean():+.3f} mm, "
          f"max |err| {max_abs:.3f} mm  (tolerance {tolerance_mm} mm)")
    if not ok:
        print("    -> collision approximation is probably not 'none', or the "
               "timeline is not playing.")
    return ok


# =====================================================================
# MAIN
# =====================================================================

async def main(sample_dir=SAMPLE_DIR, prim_path=PRIM_PATH, origin=ORIGIN_XYZ):
    if not os.path.isdir(sample_dir):
        print(f"✗ Sample directory not found: {sample_dir}")
        return

    height_mm, labels = load_sample(sample_dir)
    cell_m = float(labels["params"]["cell_m"])
    summary = labels.get("summary", {})

    print(f"Loading {labels.get('sample_id', '?')} from {sample_dir}")
    print(f"  grid {height_mm.shape[0]}x{height_mm.shape[1]} @ {cell_m*100:.1f} cm")
    print(f"  severity {labels.get('severity', float('nan')):.2f}, "
          f"{len(labels.get('features', []))} authored features")
    if summary:
        print(f"  windows {summary.get('n_windows')}, "
              f"classes {summary.get('ada_class_counts')}")

    stage = omni.usd.get_context().get_stage()
    ensure_physics_scene(stage)
    ensure_light(stage)

    nx, ny = height_mm.shape
    triangles = 2 * (nx - 1) * (ny - 1)
    if triangles > MAX_TRIANGLES and not ALLOW_OVERSIZE:
        print(f"✗ REFUSING: {triangles:,} triangles exceeds "
              f"MAX_TRIANGLES={MAX_TRIANGLES:,}")
        print(f"  {nx * cell_m:.1f} m at {cell_m * 100:.1f} cm cells.")
        print("  This size freezes the RTX viewport, which stalls the app loop")
        print("  this import waits on. Options, best first:")
        print("    - regenerate with --cell 0.02 (quarters the count; still")
        print("      finer than Spot's ~4 cm foot)")
        print("    - regenerate with fewer --segments")
        print("    - set RENDER_TERRAIN = False for a physics-only run")
        print("    - set ALLOW_OVERSIZE = True if you really mean it")
        return

    print(f"  building mesh ({height_mm.size:,} vertices, "
          f"{triangles:,} triangles, {CHUNK_LENGTH_M:.0f} m chunks) ...")
    paths = build_chunks(stage, prim_path, height_mm, cell_m, origin,
                         CHUNK_LENGTH_M)

    for path in paths:
        chunk = stage.GetPrimAtPath(path)
        add_exact_collision(chunk)
        tag_semantic(chunk, SEMANTIC_CLASS)
        if not RENDER_TERRAIN:
            UsdGeom.Imageable(chunk).MakeInvisible()

    if not RENDER_TERRAIN:
        print("  terrain hidden from the renderer (collision unaffected)")

    print(f"✓ Built {prim_path} as {len(paths)} chunk(s) with exact trimesh "
          f"collision, tagged '{SEMANTIC_CLASS}'")

    if PLACE_SPOT:
        place_spot_at_start(stage, height_mm, cell_m, origin,
                            SPOT_PRIM_PATH, SPOT_START_OFFSET_M,
                            SPOT_STAND_HEIGHT_M, SPOT_YAW_DEG)

    if not VERIFY:
        return

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        print("  timeline was stopped -- starting it so scene queries work")
        timeline.play()

    # PhysX does not see a collider the instant the USD prim is authored: the
    # mesh has to be cooked and the scene repopulated. Raycasting in the same
    # frame as the build returns no hits at all, which looks exactly like a
    # broken collider. Let the app tick first.
    print(f"  waiting {WARMUP_FRAMES} frames for PhysX to cook the collider ...")
    for _ in range(WARMUP_FRAMES):
        await omni.kit.app.get_app().next_update_async()

    verify_collision_matches_authored(
        height_mm, cell_m, origin,
        VERIFY_SAMPLES, VERIFY_TOLERANCE_MM, VERIFY_RAY_HEIGHT_M,
    )


# Cancel a previous run if this file is re-executed, matching the pattern the
# walker scripts use.
try:
    if "_terrain_import_task" in globals():
        if _terrain_import_task is not None and not _terrain_import_task.done():
            _terrain_import_task.cancel()
except Exception:
    pass

_terrain_import_task = asyncio.ensure_future(main())
