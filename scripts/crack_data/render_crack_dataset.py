"""
Render synthetic crack scenes in Isaac Sim -> RGB, depth, exact crack masks.

STANDALONE (headless), not the Script Editor:

    C:\\isaacsim\\python.bat scripts/crack_data/render_crack_dataset.py
    C:\\isaacsim\\python.bat scripts/crack_data/render_crack_dataset.py --start 0 --count 10 --views 16
    C:\\isaacsim\\python.bat scripts/crack_data/render_crack_dataset.py --validate --count 2

Camera: Spot-like, low and angled down -- height 0.45-0.75 m, pitch 30-65 deg
below horizontal, mostly facing along the walk.

How the mask is made exact. The surface's texture is mapped by planar UV
(u = (x - x0)/L, v = (y - y0)/W), so the colour a pixel shows is the albedo at
the (x, y) of the surface point it hits. Depth gives that point; the label
raster at the same (x, y) is therefore the label of exactly what the pixel
shows -- occlusion and perspective included. Pixels that hit anything other
than the sidewalk (ground, shadow casters, sky) are rejected by comparing the
hit height with the mesh height.

--validate re-renders the same views with the label raster as the texture
(white crack on black, emissive) and reports IoU against the computed mask.
That checks the claim above against the renderer itself.

Per frame:  rgb/<f>.png  depth/<f>.npz (float16 m)  mask/<f>.png (uint16
crack instance id, 0 = none)  and one JSON line in frames.jsonl with camera
intrinsics/pose, lighting, and per-crack visibility.
"""

import argparse
import json
import math
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", default="D:/sidewalk_crack_synth/scenes")
ap.add_argument("--out", default="D:/sidewalk_crack_synth/render")
ap.add_argument("--start", type=int, default=0, help="index into sorted scene list")
ap.add_argument("--count", type=int, default=100000)
ap.add_argument("--views", type=int, default=16)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--subframes", type=int, default=6)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--validate", action="store_true")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                     "width": args.width, "height": args.height})

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import omni.usd  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402


# =====================================================================
# CONFIG
# =====================================================================

CAM_HEIGHT_M = (0.45, 0.75)
CAM_PITCH_DEG = (30.0, 65.0)       # below horizontal
CAM_ROLL_DEG = 3.0
CAM_YAW_JITTER_DEG = 30.0
CAM_BACKWARD_P = 0.2               # sometimes look back along the walk
HFOV_DEG = (70.0, 95.0)
H_APERTURE_MM = 20.955
AIM_AT_CRACK_P = 0.8
SURFACE_TOL_M = 0.012              # hit must be this close to the mesh
MIN_VISIBLE_PX = 20                # below this a crack counts as not visible
# Exposure: per scene, lights are scaled so the sidewalk's mean sRGB level
# lands on a random target -- what a camera's auto-exposure does. Absolute
# light units then do not matter, so HDRI skies of any brightness are safe.
TARGET_MEAN_GRAY = (95.0, 175.0)
HDRI_DIR = ("C:/isaacsim/Assets/Isaac/5.1/Isaac/Environments/Outdoor/Rivermark/"
            "dsready_content/nv_core/common_assets/environments/hdri")
HDRI_P = 0.6
SURROUND_TEXTURES = {
    "asphalt": ["D:/textures/Asphalt015_1K_Color.jpg",
                "D:/textures/Asphalt016_1K_Color.jpg",
                "D:/textures/Road006_1K_Color.jpg",
                "D:/textures/Road007_1K_Color.jpg"],
}
SURROUND_COLORS = {
    "grass": [(0.10, 0.20, 0.05), (0.16, 0.24, 0.08), (0.20, 0.22, 0.10)],
    "soil": [(0.22, 0.16, 0.10), (0.30, 0.22, 0.14)],
    "gravel": [(0.32, 0.30, 0.27), (0.25, 0.24, 0.22)],
    "asphalt": [(0.08, 0.08, 0.08)],
}

ctx = omni.usd.get_context()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
rng = np.random.default_rng(args.seed)


# =====================================================================
# STAGE HELPERS
# =====================================================================

def make_omnipbr(path):
    omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url="OmniPBR.mdl",
                              mtl_name="OmniPBR", mtl_path=path)
    return UsdShade.Material(stage.GetPrimAtPath(path)), \
        UsdShade.Shader(stage.GetPrimAtPath(path + "/Shader"))


def set_input(shader, name, vtype, value):
    shader.CreateInput(name, vtype).Set(value)


def set_tex(shader, name, path, raw=False):
    inp = shader.CreateInput(name, Sdf.ValueTypeNames.Asset)
    inp.Set(Sdf.AssetPath(path.replace("\\", "/")))
    if raw:
        inp.GetAttr().SetColorSpace("raw")


def bind(prim, mat):
    UsdShade.MaterialBindingAPI(prim).Bind(
        mat, UsdShade.Tokens.strongerThanDescendants)


def remove(path):
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)


def build_sidewalk_mesh(meta, hm):
    """Grid mesh at mesh-cell centres, per-vertex UV and smooth normals."""
    remove("/World/sidewalk")
    cell = meta["mesh_cell_m"]
    ny, nx = hm.shape
    x0, y0 = meta["x0"], meta["y0"]
    L, W = meta["length_m"], meta["width_m"]
    xs = x0 + (np.arange(nx) + 0.5) * cell
    ys = y0 + (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X, Y, hm], axis=-1).reshape(-1, 3).astype(np.float32)
    uv = np.stack([(X - x0) / L, (Y - y0) / W], axis=-1).reshape(-1, 2)
    gy, gx = np.gradient(hm.astype(np.float64), cell)
    nrm = np.dstack([-gx, -gy, np.ones_like(gx)])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    idx = np.arange(ny * nx).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    quads = np.stack([a, b, c, d], axis=1).ravel()
    mesh = UsdGeom.Mesh.Define(stage, "/World/sidewalk")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
        np.full(len(a), 4, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(
        quads.astype(np.int32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(
        nrm.reshape(-1, 3).astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    return mesh


def build_ground(surround):
    """Large plane around the sidewalk, slightly lower (a curb-less verge)."""
    remove("/World/ground")
    remove("/World/Looks/ground")
    g = UsdGeom.Mesh.Define(stage, "/World/ground")
    s = 30.0
    z = -float(rng.uniform(0.005, 0.03))
    g.CreatePointsAttr([(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)])
    g.CreateFaceVertexCountsAttr([4])
    g.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    g.CreateNormalsAttr([(0, 0, 1)] * 4)
    g.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    st = UsdGeom.PrimvarsAPI(g).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([(0, 0), (s, 0), (s, s), (0, s)])     # ~2 m texture repeat
    mat, sh = make_omnipbr("/World/Looks/ground")
    texs = SURROUND_TEXTURES.get(surround, [])
    texs = [t for t in texs if os.path.exists(t)]
    if texs and rng.random() < 0.8:
        set_tex(sh, "diffuse_texture", texs[int(rng.integers(len(texs)))])
        set_input(sh, "texture_scale", Sdf.ValueTypeNames.Float2,
                  Gf.Vec2f(0.5, 0.5))
    else:
        col = SURROUND_COLORS[surround][int(rng.integers(
            len(SURROUND_COLORS[surround])))]
        col = np.clip(np.asarray(col) * rng.uniform(0.8, 1.2), 0, 1)
        set_input(sh, "diffuse_color_constant", Sdf.ValueTypeNames.Color3f,
                  Gf.Vec3f(*map(float, col)))
    set_input(sh, "reflection_roughness_constant", Sdf.ValueTypeNames.Float, 0.9)
    bind(g.GetPrim(), mat)
    return z


def build_slab_body(meta, ground_z, hm):
    """Solid body under the sidewalk sheet so its edge reads as a slab face,
    not a floating sheet over a black gap."""
    remove("/World/slab_body")
    remove("/World/Looks/slab_body")
    L, W = meta["length_m"], meta["width_m"]
    # Top must sit below the deepest crack floor, or it fills the cracks.
    top = float(hm.min()) - 0.003
    bottom = min(ground_z, top) - 0.05
    c = UsdGeom.Cube.Define(stage, "/World/slab_body")
    c.CreateSizeAttr(1.0)
    xf = UsdGeom.XformCommonAPI(c)
    xf.SetTranslate(Gf.Vec3d(meta["x0"] + L / 2, meta["y0"] + W / 2,
                             (top + bottom) / 2))
    xf.SetScale(Gf.Vec3f(L - 0.004, W - 0.004, top - bottom))
    mat, sh = make_omnipbr("/World/Looks/slab_body")
    g = float(meta["params"]["base_gray"]) * 0.8
    set_input(sh, "diffuse_color_constant", Sdf.ValueTypeNames.Color3f,
              Gf.Vec3f(g, g, g))
    set_input(sh, "reflection_roughness_constant", Sdf.ValueTypeNames.Float, 0.85)
    bind(c.GetPrim(), mat)


def build_shadow_casters(meta):
    """Off-walk boxes/poles that throw shadows across the walk."""
    remove("/World/casters")
    UsdGeom.Xform.Define(stage, "/World/casters")
    n = int(rng.integers(0, 4)) if rng.random() < 0.6 else 0
    for i in range(n):
        side = rng.choice([-1, 1])
        y = side * (meta["width_m"] / 2 + rng.uniform(0.3, 2.0))
        x = rng.uniform(-1.0, meta["length_m"] + 1.0)
        c = UsdGeom.Cube.Define(stage, f"/World/casters/c{i}")
        c.CreateSizeAttr(1.0)
        sx, sy, sz = (rng.uniform(0.08, 1.5), rng.uniform(0.08, 1.5),
                      rng.uniform(0.5, 4.0))
        xf = UsdGeom.XformCommonAPI(c)
        xf.SetTranslate(Gf.Vec3d(x, y, sz / 2))
        xf.SetScale(Gf.Vec3f(sx, sy, sz))
        xf.SetRotate(Gf.Vec3f(0, 0, float(rng.uniform(0, 90))))
    return n


def build_lights():
    """Sun + sky dome, randomised. Always present: the offscreen camera has
    no viewport headlight, so a stage without these renders black."""
    remove("/World/sun")
    remove("/World/sky")
    overcast = rng.random() < 0.3
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    elev = float(rng.uniform(15, 80))
    azim = float(rng.uniform(0, 360))
    sun_i = float(rng.uniform(200, 700) if overcast else rng.uniform(1800, 5000))
    sun.CreateIntensityAttr(sun_i)
    sun.CreateAngleAttr(float(rng.uniform(0.5, 2.0) if not overcast else 8.0))
    temp = float(rng.uniform(4200, 6800))
    sun.CreateEnableColorTemperatureAttr(True)
    sun.CreateColorTemperatureAttr(temp)
    # DistantLight shines along its local -Z; tilt it to the sun elevation.
    UsdGeom.XformCommonAPI(sun).SetRotate(
        Gf.Vec3f(90.0 - elev, 0.0, azim), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    sky = UsdLux.DomeLight.Define(stage, "/World/sky")
    dome_i = float(rng.uniform(1000, 2200) if overcast else rng.uniform(300, 1100))
    sky.CreateIntensityAttr(dome_i)
    hdri = None
    hdris = sorted(f for f in os.listdir(HDRI_DIR)) if os.path.isdir(HDRI_DIR) else []
    if hdris and rng.random() < HDRI_P:
        hdri = os.path.join(HDRI_DIR, hdris[int(rng.integers(len(hdris)))])
        sky.CreateTextureFileAttr(Sdf.AssetPath(hdri.replace("\\", "/")))
        sky.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
        UsdGeom.XformCommonAPI(sky).SetRotate(
            Gf.Vec3f(0, 0, float(rng.uniform(0, 360))))
        tint = np.ones(3)
    else:
        tint = np.clip(np.array([0.85, 0.92, 1.0]) * rng.uniform(0.9, 1.1, 3), 0, 1)
    sky.CreateColorAttr(Gf.Vec3f(*map(float, tint)))
    return {"overcast": overcast, "sun_elev_deg": elev, "sun_azim_deg": azim,
            "sun_intensity": sun_i, "sun_temp_k": temp,
            "dome_intensity": dome_i, "dome_tint": tint.tolist(),
            "hdri": os.path.basename(hdri) if hdri else None}


def scale_lights(k):
    for path in ("/World/sun", "/World/sky"):
        a = stage.GetPrimAtPath(path).GetAttribute("inputs:intensity")
        a.Set(float(a.Get() * k))


def calibrate_exposure(meta, labels, hm, light_info):
    """Scale lights until the sidewalk's mean gray hits a random target, from
    a fixed mid-walk view. sRGB ~ linear^(1/2.2), so intensity scales by the
    ratio^2.2. A few iterations; tonemapping makes it only roughly linear."""
    target = float(rng.uniform(*TARGET_MEAN_GRAY))
    L = meta["length_m"]
    pos = np.array([L * 0.25, 0.0, 0.6])
    fwd = np.array([math.cos(math.radians(45)), 0.0, -math.sin(math.radians(45))])
    cam = set_camera(pos, fwd, 0.0, 80.0)
    # Search log(scale) inside a shrinking bracket. The ratio^2.2 step is only
    # a proposal: saturation (mean near 255) hides how far over we are, so a
    # saturated reading just halves the scale and tightens the upper bound.
    log_lo, log_hi = math.log(0.01), math.log(20.0)
    cur = 0.0          # log of the scale currently applied
    mean = None
    for _ in range(10):
        render(2)      # the step after a light change can show the old state
        rgb, depth = render(args.subframes)
        if rgb is None or rgb.max() == 0:
            continue   # stale or empty frame: measure again, don't react
        _, on_walk = label_frame(depth, cam, meta, labels, hm)
        if on_walk.sum() < 1000:
            break
        mean = float(rgb.mean(axis=2)[on_walk].mean())
        if abs(mean - target) < 8:
            break
        if mean > target:
            log_hi = min(log_hi, cur)
        else:
            log_lo = max(log_lo, cur)
        if mean > 245:
            nxt = cur - math.log(2.0)
        else:
            nxt = cur + 2.2 * math.log(target / max(mean, 1.0))
        # keep strictly inside the bracket (bisect if the proposal leaves it)
        if not (log_lo < nxt < log_hi):
            nxt = 0.5 * (log_lo + log_hi)
        scale_lights(math.exp(nxt - cur))
        cur = nxt
    light_info.update({"exposure_target_gray": target,
                       "exposure_calib_gray": mean,
                       "exposure_scale": math.exp(cur)})
    return light_info


def build_material(scene_dir, wet, validate_tex=None):
    remove("/World/Looks/sidewalk")
    mat, sh = make_omnipbr("/World/Looks/sidewalk")
    if validate_tex:
        # Label texture as pure emission: what the camera sees IS the label.
        set_input(sh, "diffuse_color_constant", Sdf.ValueTypeNames.Color3f,
                  Gf.Vec3f(0, 0, 0))
        set_input(sh, "enable_emission", Sdf.ValueTypeNames.Bool, True)
        set_tex(sh, "emissive_color_texture", validate_tex)
        set_input(sh, "emissive_intensity", Sdf.ValueTypeNames.Float, 400.0)
        return mat, {}
    set_tex(sh, "diffuse_texture", os.path.join(scene_dir, "albedo.jpg"))
    set_tex(sh, "normalmap_texture", os.path.join(scene_dir, "normal.png"),
            raw=True)
    set_tex(sh, "reflectionroughness_texture",
            os.path.join(scene_dir, "roughness.jpg"), raw=True)
    bump = float(rng.uniform(0.3, 0.8))
    set_input(sh, "bump_factor", Sdf.ValueTypeNames.Float, bump)
    tint = 0.72 if wet else float(rng.uniform(0.9, 1.08))
    set_input(sh, "diffuse_tint", Sdf.ValueTypeNames.Color3f,
              Gf.Vec3f(tint, tint, tint))
    infl = 0.3 if wet else 1.0
    set_input(sh, "reflection_roughness_texture_influence",
              Sdf.ValueTypeNames.Float, infl)
    set_input(sh, "reflection_roughness_constant", Sdf.ValueTypeNames.Float,
              0.12 if wet else 0.7)
    return mat, {"wet": wet, "bump_factor": bump, "diffuse_tint": tint}


# =====================================================================
# CAMERA
# =====================================================================

cam_prim = UsdGeom.Camera.Define(stage, "/World/crack_cam")
cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.03, 200.0))
render_product = rep.create.render_product("/World/crack_cam",
                                           (args.width, args.height))
ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
ann_rgb.attach(render_product)
ann_depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
ann_depth.attach(render_product)


def set_camera(pos, fwd, roll_deg, hfov_deg):
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    r = math.radians(roll_deg)
    right, up = (math.cos(r) * right + math.sin(r) * up,
                 -math.sin(r) * right + math.cos(r) * up)
    # USD camera: +X right, +Y up, looks along -Z. Row-vector matrix.
    m = Gf.Matrix4d(
        right[0], right[1], right[2], 0,
        up[0], up[1], up[2], 0,
        -fwd[0], -fwd[1], -fwd[2], 0,
        pos[0], pos[1], pos[2], 1)
    xf = UsdGeom.Xformable(cam_prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)
    focal = H_APERTURE_MM / (2 * math.tan(math.radians(hfov_deg) / 2))
    cam_prim.GetFocalLengthAttr().Set(float(focal))
    cam_prim.GetHorizontalApertureAttr().Set(H_APERTURE_MM)
    cam_prim.GetVerticalApertureAttr().Set(
        H_APERTURE_MM * args.height / args.width)
    fx = focal / H_APERTURE_MM * args.width
    K = [[fx, 0, args.width / 2], [0, fx, args.height / 2], [0, 0, 1]]
    return {"position": list(map(float, pos)), "forward": fwd.tolist(),
            "right": right.tolist(), "up": up.tolist(), "K": K,
            "hfov_deg": hfov_deg, "focal_mm": focal,
            "h_aperture_mm": H_APERTURE_MM}


def sample_view(meta):
    """Spot-like pose; aim at a crack most of the time."""
    L, W = meta["length_m"], meta["width_m"]
    lines = meta["centerlines"]
    if lines and rng.random() < AIM_AT_CRACK_P:
        cl = lines[int(rng.integers(len(lines)))]
        path = cl["paths"][int(rng.integers(len(cl["paths"])))]["xy"]
        target = np.array(path[int(rng.integers(len(path)))], float)
        aimed = cl["instance_id"]
    else:
        target = np.array([rng.uniform(0.3, L - 0.3),
                           rng.uniform(-W / 2 + 0.1, W / 2 - 0.1)])
        aimed = 0
    h = float(rng.uniform(*CAM_HEIGHT_M))
    pitch = math.radians(float(rng.uniform(*CAM_PITCH_DEG)))
    yaw = math.radians(float(rng.normal(0, CAM_YAW_JITTER_DEG / 2)))
    if rng.random() < CAM_BACKWARD_P:
        yaw += math.pi
    hfov = float(rng.uniform(*HFOV_DEG))
    # Put the target somewhere in the middle 60% of the image, not dead centre.
    yaw_off = math.radians(rng.uniform(-0.3, 0.3) * hfov)
    pitch_off = math.radians(rng.uniform(-0.3, 0.3) * hfov * args.height / args.width)
    p_aim = min(max(pitch + pitch_off, math.radians(15)), math.radians(80))
    dist = h / math.tan(p_aim)
    yaw_aim = yaw + yaw_off
    cam_xy = target - dist * np.array([math.cos(yaw_aim), math.sin(yaw_aim)])
    pos = np.array([cam_xy[0], cam_xy[1], h])
    fwd = np.array([math.cos(yaw) * math.cos(pitch),
                    math.sin(yaw) * math.cos(pitch), -math.sin(pitch)])
    roll = float(rng.uniform(-CAM_ROLL_DEG, CAM_ROLL_DEG))
    return pos, fwd, roll, hfov, aimed, {"height_m": h,
                                         "pitch_deg": math.degrees(pitch),
                                         "yaw_deg": math.degrees(yaw) % 360,
                                         "roll_deg": roll}


# =====================================================================
# RENDER + LABEL
# =====================================================================

def render(n_subframes):
    rep.orchestrator.step(rt_subframes=n_subframes, delta_time=0.0)
    rgb = np.asarray(ann_rgb.get_data())
    depth = np.asarray(ann_depth.get_data())
    if rgb.ndim != 3 or depth.ndim != 2:
        return None, None
    return rgb[..., :3].copy(), depth.astype(np.float32)


def deproject(depth, cam):
    H, Wd = depth.shape
    K = np.asarray(cam["K"])
    u = (np.arange(Wd) + 0.5 - K[0, 2]) / K[0, 0]
    v = (np.arange(H) + 0.5 - K[1, 2]) / K[1, 1]
    U, V = np.meshgrid(u, v)
    f, r, up = (np.asarray(cam[k]) for k in ("forward", "right", "up"))
    d = depth[..., None]
    pts = (np.asarray(cam["position"]) + d * f
           + (U * depth)[..., None] * r - (V * depth)[..., None] * up)
    return pts


def label_frame(depth, cam, meta, labels, hm):
    """Instance mask from depth -> world (x, y) -> label raster."""
    res, cell = meta["res_m"], meta["mesh_cell_m"]
    x0, y0 = meta["x0"], meta["y0"]
    ny, nx = labels.shape
    valid = np.isfinite(depth) & (depth > 0) & (depth < 100)
    pts = deproject(np.where(valid, depth, 0), cam)
    ix = np.floor((pts[..., 0] - x0) / res).astype(np.int64)
    iy = np.floor((pts[..., 1] - y0) / res).astype(np.int64)
    inside = valid & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    mx = np.clip(((pts[..., 0] - x0) / cell - 0.5).round().astype(np.int64),
                 0, hm.shape[1] - 1)
    my = np.clip(((pts[..., 1] - y0) / cell - 0.5).round().astype(np.int64),
                 0, hm.shape[0] - 1)
    on_walk = inside & (np.abs(pts[..., 2] - hm[my, mx]) < SURFACE_TOL_M)
    mask = np.zeros(depth.shape, np.uint16)
    mask[on_walk] = labels[iy[on_walk], ix[on_walk]]
    return mask, on_walk


def crack_visibility(meta, mask, depth, cam):
    """Per crack: visible pixels, image bbox, and the fraction of its
    centreline that projects into the image unoccluded."""
    K = np.asarray(cam["K"])
    pos = np.asarray(cam["position"])
    f, r, up = (np.asarray(cam[k]) for k in ("forward", "right", "up"))
    H, Wd = mask.shape
    counts = np.bincount(mask.ravel(), minlength=len(meta["cracks"]) + 1)
    out = []
    for rec, cl in zip(meta["cracks"], meta["centerlines"]):
        iid = rec["instance_id"]
        xy = np.concatenate([np.asarray(p["xy"]) for p in cl["paths"]])
        P = np.c_[xy, np.full(len(xy), rec["center_world_m"][2])] - pos
        z = P @ f
        uu = K[0, 0] * (P @ r) / np.maximum(z, 1e-6) + K[0, 2]
        vv = -K[1, 1] * (P @ up) / np.maximum(z, 1e-6) + K[1, 2]
        ok = (z > 0.03) & (uu >= 0) & (uu < Wd) & (vv >= 0) & (vv < H)
        vis = np.zeros(len(xy), bool)
        if ok.any():
            ui, vi = uu[ok].astype(int), vv[ok].astype(int)
            vis[ok] = np.abs(depth[vi, ui] - z[ok]) < 0.03
        n_px = int(counts[iid]) if iid < len(counts) else 0
        item = {"instance_id": iid, "visible_pixels": n_px,
                "centerline_visible_fraction": float(vis.mean()),
                "visible": n_px >= MIN_VISIBLE_PX}
        if n_px:
            rr, cc = np.nonzero(mask == iid)
            item["bbox_xyxy"] = [int(cc.min()), int(rr.min()),
                                 int(cc.max()), int(rr.max())]
            c = np.asarray(rec["center_world_m"]) - pos
            item["range_m"] = float(np.linalg.norm(c))
            item["view_elev_deg"] = float(math.degrees(math.asin(
                min(1.0, -c[2] / max(np.linalg.norm(c), 1e-6)))))
        out.append(item)
    return out


def write_label_texture(scene_dir, labels):
    p = os.path.join(scene_dir, "_validate_labels.png")
    img = ((labels > 0).astype(np.uint8) * 255)[::-1]
    Image.fromarray(img).convert("RGB").save(p)
    return p


def settle(n=40):
    for _ in range(n):
        app.update()


# =====================================================================
# MAIN
# =====================================================================

def main():
    for sub in ("rgb", "depth", "mask"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    scene_dirs = sorted(d for d in os.listdir(args.scenes)
                        if os.path.exists(os.path.join(args.scenes, d, "scene.json")))
    scene_dirs = scene_dirs[args.start:args.start + args.count]
    index_path = os.path.join(args.out, "frames.jsonl" if not args.validate
                              else "validate.jsonl")
    done = set()
    if os.path.exists(index_path) and not args.validate:
        with open(index_path) as fh:
            done = {json.loads(l)["scene_id"] for l in fh if l.strip()}
    log = open(index_path, "a")
    UsdGeom.Xform.Define(stage, "/World/Looks")
    t_start = time.time()
    n_frames = 0
    ious, tols = [], []
    for si, sd in enumerate(scene_dirs):
        scene_dir = os.path.join(args.scenes, sd)
        meta = json.load(open(os.path.join(scene_dir, "scene.json")))
        sid = meta["scene_id"]
        if sid in done:
            continue
        labels = np.load(os.path.join(scene_dir, "labels.npz"))["labels"]
        hm = np.load(os.path.join(scene_dir, "height_mesh.npy"))
        mesh = build_sidewalk_mesh(meta, hm)
        ground_z = build_ground(meta["params"]["surround"])
        build_slab_body(meta, ground_z, hm)
        n_cast = build_shadow_casters(meta)
        wet = bool(rng.random() < 0.15)
        val_tex = write_label_texture(scene_dir, labels) if args.validate else None
        mat, mat_info = build_material(scene_dir, wet, val_tex)
        bind(mesh.GetPrim(), mat)
        light_info = build_lights()
        if args.validate:
            remove("/World/sun")
            stage.GetPrimAtPath("/World/sky").GetAttribute(
                "inputs:intensity").Set(0.0)
        settle(60)
        render(2)   # first step after a scene change can return stale data
        if not args.validate:
            light_info = calibrate_exposure(meta, labels, hm, light_info)
        vi = 0
        tries = 0
        while vi < args.views and tries < args.views * 3:
            tries += 1
            pos, fwd, roll, hfov, aimed, pose = sample_view(meta)
            cam = set_camera(pos, fwd, roll, hfov)
            rgb, depth = render(args.subframes)
            if rgb is None:
                settle(10)
                continue
            if rgb.max() < 10 or rgb.std() < 2:
                print(f"[{sid}] black/flat frame rejected "
                      f"(max {rgb.max()}, std {rgb.std():.1f})", flush=True)
                settle(20)
                continue
            mask, on_walk = label_frame(depth, cam, meta, labels, hm)
            if on_walk.mean() < 0.2:
                continue   # mostly verge/sky: not a sidewalk view
            if not args.validate:
                walk_gray = rgb.mean(axis=2)[on_walk]
                if walk_gray.mean() > 235 or walk_gray.mean() < 25 \
                        or (walk_gray >= 254).mean() > 0.2:
                    print(f"[{sid}] exposure rejected (walk mean "
                          f"{walk_gray.mean():.0f})", flush=True)
                    continue
            fname = f"{sid}_v{vi:03d}"
            if args.validate:
                # Threshold at half the lit level: a pixel half covered by
                # crack sits on the true edge. Report strict IoU and a 1 px
                # boundary-tolerant agreement (render anti-aliasing and
                # texture filtering blur exactly one pixel of edge).
                gray = rgb.mean(axis=2)
                lit = gray[gray > 20]
                thr = 0.5 * np.percentile(lit, 99) if lit.size else 127
                rendered = gray > thr
                pred = mask > 0
                inter = (rendered & pred).sum()
                union = (rendered | pred).sum()
                iou = inter / union if union else 1.0
                from scipy import ndimage as _nd
                tol_p = (pred & _nd.binary_dilation(rendered)).sum() / max(pred.sum(), 1)
                tol_r = (rendered & _nd.binary_dilation(pred)).sum() / max(rendered.sum(), 1)
                ious.append(iou)
                tols.append(min(tol_p, tol_r))
                Image.fromarray(np.dstack([rendered * 255, pred * 255,
                                           np.zeros_like(pred)]).astype(np.uint8)
                                ).save(os.path.join(args.out, "mask",
                                                    fname + "_validate.png"))
                log.write(json.dumps({"scene_id": sid, "file": fname,
                                      "iou": float(iou),
                                      "rendered_px": int(rendered.sum()),
                                      "mask_px": int(pred.sum())}) + "\n")
                print(f"[{sid}] v{vi} IoU {iou:.3f} 1px-tol {min(tol_p, tol_r):.3f} "
                      f"(rendered {rendered.sum()}, mask {pred.sum()})", flush=True)
                vi += 1
                continue
            Image.fromarray(rgb).save(os.path.join(args.out, "rgb", fname + ".png"))
            np.savez_compressed(os.path.join(args.out, "depth", fname + ".npz"),
                                depth=depth.astype(np.float16))
            Image.fromarray(mask).save(os.path.join(args.out, "mask", fname + ".png"))
            vis = crack_visibility(meta, mask, depth, cam)
            log.write(json.dumps({
                "scene_id": sid, "file": fname, "view": vi,
                "aimed_instance": aimed, "camera": cam, "pose": pose,
                "lighting": light_info, "material": mat_info,
                "n_shadow_casters": n_cast,
                "on_walk_fraction": float(on_walk.mean()),
                "cracks": vis,
            }) + "\n")
            log.flush()
            vi += 1
            n_frames += 1
        el = time.time() - t_start
        print(f"scene {si + 1}/{len(scene_dirs)} {sid}: {vi} views, "
              f"{n_frames} frames total, {el / max(n_frames, 1):.1f}s/frame, "
              f"{el / 60:.1f} min", flush=True)
    log.close()
    if args.validate and ious:
        print(f"VALIDATE: {len(ious)} views, IoU mean {np.mean(ious):.4f}, "
              f"min {np.min(ious):.4f}; 1px-tolerant agreement mean "
              f"{np.mean(tols):.4f}, min {np.min(tols):.4f}", flush=True)


try:
    main()
finally:
    app.close()
