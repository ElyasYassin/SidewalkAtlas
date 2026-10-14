"""
Render synthetic crack scenes in Isaac Sim -> RGB, depth, exact crack masks.

STANDALONE (headless), not the Script Editor:

    C:\\isaacsim\\python.bat scripts/crack_data/render_crack_dataset.py
    C:\\isaacsim\\python.bat scripts/crack_data/render_crack_dataset.py --start 0 --count 10 --views 16
    C:\\isaacsim\\python.bat scripts/crack_data/render_crack_dataset.py --validate --count 2

Camera: Spot-like, low and angled down -- height 0.45-0.75 m, pitch 30-65 deg
below horizontal, mostly facing along the walk.

Street context (curb, gutter, road, parkway, frontage, trees, poles,
hydrants, bins, signs) is built from scene.json["street"]; the sidewalk
continues past both ends of the labelled patch so no view ends in a void.

How the mask is made exact. The surface's texture is mapped by planar UV
(u = (x - x0)/L, v = (y - y0)/W), so the colour a pixel shows is the albedo at
the (x, y) of the surface point it hits. Depth gives that point; the label
raster at the same (x, y) is therefore the label of exactly what the pixel
shows -- occlusion and perspective included. Pixels that hit anything other
than the labelled patch (street, props, sky) are rejected by comparing the
hit height with the mesh height.

IGNORE label. A crack narrower than ~0.4 px at its range is labelled but
not resolvable; such pixels get mask value 65535 so training neither rewards
nor penalises them.

--validate re-renders the same views with the label raster as an emissive
texture and reports agreement with the computed mask.

Per frame:
  rgb/<f>.png             8-bit RGB
  depth/<f>.png           uint16, exact z-depth in 0.1 mm (0 = none / >6.55 m)
  depth_stereo/<f>.png    uint16, Spot-like stereo depth at half resolution,
                          0.1 mm, 0 = invalid (noise ~ z^2, edge dropouts,
                          matching-window blur -- what a real depth cam gives)
  mask/<f>.png            uint16 crack instance id, 0 = none, 65535 = ignore
  frames.jsonl            camera K/pose, lighting, per-crack visibility
"""

import argparse
import json
import math
import os
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
ap.add_argument("--aa", type=int, default=4, help="/rtx/post/aa/op: 3 DLSS, 4 DLAA")
ap.add_argument("--validate", action="store_true")
ap.add_argument("--shard", type=int, default=0, help="render scenes i with i %% nshards == shard")
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--follow", type=int, default=0,
                help="keep picking up newly generated scenes until this many exist "
                     "and are rendered (run alongside generate_crack_scenes.py)")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                     "width": args.width, "height": args.height})

import carb  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import omni.usd  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

# Isaac's default is DLSS *Performance*: ~half-resolution render, upscaled.
# That smears texture at grazing angles and erases hairline cracks. DLAA
# renders at native resolution.
carb.settings.get_settings().set("/rtx/post/aa/op", args.aa)


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
AIM_AT_CRACK_P = 0.65
AIM_AT_DECOY_P = 0.15              # inserts / sealant / paint: hard negatives
SURFACE_TOL_M = 0.012              # hit must be this close to the mesh
MIN_VISIBLE_PX = 20                # below this a crack counts as not visible
MIN_RESOLVABLE_PX = 0.4            # crack width in pixels below which -> ignore
IGNORE = 65535
DEPTH_UNIT_M = 1e-4
# Stereo depth model (Spot body cams are stereo pairs): sigma_z = k * z^2,
# smoothing from the matching window, dropouts at depth edges.
STEREO_K = (0.004, 0.012)
TARGET_MEAN_GRAY = (95.0, 175.0)
HDRI_DIR = ("C:/isaacsim/Assets/Isaac/5.1/Isaac/Environments/Outdoor/Rivermark/"
            "dsready_content/nv_core/common_assets/environments/hdri")
HDRI_P = 0.6
EXTENT_M = 20.0                    # street geometry beyond the patch ends

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


_mat_n = [0]


def material(color=None, tex=None, rough=0.8, tint=1.0):
    """New OmniPBR under /World/Looks/street (rebuilt per scene)."""
    _mat_n[0] += 1
    mat, sh = make_omnipbr(f"/World/Looks/street/m{_mat_n[0]}")
    if tex:
        set_tex(sh, "diffuse_texture", tex)
        set_input(sh, "diffuse_tint", Sdf.ValueTypeNames.Color3f,
                  Gf.Vec3f(tint, tint, tint))
    else:
        c = np.clip(np.asarray(color if color is not None else (0.5, 0.5, 0.5)), 0, 1)
        set_input(sh, "diffuse_color_constant", Sdf.ValueTypeNames.Color3f,
                  Gf.Vec3f(*map(float, c)))
    set_input(sh, "reflection_roughness_constant", Sdf.ValueTypeNames.Float, float(rough))
    return mat


def tex_of(asset, kind="Color"):
    if not asset:
        return None
    import glob
    f = glob.glob(f"D:/textures/acg/{asset}/*_{kind}.jpg")
    return f[0].replace("\\", "/") if f else None


def plane(path, xa, xb, ya, yb, z, mat, tile_m=2.0):
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreatePointsAttr([(xa, ya, z), (xb, ya, z), (xb, yb, z), (xa, yb, z)])
    g.CreateFaceVertexCountsAttr([4])
    g.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    g.CreateNormalsAttr([(0, 0, 1)] * 4)
    g.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    st = UsdGeom.PrimvarsAPI(g).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([(xa / tile_m, ya / tile_m), (xb / tile_m, ya / tile_m),
            (xb / tile_m, yb / tile_m), (xa / tile_m, yb / tile_m)])
    bind(g.GetPrim(), mat)
    return g


def box(path, center, size, mat, rot_z=0.0):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    xf = UsdGeom.XformCommonAPI(c)
    xf.SetTranslate(Gf.Vec3d(*map(float, center)))
    xf.SetScale(Gf.Vec3f(*map(float, size)))
    if rot_z:
        xf.SetRotate(Gf.Vec3f(0, 0, float(rot_z)))
    bind(c.GetPrim(), mat)
    return c


def cylinder(path, x, y, z0, r, h, mat):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(float(r))
    c.CreateHeightAttr(float(h))
    c.CreateAxisAttr("Z")
    UsdGeom.XformCommonAPI(c).SetTranslate(Gf.Vec3d(x, y, z0 + h / 2))
    bind(c.GetPrim(), mat)
    return c


def sphere(path, x, y, z, r, mat, squash=1.0):
    s = UsdGeom.Sphere.Define(stage, path)
    s.CreateRadiusAttr(float(r))
    xf = UsdGeom.XformCommonAPI(s)
    xf.SetTranslate(Gf.Vec3d(x, y, z))
    xf.SetScale(Gf.Vec3f(1, 1, squash))
    bind(s.GetPrim(), mat)
    return s


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
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(a), 4, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(quads.astype(np.int32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nrm.reshape(-1, 3).astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    return mesh


def build_street(meta, hm):
    """Everything around the labelled patch. Returns a summary for the log."""
    remove("/World/street")
    remove("/World/Looks/street")
    UsdGeom.Xform.Define(stage, "/World/street")
    UsdGeom.Xform.Define(stage, "/World/Looks/street")
    S = "/World/street"
    st, p = meta["street"], meta["params"]
    L, W = meta["length_m"], meta["width_m"]
    xa, xb = -EXTENT_M, L + EXTENT_M
    g = p["base_gray"]

    # sidewalk continues past the patch ends (unlabelled, same surface look)
    tex = tex_of(p.get("surface_texture"))
    walk_mat = material((g, g, g), tex, 0.8, tint=1.0) if tex else material((g, g, g))
    tile_m = 2.0
    plane(f"{S}/walk_before", xa, meta["x0"] - 0.002, 0, W, 0.0, walk_mat, tile_m)
    plane(f"{S}/walk_after", meta["x0"] + L + 0.002, xb, 0, W, 0.0, walk_mat, tile_m)
    # slab body under the patch: top below the deepest crack floor
    top = float(hm.min()) - 0.003
    box(f"{S}/slab_body", (L / 2, W / 2, (top - 0.25) / 2),
        (L - 0.004, W - 0.004, top + 0.25), material((g * 0.8,) * 3))

    def ground_mat(kind, asset):
        t = tex_of(asset)
        if t:
            return material(tex=t, rough=0.95)
        cols = {"grass": (0.12, 0.22, 0.06), "soil": (0.25, 0.18, 0.11),
                "gravel": (0.3, 0.28, 0.25)}
        return material(cols.get(kind, (0.2, 0.2, 0.2)), rough=0.95)

    pw = st["parkway"]["width_m"]
    y_edge = -pw                          # road-side edge of parkway / walk
    if pw > 0:
        plane(f"{S}/parkway", xa, xb, -pw, -0.002, st["parkway"]["z_m"],
              ground_mat(st["parkway"]["kind"], st["parkway"].get("texture")), 1.5)
    road_tex = tex_of(st["road"]["texture"])
    road_mat = material(tex=road_tex, rough=0.9) if road_tex else material((0.07,) * 3, rough=0.9)
    c = st["curb"]
    if c["present"]:
        h, tw, gw = c["height_m"], c["top_width_m"], c["gutter_width_m"]
        cg = c["color_gray"]
        box(f"{S}/curb", ((xa + xb) / 2, y_edge - tw / 2, -h / 2),
            (xb - xa, tw, h), material((cg, cg, cg * 0.97)))
        plane(f"{S}/gutter", xa, xb, y_edge - tw - gw, y_edge - tw, -h + 0.01,
              material((cg * 0.9,) * 3), 2.0)
        plane(f"{S}/road", xa, xb, -30.0, y_edge - tw - gw, -h, road_mat,
              st["road"]["tile_m"])
    else:
        vw = st["verge"]["width_m"]
        plane(f"{S}/verge", xa, xb, y_edge - vw, y_edge, st["verge"]["z_m"],
              ground_mat(st["verge"]["kind"], st["verge"].get("texture")), 1.5)
        plane(f"{S}/road", xa, xb, -30.0, y_edge - vw, -0.06, road_mat,
              st["road"]["tile_m"])

    # frontage
    f = st["frontage"]
    plane(f"{S}/lawn", xa, xb, W + 0.002, 30.0, -0.01,
          ground_mat("grass", f.get("lawn_texture")), 1.5)
    yf = W + f["offset_m"]
    brick = tex_of(f.get("texture"))
    if f["kind"] in ("wall", "building"):
        hgt = f["height_m"] if f["kind"] == "wall" else max(f["height_m"], 4.0) * 2
        depth = 0.3 if f["kind"] == "wall" else 8.0
        m = material(tex=brick, rough=0.85) if brick else material((0.45, 0.3, 0.25))
        box(f"{S}/front", ((xa + xb) / 2, yf + depth / 2, hgt / 2),
            (xb - xa, depth, hgt), m)
    elif f["kind"] == "hedge":
        box(f"{S}/front", ((xa + xb) / 2, yf + 0.4, 0.5),
            (xb - xa, 0.8, rng.uniform(0.6, 1.4)), material((0.06, 0.16, 0.04), rough=0.95))
    elif f["kind"] == "fence":
        fm = material((rng.uniform(0.1, 0.6),) * 3, rough=0.6)
        hgt = float(np.clip(f["height_m"], 0.8, 1.8))
        for i, x in enumerate(np.arange(xa, xb, rng.uniform(1.8, 2.5))):
            box(f"{S}/post{i}", (x, yf, hgt / 2), (0.06, 0.06, hgt), fm)
        for j, z in enumerate((0.3, hgt - 0.1)):
            box(f"{S}/rail{j}", ((xa + xb) / 2, yf, z), (xb - xa, 0.04, 0.06), fm)

    # props
    trunk = material((0.18, 0.12, 0.08), rough=0.9)
    leaves = material((0.07, 0.17, 0.05), rough=0.9)
    metal = material((0.35, 0.36, 0.37), rough=0.4)
    for i, q in enumerate(st["props"]):
        P = f"{S}/prop{i}"
        if q["type"] == "tree":
            cylinder(P + "_trunk", q["x"], q["y"], -0.05, q["trunk_r"], q["height"], trunk)
            sphere(P + "_canopy", q["x"], q["y"], q["height"] + q["canopy_r"] * 0.5,
                   q["canopy_r"], leaves, squash=0.7)
        elif q["type"] == "pole":
            cylinder(P, q["x"], q["y"], -0.05, q["radius"], q["height"], metal)
        elif q["type"] == "hydrant":
            hm_ = material(q["color"], rough=0.5)
            cylinder(P, q["x"], q["y"], -0.02, 0.11, 0.55, hm_)
            sphere(P + "_cap", q["x"], q["y"], 0.55, 0.12, hm_)
            cylinder(P + "_nozzle", q["x"], q["y"] + 0.12, 0.35, 0.04, 0.12, hm_)
        elif q["type"] == "bin":
            cylinder(P, q["x"], q["y"], 0.0, q["radius"], q["height"],
                     material((rng.uniform(0.05, 0.3), rng.uniform(0.05, 0.3), rng.uniform(0.05, 0.3)), rough=0.5))
        elif q["type"] == "sign":
            cylinder(P, q["x"], q["y"], -0.05, 0.03, q["height"], metal)
            box(P + "_plate", (q["x"], q["y"], q["height"] - 0.2), (0.02, 0.45, 0.6),
                material([(0.8, 0.1, 0.1), (0.9, 0.9, 0.9), (0.1, 0.4, 0.15)][int(rng.integers(3))], rough=0.4))
    return {"curb": c["present"], "parkway_m": pw, "frontage": f["kind"],
            "props": [q["type"] for q in st["props"]]}


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
        UsdGeom.XformCommonAPI(sky).SetRotate(Gf.Vec3f(0, 0, float(rng.uniform(0, 360))))
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
    a fixed mid-walk view -- what a camera's auto-exposure does."""
    target = float(rng.uniform(*TARGET_MEAN_GRAY))
    L, W = meta["length_m"], meta["width_m"]
    pos = np.array([L * 0.25, W / 2, 0.6])
    fwd = np.array([math.cos(math.radians(45)), 0.0, -math.sin(math.radians(45))])
    cam = set_camera(pos, fwd, 0.0, 80.0)
    # Search log(scale) inside a shrinking bracket. The ratio^2.2 step is only
    # a proposal: saturation hides how far over we are, so a saturated
    # reading just halves the scale and tightens the upper bound.
    log_lo, log_hi = math.log(0.002), math.log(500.0)   # deep shade needs a lot
    cur = 0.0
    mean = None
    for _ in range(10):
        render(2)      # the step after a light change can show the old state
        rgb, depth = render(args.subframes)
        if rgb is None or rgb.max() == 0:
            continue
        _, on_walk = label_frame(depth, cam, meta, labels, None, hm)[:2]
        if on_walk.sum() < 1000:
            break
        mean = float(rgb.mean(axis=2)[on_walk].mean())
        if abs(mean - target) < 8:
            break
        if mean > target:
            log_hi = min(log_hi, cur)
        else:
            log_lo = max(log_lo, cur)
        nxt = cur - math.log(2.0) if mean > 245 else cur + 2.2 * math.log(target / max(mean, 1.0))
        if not (log_lo < nxt < log_hi):
            nxt = 0.5 * (log_lo + log_hi)
        scale_lights(math.exp(nxt - cur))
        cur = nxt
    light_info.update({"exposure_target_gray": target, "exposure_calib_gray": mean,
                       "exposure_scale": math.exp(cur)})
    return light_info


def build_material(scene_dir, wet, validate_tex=None):
    remove("/World/Looks/sidewalk")
    mat, sh = make_omnipbr("/World/Looks/sidewalk")
    if validate_tex:
        set_input(sh, "diffuse_color_constant", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0, 0, 0))
        set_input(sh, "enable_emission", Sdf.ValueTypeNames.Bool, True)
        set_tex(sh, "emissive_color_texture", validate_tex)
        set_input(sh, "emissive_intensity", Sdf.ValueTypeNames.Float, 400.0)
        return mat, {}
    set_tex(sh, "diffuse_texture", os.path.join(scene_dir, "albedo.jpg"))
    set_tex(sh, "normalmap_texture", os.path.join(scene_dir, "normal.png"), raw=True)
    set_tex(sh, "reflectionroughness_texture", os.path.join(scene_dir, "roughness.jpg"), raw=True)
    bump = float(rng.uniform(0.3, 0.8))
    set_input(sh, "bump_factor", Sdf.ValueTypeNames.Float, bump)
    tint = 0.72 if wet else float(rng.uniform(0.9, 1.08))
    set_input(sh, "diffuse_tint", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(tint, tint, tint))
    set_input(sh, "reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float,
              0.3 if wet else 1.0)
    set_input(sh, "reflection_roughness_constant", Sdf.ValueTypeNames.Float,
              0.12 if wet else 0.7)
    return mat, {"wet": wet, "bump_factor": bump, "diffuse_tint": tint}


# =====================================================================
# CAMERA
# =====================================================================

cam_prim = UsdGeom.Camera.Define(stage, "/World/crack_cam")
cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.03, 200.0))
render_product = rep.create.render_product("/World/crack_cam", (args.width, args.height))
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
    m = Gf.Matrix4d(right[0], right[1], right[2], 0,
                    up[0], up[1], up[2], 0,
                    -fwd[0], -fwd[1], -fwd[2], 0,
                    pos[0], pos[1], pos[2], 1)
    xf = UsdGeom.Xformable(cam_prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)
    focal = H_APERTURE_MM / (2 * math.tan(math.radians(hfov_deg) / 2))
    cam_prim.GetFocalLengthAttr().Set(float(focal))
    cam_prim.GetHorizontalApertureAttr().Set(H_APERTURE_MM)
    cam_prim.GetVerticalApertureAttr().Set(H_APERTURE_MM * args.height / args.width)
    fx = focal / H_APERTURE_MM * args.width
    K = [[fx, 0, args.width / 2], [0, fx, args.height / 2], [0, 0, 1]]
    return {"position": list(map(float, pos)), "forward": fwd.tolist(),
            "right": right.tolist(), "up": up.tolist(), "K": K,
            "hfov_deg": hfov_deg, "focal_mm": focal, "h_aperture_mm": H_APERTURE_MM}


def sample_view(meta):
    """Spot-like pose with the camera ON the sidewalk (Spot walks it; it is
    never behind the frontage wall or out in the road)."""
    W = meta["width_m"]
    for _ in range(30):
        out = _sample_view(meta)
        y = out[0][1]
        if 0.1 <= y <= W - 0.1:
            return out
    return out


def _sample_view(meta):
    """Spot-like pose; aim at a crack most of the time, at a decoy some."""
    L, W = meta["length_m"], meta["width_m"]
    lines = meta["centerlines"]
    decoys = [i["center"] for i in meta.get("inserts", []) if "center" in i]
    decoys += [d["xy"][len(d["xy"]) // 2] for d in meta.get("distractors", []) if d.get("xy")]
    u = rng.random()
    aimed = 0
    if lines and u < AIM_AT_CRACK_P:
        cl = lines[int(rng.integers(len(lines)))]
        path = cl["paths"][int(rng.integers(len(cl["paths"])))]["xy"]
        target = np.array(path[int(rng.integers(len(path)))], float)
        aimed = cl["instance_id"]
    elif decoys and u < AIM_AT_CRACK_P + AIM_AT_DECOY_P:
        target = np.array(decoys[int(rng.integers(len(decoys)))], float)
        aimed = -1
    else:
        target = np.array([rng.uniform(0.3, L - 0.3), rng.uniform(0.1, W - 0.1)])
    h = float(rng.uniform(*CAM_HEIGHT_M))
    pitch = math.radians(float(rng.uniform(*CAM_PITCH_DEG)))
    yaw = math.radians(float(rng.normal(0, CAM_YAW_JITTER_DEG / 2)))
    if rng.random() < CAM_BACKWARD_P:
        yaw += math.pi
    hfov = float(rng.uniform(*HFOV_DEG))
    yaw_off = math.radians(rng.uniform(-0.3, 0.3) * hfov)
    pitch_off = math.radians(rng.uniform(-0.3, 0.3) * hfov * args.height / args.width)
    p_aim = min(max(pitch + pitch_off, math.radians(15)), math.radians(80))
    dist = h / math.tan(p_aim)
    yaw_aim = yaw + yaw_off
    cam_xy = target - dist * np.array([math.cos(yaw_aim), math.sin(yaw_aim)])
    pos = np.array([cam_xy[0], cam_xy[1], h])
    fwd = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch),
                    -math.sin(pitch)])
    roll = float(rng.uniform(-CAM_ROLL_DEG, CAM_ROLL_DEG))
    return pos, fwd, roll, hfov, aimed, {"height_m": h, "pitch_deg": math.degrees(pitch),
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
    return (np.asarray(cam["position"]) + depth[..., None] * f
            + (U * depth)[..., None] * r - (V * depth)[..., None] * up)


def label_frame(depth, cam, meta, labels, width01, hm):
    """Instance mask from depth -> world (x, y) -> label raster.
    Returns (mask with IGNORE for unresolvable crack pixels, on_walk, raw ids)."""
    res, cell = meta["res_m"], meta["mesh_cell_m"]
    x0, y0 = meta["x0"], meta["y0"]
    ny, nx = labels.shape
    valid = np.isfinite(depth) & (depth > 0) & (depth < 100)
    pts = deproject(np.where(valid, depth, 0), cam)
    ix = np.floor((pts[..., 0] - x0) / res).astype(np.int64)
    iy = np.floor((pts[..., 1] - y0) / res).astype(np.int64)
    inside = valid & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    mx = np.clip(((pts[..., 0] - x0) / cell - 0.5).round().astype(np.int64), 0, hm.shape[1] - 1)
    my = np.clip(((pts[..., 1] - y0) / cell - 0.5).round().astype(np.int64), 0, hm.shape[0] - 1)
    on_walk = inside & (np.abs(pts[..., 2] - hm[my, mx]) < SURFACE_TOL_M)
    ids = np.zeros(depth.shape, np.uint16)
    ids[on_walk] = labels[iy[on_walk], ix[on_walk]]
    mask = ids.copy()
    if width01 is not None:
        crack = ids > 0
        w_m = np.zeros(depth.shape, np.float32)
        w_m[crack] = width01[iy[crack], ix[crack]] * 1e-4
        w_px = w_m * cam["K"][0][0] / np.maximum(depth, 1e-3)
        mask[crack & (w_px < MIN_RESOLVABLE_PX)] = IGNORE
    return mask, on_walk, ids


def stereo_depth(depth, k):
    """Spot-like stereo depth at half resolution: matching-window smoothing,
    noise growing with z^2 (spatially correlated, as block matching gives),
    dropouts at depth discontinuities and a few textureless blobs."""
    H, W = depth.shape
    d = cv2.resize(np.where(np.isfinite(depth), depth, 0), (W // 2, H // 2),
                   interpolation=cv2.INTER_AREA)
    d = cv2.medianBlur(d.astype(np.float32), 5)
    d = cv2.GaussianBlur(d, (0, 0), 1.2)
    n = cv2.GaussianBlur(rng.standard_normal(d.shape).astype(np.float32), (0, 0), 2.0)
    n /= max(n.std(), 1e-6)
    d = d + k * d ** 2 * n
    gy, gx = np.gradient(d)
    edge = np.hypot(gx, gy) > 0.05 * np.maximum(d, 0.1)
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    holes = cv2.GaussianBlur(rng.standard_normal(d.shape).astype(np.float32), (0, 0), 15)
    holes = holes > np.percentile(holes, 98.5)
    d[edge | holes | (d > 6.0) | (d <= 0)] = 0
    return d


def to_u16(depth_m):
    d = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m / DEPTH_UNIT_M, 0)
    d[d > 65535] = 0
    return d.astype(np.uint16)


def crack_visibility(meta, mask, ids, depth, cam):
    """Per crack: resolvable / ignored pixels, image bbox, and the fraction of
    its centreline that projects into the image unoccluded."""
    K = np.asarray(cam["K"])
    pos = np.asarray(cam["position"])
    f, r, up = (np.asarray(cam[k]) for k in ("forward", "right", "up"))
    H, Wd = mask.shape
    n = len(meta["cracks"]) + 1
    vis_counts = np.bincount(np.where(mask == IGNORE, 0, mask).ravel(), minlength=n)
    ign_counts = np.bincount(ids[mask == IGNORE].ravel(), minlength=n)
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
        n_px = int(vis_counts[iid])
        item = {"instance_id": iid, "visible_pixels": n_px,
                "ignore_pixels": int(ign_counts[iid]),
                "centerline_visible_fraction": float(vis.mean()),
                "visible": n_px >= MIN_VISIBLE_PX}
        sel = (ids == iid)
        if sel.any():
            rr, cc = np.nonzero(sel)
            item["bbox_xyxy"] = [int(cc.min()), int(rr.min()), int(cc.max()), int(rr.max())]
            c = np.asarray(rec["center_world_m"]) - pos
            item["range_m"] = float(np.linalg.norm(c))
            item["view_elev_deg"] = float(math.degrees(math.asin(
                min(1.0, -c[2] / max(np.linalg.norm(c), 1e-6)))))
            item["width_px_at_center"] = float(rec["mean_width_m"] * K[0, 0]
                                               / max(item["range_m"], 1e-3))
        out.append(item)
    return out


def write_label_texture(scene_dir, labels):
    p = os.path.join(scene_dir, "_validate_labels.png")
    Image.fromarray(((labels > 0).astype(np.uint8) * 255)[::-1]).convert("RGB").save(p)
    return p


def settle(n=40):
    for _ in range(n):
        app.update()


# =====================================================================
# MAIN
# =====================================================================

def main():
    for sub in ("rgb", "depth", "depth_stereo", "mask"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    import glob as _glob
    if args.validate:
        index_path = os.path.join(args.out, "validate.jsonl")
    elif args.nshards > 1:
        index_path = os.path.join(args.out, f"frames_{args.shard}.jsonl")
    else:
        index_path = os.path.join(args.out, "frames.jsonl")
    done = set()
    if not args.validate:   # resume across every shard's index
        for fp in _glob.glob(os.path.join(args.out, "frames*.jsonl")):
            with open(fp) as fh:
                done |= {json.loads(l)["scene_id"] for l in fh if l.strip()}

    def pending():
        ready = sorted(d for d in os.listdir(args.scenes)
                       if os.path.exists(os.path.join(args.scenes, d, "scene.json")))
        ready = ready[args.start:args.start + args.count]
        return [d for i, d in enumerate(ready)
                if i % args.nshards == args.shard and d not in done]

    def scene_stream():
        """Yield scenes as they become ready. With --follow, wait for the
        generator; give up after 15 idle minutes."""
        idle = 0.0
        while True:
            todo = pending()
            if todo:
                idle = 0.0
                yield todo[0]
                continue
            n_ready = len([d for d in os.listdir(args.scenes)
                           if os.path.exists(os.path.join(args.scenes, d, "scene.json"))])
            if not args.follow or n_ready >= args.follow or idle > 900:
                return
            time.sleep(20)
            idle += 20

    log = open(index_path, "a")
    UsdGeom.Xform.Define(stage, "/World/Looks")
    t_start = time.time()
    n_frames = 0
    ious, tols = [], []
    for si, sd in enumerate(scene_stream()):
        done.add(sd)
        scene_dir = os.path.join(args.scenes, sd)
        try:   # the generator may still be writing this file
            meta = json.load(open(os.path.join(scene_dir, "scene.json")))
        except (json.JSONDecodeError, OSError):
            done.discard(sd)
            time.sleep(5)
            continue
        sid = meta["scene_id"]
        lab = np.load(os.path.join(scene_dir, "labels.npz"))
        labels, width01 = lab["labels"], lab["width01mm"]
        hm = np.load(os.path.join(scene_dir, "height_mesh.npy"))
        mesh = build_sidewalk_mesh(meta, hm)
        street_info = build_street(meta, hm)
        wet = bool(rng.random() < 0.15)
        val_tex = write_label_texture(scene_dir, labels) if args.validate else None
        mat, mat_info = build_material(scene_dir, wet, val_tex)
        bind(mesh.GetPrim(), mat)
        light_info = build_lights()
        if args.validate:
            remove("/World/sun")
            stage.GetPrimAtPath("/World/sky").GetAttribute("inputs:intensity").Set(0.0)
        settle(60)
        render(2)   # first step after a scene change can return stale data
        if not args.validate:
            light_info = calibrate_exposure(meta, labels, hm, light_info)
        stereo_k = float(rng.uniform(*STEREO_K))
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
                settle(20)    # stale annotator read; retried, never saved
                continue
            mask, on_walk, ids = label_frame(depth, cam, meta, labels, width01, hm)
            # A crack in clipped highlight or crushed shadow is not visible in
            # this image, whatever the geometry says.
            clipped = (rgb.min(axis=2) >= 250) | (rgb.max(axis=2) <= 5)
            mask[(ids > 0) & clipped] = IGNORE
            if on_walk.mean() < 0.2:
                continue   # mostly street/sky: not a sidewalk view
            if not args.validate:
                walk_gray = rgb.mean(axis=2)[on_walk]
                if walk_gray.mean() > 235 or walk_gray.mean() < 25 \
                        or (walk_gray >= 254).mean() > 0.2:
                    print(f"[{sid}] exposure rejected (walk mean {walk_gray.mean():.0f})",
                          flush=True)
                    continue
            fname = f"{sid}_v{vi:03d}"
            if args.validate:
                gray = rgb.mean(axis=2)
                lit = gray[gray > 20]
                thr = 0.5 * np.percentile(lit, 99) if lit.size else 127
                rendered = gray > thr
                pred = ids > 0
                inter = (rendered & pred).sum()
                union = (rendered | pred).sum()
                iou = inter / union if union else 1.0
                from scipy import ndimage as _nd
                tol_p = (pred & _nd.binary_dilation(rendered)).sum() / max(pred.sum(), 1)
                tol_r = (rendered & _nd.binary_dilation(pred)).sum() / max(rendered.sum(), 1)
                ious.append(iou)
                tols.append(min(tol_p, tol_r))
                Image.fromarray(np.dstack([rendered * 255, pred * 255, np.zeros_like(pred)])
                                .astype(np.uint8)).save(
                    os.path.join(args.out, "mask", fname + "_validate.png"))
                log.write(json.dumps({"scene_id": sid, "file": fname, "iou": float(iou),
                                      "tol": float(min(tol_p, tol_r)),
                                      "rendered_px": int(rendered.sum()),
                                      "mask_px": int(pred.sum())}) + "\n")
                print(f"[{sid}] v{vi} IoU {iou:.3f} 1px-tol {min(tol_p, tol_r):.3f} "
                      f"(rendered {rendered.sum()}, mask {pred.sum()})", flush=True)
                vi += 1
                continue
            Image.fromarray(rgb).save(os.path.join(args.out, "rgb", fname + ".png"))
            cv2.imwrite(os.path.join(args.out, "depth", fname + ".png"), to_u16(depth))
            cv2.imwrite(os.path.join(args.out, "depth_stereo", fname + ".png"),
                        to_u16(stereo_depth(depth, stereo_k)))
            cv2.imwrite(os.path.join(args.out, "mask", fname + ".png"), mask)
            vis = crack_visibility(meta, mask, ids, depth, cam)
            log.write(json.dumps({
                "scene_id": sid, "file": fname, "view": vi,
                "aimed_instance": aimed, "camera": cam, "pose": pose,
                "lighting": light_info, "material": mat_info, "street": street_info,
                "surface": meta["params"]["surface"],
                "stereo_k": stereo_k,
                "on_walk_fraction": float(on_walk.mean()),
                "cracks": vis,
            }) + "\n")
            log.flush()
            vi += 1
            n_frames += 1
        el = time.time() - t_start
        print(f"[shard {args.shard}] scene {si + 1} {sid}: {vi} views, "
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
