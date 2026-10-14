"""Visualize the VLP-16 output of a sidewalk_walk_vlp16.py run.

Plain Python (numpy + matplotlib), no Isaac needed:

    python view_lidar_run.py                      # newest run in CAPTURE_DIR
    python view_lidar_run.py C:/.../run_20261002_114313

Writes to <run>/lidar/viz/:
    heightmap.png       absolute height | local relief | returns per cell,
                        raycast path overlaid
    path_classes.png    evenness class along the path: raycast vs lidar
    clouds_bev.png      top-down views of the saved full-cloud snapshots
    heightmap.ply       height map as a coloured point cloud (relief colours)
    cloud_*.ply         each saved snapshot, coloured by height
The .ply files open in CloudCompare or MeshLab for orbiting in 3D.
"""
import csv
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CAPTURE_DIR = "C:/Users/Elyas/sidewalk_dataset/walk_captures"
RELIEF_WINDOW_M = 1.0      # relief = height minus the local mean over this square
RELIEF_RANGE_MM = 20.0     # colour saturates at +/- this
CLASSES = ("smooth", "mild", "uneven", "severe", "no_data")
CLASS_COLORS = ("#2e9e4f", "#e0c22e", "#e8742a", "#c8283c", "#9a9a9a")


def lidar_contents(run):
    d = os.path.join(run, "lidar")
    return {
        "lidar_dir": os.path.isdir(d),
        "heightmap": os.path.exists(os.path.join(d, "lidar_heightmap.npz")),
        "snapshots": len(glob.glob(os.path.join(d, "cloud_world_step_*.npy"))),
        "comparison": os.path.exists(os.path.join(d, "lidar_vs_raycast_windows.csv")),
    }


def latest_run():
    runs = sorted(glob.glob(os.path.join(CAPTURE_DIR, "run_*")))
    usable = [r for r in runs
              if (c := lidar_contents(r))["heightmap"] or c["snapshots"]]
    if usable:
        return usable[-1]
    print(f"No run under {CAPTURE_DIR} has any lidar output. Newest runs:")
    for r in runs[-5:]:
        c = lidar_contents(r)
        print(f"  {os.path.basename(r)}: lidar folder={c['lidar_dir']}  "
              f"heightmap={c['heightmap']}  snapshots={c['snapshots']}")
    print("\nA run without a lidar folder was made with the old script (no lidar).\n"
          "A lidar folder with no snapshots means the lidar returned no frames:\n"
          "check the walk's console for 'VLP-16 lidar authored', 'annotator attached'\n"
          "and 'Lidar point frame detected', and the 'LIDAR status' frame counts.")
    sys.exit(1)


def grid_from_snapshots(lidar_dir, cell):
    """Coarse height map from the periodic full-cloud snapshots.

    Used when the run stopped before lidar_heightmap.npz was written. The
    snapshots are every 250th frame, so the map is patchy, and they include
    walls and legs: each cell takes its 10th-percentile height so the ground
    under clutter wins.
    """
    pts = np.concatenate([np.load(p) for p in
                          sorted(glob.glob(os.path.join(lidar_dir, "cloud_world_step_*.npy")))])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    ij = np.floor(pts[:, :2] / cell).astype(np.int64)
    uniq, inv = np.unique(ij, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    groups = np.split(pts[order, 2], bounds)
    i0, j0 = uniq[:, 0].min(), uniq[:, 1].min()
    W, H = uniq[:, 0].max() - i0 + 1, uniq[:, 1].max() - j0 + 1
    if W * H > 40_000_000:
        sys.exit("Snapshot cloud spans too large an area to grid; increase cell size.")
    Z = np.full((H, W), np.nan)
    N = np.zeros((H, W))
    for (i, j), zs in zip(uniq.tolist(), groups):
        Z[j - j0, i - i0] = np.percentile(zs, 10)
        N[j - j0, i - i0] = zs.size
    return Z, N, cell, (i0, j0), (i0 * cell, (i0 + W) * cell, j0 * cell, (j0 + H) * cell)


def load_grid(npz_path):
    d = np.load(npz_path)
    ix, iy, cell = d["ix"], d["iy"], float(d["cell_m"])
    i0, j0 = ix.min(), iy.min()
    W, H = ix.max() - i0 + 1, iy.max() - j0 + 1
    Z = np.full((H, W), np.nan)
    N = np.zeros((H, W))
    Z[iy - j0, ix - i0] = d["median_z"]
    N[iy - j0, ix - i0] = d["count"]
    extent = (i0 * cell, (i0 + W) * cell, j0 * cell, (j0 + H) * cell)
    return Z, N, cell, (i0, j0), extent


def box_mean_nan(Z, k):
    """Mean over a k x k box ignoring NaNs (integral images, no scipy)."""
    valid = np.isfinite(Z).astype(float)
    Zf = np.where(np.isfinite(Z), Z, 0.0)

    def boxsum(A):
        S = np.pad(A, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        r = k // 2
        H, W = A.shape
        y0 = np.clip(np.arange(H) - r, 0, H)
        y1 = np.clip(np.arange(H) + r + 1, 0, H)
        x0 = np.clip(np.arange(W) - r, 0, W)
        x1 = np.clip(np.arange(W) + r + 1, 0, W)
        return (S[y1][:, x1] - S[y0][:, x1] - S[y1][:, x0] + S[y0][:, x0])

    cnt = boxsum(valid)
    with np.errstate(invalid="ignore", divide="ignore"):
        return boxsum(Zf) / cnt


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def write_ply(path, xyz, rgb):
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    with open(path, "w", encoding="ascii") as fh:
        fh.write("ply\nformat ascii 1.0\n"
                 f"element vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            fh.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")


def _jet(t):
    t = np.clip(np.nan_to_num(t), 0, 1)
    return np.stack([np.clip(1.5 - np.abs(4 * t - 3), 0, 1),
                     np.clip(1.5 - np.abs(4 * t - 2), 0, 1),
                     np.clip(1.5 - np.abs(4 * t - 1), 0, 1)], axis=1)


MAX_CLOUD_POINTS_3D = 900_000
CLOUD_VOXEL_M = 0.03        # merged cloud is thinned to one point per 3 cm voxel


def _turbo(t):
    """Google's Turbo colormap (polynomial fit), t in [0, 1] -> (N, 3)."""
    t = np.clip(np.nan_to_num(t), 0, 1)[:, None]
    r = np.array([0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943])
    g = np.array([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604])
    b = np.array([0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973])
    T = np.hstack([t ** k for k in range(6)])
    return np.clip(np.stack([T @ r, T @ g, T @ b], axis=1), 0, 1)


def merge_scans(scans):
    """All saved scans as one map: voxel-thinned so overlap does not pile up."""
    if not scans:
        return np.zeros((0, 3))
    P = np.concatenate(scans)
    key = np.floor(P / CLOUD_VOXEL_M).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    P = P[np.sort(first)]
    if len(P) > MAX_CLOUD_POINTS_3D:
        P = P[:: int(np.ceil(len(P) / MAX_CLOUD_POINTS_3D))]
    return P


def height_above_ground(P, tile=0.5):
    """Each point's height above the local ground (5th-percentile z per tile).

    Only used for colouring, so a coarse estimate is fine: it makes the street,
    sidewalk, curb, walls and poles read at a glance whatever the grade.
    """
    if not len(P):
        return np.zeros(0)
    ij = np.floor(P[:, :2] / tile).astype(np.int64)
    uniq, inv = np.unique(ij, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    ground = np.array([np.percentile(z, 5) for z in np.split(P[order, 2], bounds)])
    return P[:, 2] - ground[inv]


def write_3d_html(path, cloud, cloud_h, hm_xyz, hm_relief_rgb, path_xyz, feet_xyz, feet_rgb,
                  title, base_fn=None):
    """Self-contained three.js page, TRUE SCALE by default.

    Layers: lidar point cloud (merged scans, coloured by height above ground
    or by elevation), the ground height map (optional, relief-coloured),
    raycast path, foot stances. An optional vertical exaggeration (max 5x)
    stretches only local relief against the smoothed ground. Data is embedded
    as base64 arrays; three.js loads from the jsDelivr CDN.
    """
    import base64

    def b64(a, dtype):
        return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()

    ref = cloud if len(cloud) else hm_xyz
    origin = np.nanmedian(ref, axis=0) if len(ref) else np.zeros(3)

    def base(A):
        if base_fn is None or not len(A):
            return A[:, 2].copy()
        b = base_fn(A[:, 0], A[:, 1])
        return np.where(np.isfinite(b), b, A[:, 2])

    P = path_xyz if path_xyz is not None and len(path_xyz) else np.zeros((0, 3))
    F = feet_xyz if feet_xyz is not None and len(feet_xyz) else np.zeros((0, 3))
    Fc = feet_rgb if feet_xyz is not None and len(feet_xyz) else np.zeros((0, 3))

    # Two colourings for the cloud: height above local ground (default) and
    # absolute elevation, both on Turbo.
    c_hag = _turbo(np.clip(cloud_h, -0.1, 3.0) / 3.0 + 0.0) if len(cloud) else np.zeros((0, 3))
    if len(cloud):
        lo, hi = np.percentile(cloud[:, 2], [2, 98])
        c_elev = _turbo((cloud[:, 2] - lo) / max(hi - lo, 1e-6))
    else:
        c_elev = np.zeros((0, 3))

    data = {
        "CL_XYZ": b64(cloud - origin, np.float32),
        "CL_HAG": b64(c_hag * 255, np.uint8),
        "CL_ELEV": b64(c_elev * 255, np.uint8),
        "CL_BASE": b64(base(cloud) - origin[2], np.float32),
        "HM_XYZ": b64(hm_xyz - origin, np.float32),
        "HM_REL": b64(np.clip(hm_relief_rgb, 0, 1) * 255, np.uint8),
        "HM_BASE": b64(base(hm_xyz) - origin[2], np.float32),
        "PATH_XYZ": b64(P - origin, np.float32),
        "PATH_BASE": b64(base(P) - origin[2], np.float32),
        "FT_XYZ": b64(F - origin, np.float32),
        "FT_RGB": b64(np.clip(Fc, 0, 1) * 255, np.uint8),
        "FT_BASE": b64(base(F) - origin[2], np.float32),
        "TITLE": title.replace("<", "&lt;"),
        "RANGE": f"{RELIEF_RANGE_MM:.0f}",
        "NPTS": f"{len(cloud):,}",
    }
    page = _HTML_TEMPLATE
    for k, v in data.items():
        page = page.replace("%%" + k + "%%", v)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lidar 3D</title>
<style>
  html,body{margin:0;height:100%;background:#101215;color:#e6e6e6;
    font:13px/1.45 system-ui,Segoe UI,sans-serif;overflow:hidden}
  #panel{position:absolute;top:12px;left:12px;background:rgba(18,20,24,.9);
    border:1px solid #2c2f35;border-radius:8px;padding:10px 12px;width:290px}
  #panel h1{font-size:13px;margin:0 0 6px;font-weight:600}
  .sub{color:#8f969e;font-size:11px;margin:-2px 0 8px}
  label{display:flex;align-items:center;gap:6px;margin:4px 0}
  .ind{padding-left:22px}
  input[type=range]{flex:1}
  .val{min-width:56px;text-align:right;font-variant-numeric:tabular-nums}
  .legend{display:flex;align-items:center;gap:6px;margin:4px 0 2px 22px;font-size:11px;color:#aab}
  .bar{flex:1;height:8px;border-radius:4px}
  .hint{color:#8f969e;font-size:11px;margin-top:8px}
  select{background:#1e2126;color:#e6e6e6;border:1px solid #3a3e45;border-radius:4px}
  hr{border:0;border-top:1px solid #2c2f35;margin:8px 0}
  #scale{position:absolute;bottom:12px;left:12px;color:#8f969e;font-size:11px}
</style></head><body>
<div id="panel">
  <h1>%%TITLE%%</h1>
  <div class="sub">%%NPTS%% lidar points · true scale unless exaggerated</div>
  <label><input type="checkbox" id="showCL" checked> lidar point cloud</label>
  <label class="ind">colour <select id="clColor">
    <option value="hag">height above ground</option>
    <option value="elev">elevation</option></select></label>
  <div class="legend" id="legCL"><span>0 m</span>
    <div class="bar" style="background:linear-gradient(90deg,#30123b,#4686fb,#1ae4b6,#a2fc3c,#fabd2f,#e4460a,#7a0403)"></div>
    <span id="legCLmax">3 m</span></div>
  <label><input type="checkbox" id="showHM"> ground height map (5 cm cells)</label>
  <div class="legend" id="legHM"><span>−%%RANGE%% mm</span>
    <div class="bar" style="background:linear-gradient(90deg,#00f,#fff,#f00)"></div>
    <span>+%%RANGE%% mm</span></div>
  <label><input type="checkbox" id="showPath" checked> raycast path</label>
  <label id="ftRow"><input type="checkbox" id="showFT" checked> foot stances
    <span style="color:#1f77b4">FL</span> <span style="color:#ff7f0e">FR</span>
    <span style="color:#2ca02c">HL</span> <span style="color:#d62728">HR</span></label>
  <hr>
  <label>point size <input type="range" id="psize" min="1" max="8" step="0.5" value="2">
    <span class="val" id="psizeV">2</span></label>
  <label>relief × <input type="range" id="exag" min="1" max="5" step="0.5" value="1">
    <span class="val" id="exagV">true</span></label>
  <div class="hint">Drag to orbit · right-drag to pan · scroll to zoom.<br>
    "relief ×" stretches only bumps against the 1 m smoothed ground
    (1 = true scale); slopes, curbs and objects keep their real size.</div>
</div>
<div id="scale">grid: 1 m squares</div>
<script type="importmap">{"imports":{
  "three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const dec = (s, T) => { const b = atob(s), u = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
const F32 = Float32Array, U8 = Uint8Array;
const cl = dec("%%CL_XYZ%%", F32), clHag = dec("%%CL_HAG%%", U8), clElev = dec("%%CL_ELEV%%", U8),
      clB = dec("%%CL_BASE%%", F32),
      hm = dec("%%HM_XYZ%%", F32), hmRel = dec("%%HM_REL%%", U8), hmB = dec("%%HM_BASE%%", F32),
      pth = dec("%%PATH_XYZ%%", F32), pthB = dec("%%PATH_BASE%%", F32),
      ft = dec("%%FT_XYZ%%", F32), ftRgb = dec("%%FT_RGB%%", U8), ftB = dec("%%FT_BASE%%", F32);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101215);
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 3000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// World is Z-up; three.js is Y-up: (x, y, z) -> (x, z, -y).
function points(n, rgb, size) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n), 3));
  const c = new Float32Array(n); for (let i = 0; i < rgb.length; i++) c[i] = rgb[i] / 255;
  g.setAttribute("color", new THREE.BufferAttribute(c, 3));
  const p = new THREE.Points(g, new THREE.PointsMaterial({ size, sizeAttenuation: false, vertexColors: true }));
  scene.add(p); return p;
}
function place(obj, src, base, k, lift = 0) {
  const a = obj.geometry.attributes.position.array;
  for (let i = 0, j = 0; i < src.length; i += 3, j++) {
    a[i] = src[i]; a[i + 1] = base[j] + (src[i + 2] - base[j]) * k + lift; a[i + 2] = -src[i + 1];
  }
  obj.geometry.attributes.position.needsUpdate = true;
  obj.geometry.computeBoundingSphere();
}
function recolor(obj, rgb) {
  const c = obj.geometry.attributes.color.array;
  for (let i = 0; i < rgb.length; i++) c[i] = rgb[i] / 255;
  obj.geometry.attributes.color.needsUpdate = true;
}
const clPts = points(cl.length, clHag, 2);
const hmPts = points(hm.length, hmRel, 3); hmPts.visible = false;
const ftPts = points(ft.length, ftRgb, 7);
if (!ft.length) document.getElementById("ftRow").style.display = "none";
const pathGeo = new THREE.BufferGeometry();
pathGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pth.length), 3));
const pathLine = new THREE.Line(pathGeo, new THREE.LineBasicMaterial({ color: 0xffd400 }));
scene.add(pathLine);

const $ = id => document.getElementById(id);
function apply() {
  const k = +$("exag").value;
  $("exagV").textContent = k === 1 ? "true" : k + "×";
  place(clPts, cl, clB, k); place(hmPts, hm, hmB, k);
  place(pathLine, pth, pthB, k, 0.03); place(ftPts, ft, ftB, k, 0.01);
  const s = +$("psize").value; $("psizeV").textContent = s;
  clPts.material.size = s; hmPts.material.size = s * 1.5; ftPts.material.size = s * 3;
  clPts.visible = $("showCL").checked; hmPts.visible = $("showHM").checked;
  pathLine.visible = $("showPath").checked; ftPts.visible = $("showFT").checked;
  const hag = $("clColor").value === "hag";
  recolor(clPts, hag ? clHag : clElev);
  $("legCLmax").textContent = hag ? "3 m" : "high";
  $("legCL").firstElementChild.textContent = hag ? "0 m" : "low";
  $("legCL").style.display = clPts.visible ? "flex" : "none";
  $("legHM").style.display = hmPts.visible ? "flex" : "none";
}
["exag","psize","showCL","clColor","showHM","showPath","showFT"].forEach(id => $(id).addEventListener("input", apply));
apply();

// Frame the walked path (or the cloud), from a natural oblique viewpoint.
const frameObj = pth.length ? pathLine : (cl.length ? clPts : hmPts);
frameObj.geometry.computeBoundingBox();
const bb = frameObj.geometry.boundingBox, ctr = bb.getCenter(new THREE.Vector3());
const size = Math.max(bb.getSize(new THREE.Vector3()).length(), 8);
const grid = new THREE.GridHelper(Math.ceil(size * 2), Math.ceil(size * 2), 0x2a2e34, 0x1b1e22);
grid.position.set(ctr.x, bb.min.y - 0.02, ctr.z); scene.add(grid);
controls.target.copy(ctr);
camera.position.set(ctr.x + size * 0.45, ctr.y + size * 0.35, ctr.z + size * 0.6);
controls.update();

function resize() { renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); }
addEventListener("resize", resize); resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script></body></html>
"""


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run = args[0] if args else latest_run()
    lidar_dir = os.path.join(run, "lidar")
    out = os.path.join(lidar_dir, "viz")
    os.makedirs(out, exist_ok=True)
    print(f"run: {run}")
    c = lidar_contents(run)
    print(f"  heightmap={c['heightmap']}  snapshots={c['snapshots']}  "
          f"comparison={c['comparison']}")

    if c["heightmap"]:
        Z, N, cell, (i0, j0), extent = load_grid(os.path.join(lidar_dir, "lidar_heightmap.npz"))
        source = "height map"
    elif c["snapshots"]:
        print("  no lidar_heightmap.npz (walk stopped early?) -> gridding the snapshots instead")
        Z, N, cell, (i0, j0), extent = grid_from_snapshots(lidar_dir, 0.05)
        source = "snapshots only, sparse"
    else:
        sys.exit("This run has no lidar output to visualize.")
    k = max(3, int(round(RELIEF_WINDOW_M / cell)) | 1)
    relief_mm = (Z - box_mean_nan(Z, k)) * 1000.0

    rows = read_csv(os.path.join(run, "sidewalk_evenness_timeseries.csv"))
    px = np.array([fnum(r["ground_x_m"]) for r in rows])
    py = np.array([fnum(r["ground_y_m"]) for r in rows])
    pdist = np.array([fnum(r["distance_m"]) for r in rows])
    on_sw = np.array([r.get("on_sidewalk", "1") in ("1", "1.0") for r in rows], dtype=bool)

    # ---- heightmap.png -------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharex=True, sharey=True)
    panels = [
        (Z * 1000.0, "ground height (mm, world)", "viridis", None),
        (relief_mm, f"local relief (mm, minus {RELIEF_WINDOW_M:.1f} m mean)", "RdBu_r",
         (-RELIEF_RANGE_MM, RELIEF_RANGE_MM)),
        (np.where(N > 0, N, np.nan), "lidar returns per 5 cm cell", "magma", None),
    ]
    for ax, (img, title, cmap, lim) in zip(axes, panels):
        kw = dict(vmin=lim[0], vmax=lim[1]) if lim else {}
        im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap,
                       interpolation="nearest", **kw)
        if len(px):
            ax.plot(px, py, color="k", lw=0.6, label="raycast path")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        fig.colorbar(im, ax=ax, shrink=0.7)
    axes[0].set_ylabel("y (m)")
    if len(px):
        axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{os.path.basename(run)}  ({source})")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "heightmap.png"), dpi=130)
    plt.close(fig)
    print("✓ heightmap.png")

    # ---- path_classes.png ----------------------------------------------
    wins = read_csv(os.path.join(lidar_dir, "lidar_vs_raycast_windows.csv"))
    if wins and len(px):
        ends = np.array([fnum(w["window_end_m"]) for w in wins])
        sel = np.flatnonzero(on_sw)
        idx = sel[np.clip(np.searchsorted(pdist[sel], ends), 0, len(sel) - 1)]
        cmap = ListedColormap(CLASS_COLORS)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        for ax, key in zip(axes, ("raycast_class", "lidar_class")):
            c = [CLASSES.index(w[key]) if w[key] in CLASSES else 4 for w in wins]
            ax.plot(px, py, color="#cccccc", lw=0.6, zorder=1)
            ax.scatter(px[idx], py[idx], c=c, cmap=cmap, vmin=-0.5, vmax=4.5, s=14, zorder=2)
            ax.set_title(f"{key.split('_')[0]}: 1 m window class")
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        handles = [plt.Line2D([], [], marker="o", ls="", color=col, label=lab)
                   for lab, col in zip(CLASSES, CLASS_COLORS)]
        axes[1].legend(handles=handles, loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "path_classes.png"), dpi=130)
        plt.close(fig)
        print("✓ path_classes.png")

    # ---- heightmap.ply ---------------------------------------------------
    jj, ii = np.nonzero(np.isfinite(Z))
    xyz = np.stack([(i0 + ii + 0.5) * cell, (j0 + jj + 0.5) * cell, Z[jj, ii]], axis=1)
    t = np.nan_to_num(relief_mm[jj, ii]) / RELIEF_RANGE_MM
    rgb = np.ones((len(t), 3))
    rgb[t < 0, 0] = rgb[t < 0, 1] = 1 + np.clip(t[t < 0], -1, 0)
    rgb[t > 0, 1] = rgb[t > 0, 2] = 1 - np.clip(t[t > 0], 0, 1)
    write_ply(os.path.join(out, "heightmap.ply"), xyz, rgb)
    print(f"✓ heightmap.ply ({len(xyz)} cells)")

    # ---- snapshots ------------------------------------------------------
    snaps = sorted(glob.glob(os.path.join(lidar_dir, "cloud_world_step_*.npy")))
    if snaps:
        show = snaps[:: max(1, len(snaps) // 6)][:6]
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for ax, path in zip(axes.ravel(), show):
            P = np.load(path)
            zrel = P[:, 2] - np.percentile(P[:, 2], 10)
            ax.scatter(P[:, 0], P[:, 1], c=zrel, cmap="jet", vmin=0, vmax=1.5, s=0.5)
            ax.set_title(os.path.basename(path).replace("cloud_world_", "").replace(".npy", ""))
            ax.set_aspect("equal")
        for ax in axes.ravel()[len(show):]:
            ax.axis("off")
        fig.suptitle("lidar snapshots, top-down, colour = height above local ground (0-1.5 m)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "clouds_bev.png"), dpi=110)
        plt.close(fig)
        print(f"✓ clouds_bev.png ({len(show)} of {len(snaps)} snapshots)")

    print(f"\nAll outputs in {out}")

    # ---- interactive 3D page ------------------------------------------
    scans = []
    for path in sorted(glob.glob(os.path.join(lidar_dir, "cloud_world_step_*.npy"))):
        P = np.load(path).astype(float)
        scans.append(P[np.all(np.isfinite(P), axis=1)])
    cloud = merge_scans(scans)
    cloud_h = height_above_ground(cloud)
    if len(cloud):
        write_ply(os.path.join(out, "lidar_cloud.ply"), cloud,
                  _turbo(np.clip(cloud_h, -0.1, 3.0) / 3.0))
        print(f"✓ lidar_cloud.ply ({len(cloud):,} points merged from {len(scans)} scans)")
    path_xyz = None
    if len(px):
        pz = np.array([fnum(r["ground_z_m"]) for r in rows])
        keep = np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)
        path_xyz = np.stack([px[keep], py[keep], pz[keep]], axis=1)[::5]
    html = os.path.join(out, "lidar_3d.html")
    smooth = Z - relief_mm / 1000.0          # 1 m box-mean ground surface

    def base_fn(x, y):
        a = np.floor(x / cell).astype(np.int64) - i0
        b = np.floor(y / cell).astype(np.int64) - j0
        ok = (a >= 0) & (a < smooth.shape[1]) & (b >= 0) & (b < smooth.shape[0])
        res = np.full(len(x), np.nan)
        res[ok] = smooth[b[ok], a[ok]]
        return res

    feet_xyz = feet_rgb = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import compare_feet_lidar as cfl
        st = [x for x in cfl.extract_stances(cfl.load_rows(run)) if x["on_sidewalk"]]
        if st:
            fxyz = np.array([[x["x_m"], x["y_m"], x["foot_z_m"]] for x in st])
            lz = (cfl.LidarMap(os.path.join(lidar_dir, "lidar_heightmap.npz")).at(fxyz[:, 0], fxyz[:, 1])
                  if c["heightmap"] else np.full(len(st), np.nan))
            ok = np.isfinite(lz)
            offset = float(np.median(fxyz[ok, 2] - lz[ok])) if ok.any() else 0.0
            fxyz[:, 2] -= offset               # foot frame origin -> ground contact
            feet_xyz = fxyz
            feet_rgb = np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
                                 for h in (cfl.FOOT_COLORS[x["foot"]] for x in st)])
            print(f"  + {len(st)} foot stances in the 3D view")
    except SystemExit:
        pass                                    # run has no foot columns
    except Exception as exc:
        print(f"  (foot stances not added to 3D view: {exc})")

    write_3d_html(html, cloud, cloud_h, xyz, rgb, path_xyz, feet_xyz, feet_rgb,
                  f"{os.path.basename(run)} - VLP-16", base_fn=base_fn)
    print(f"✓ lidar_3d.html  (interactive 3D, opens in your browser)")
    if "--no-open" not in sys.argv:
        import webbrowser
        webbrowser.open("file:///" + os.path.abspath(html).replace(os.sep, "/"))


if __name__ == "__main__":
    main()