"""Lidar pipeline strip: raw scan -> ground segmentation -> object clusters -> decision.

Slide-ready figures from one saved VLP-16 scan of a sidewalk_walk_vlp16.py run.
numpy + matplotlib only.

    python lidar_pipeline_strip.py                       # newest run, busiest scan
    python lidar_pipeline_strip.py C:/.../run_XXXX       # a specific run
    python lidar_pipeline_strip.py RUN --step 12345      # a specific scan
    python lidar_pipeline_strip.py RUN --radius 15       # wider view (default 12 m)

Writes <run>/lidar/pipeline/:
    strip_1x4.png          the four stages side by side (one slide row)
    strip_2x2.png          the same as a 2x2 grid
    panel_1_raw.png ... panel_4_decision.png   each stage on its own
    clusters.csv           per-object stats for the chosen scan

Every panel uses the same robot-centred frame (x forward, y left, z up from the
ground under Spot) and the same camera, at TRUE SCALE, so on a slide only the
thing that changes between panels changes.

Methods (deliberately simple, explainable on one slide each)
-------------------------------------------------------------
Ground: slope-based segmentation along each 1 deg azimuth. Walking outward
from Spot, a return is ground if its height change from the previous ground
return is within 0.20 m + tan(8 deg) x range gap. Curbs and grades pass; faces
of walls, cars and poles fail. Flat leftovers (ground seen past an obstacle)
are relabelled ground rather than reported as objects.

Clusters: non-ground points -> 0.2 m voxels -> connected components (26-
neighbourhood). Oriented box from 2-D PCA of the footprint; a size-only label.

Decision: Spot's corridor (CORRIDOR_W_M wide, CORRIDOR_LEN_M ahead) is clear up
to the first object point at body height (0.1-2.0 m above ground) inside it.
"""
import csv
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrow

CAPTURE_DIR = "C:/Users/Elyas/sidewalk_dataset/walk_captures"
LIDAR_MOUNT_OFFSET_M = (-0.10, 0.0, 0.25)   # must match sidewalk_walk_vlp16.py

AZ_BIN_DEG = 1.0
GROUND_STEP_M = 0.20        # max height change between consecutive ground returns
GROUND_SLOPE_DEG = 8.0      # ... plus this slope times the range gap between them
SELF_RADIUS_M = 0.6         # returns this close are Spot itself
VOXEL_M = 0.20
MIN_CLUSTER_PTS = 15
CORRIDOR_W_M = 1.0
CORRIDOR_LEN_M = 8.0
BODY_BAND_M = (0.10, 2.0)

BG = "#0f1115"
FG = "#e8e8e8"
MUTED = "#8b929b"
GROUND_C = "#7d8794"
PALETTE = ["#4cc9f0", "#f72585", "#ffd166", "#06d6a0", "#ff9f1c",
           "#b388ff", "#ef476f", "#90e0ef", "#c5f277", "#ff70a6"]


# ------------------------------------------------------------ loading ----
def latest_run():
    runs = sorted(r for r in glob.glob(os.path.join(CAPTURE_DIR, "run_*"))
                  if glob.glob(os.path.join(r, "lidar", "cloud_world_step_*.npy")))
    if not runs:
        sys.exit(f"No run under {CAPTURE_DIR} has lidar scans (cloud_world_step_*.npy)")
    return runs[-1]


def step_of(path):
    return int(os.path.basename(path).split("_")[-1].split(".")[0])


def load_rows(run):
    path = os.path.join(run, "sidewalk_evenness_timeseries.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def sensor_pose(run, step, rows, P):
    """(sensor_xyz, heading_rad) for the scan at `step`.

    Best: the debug record saved with the scan. Next: body pose from the
    evenness CSV row nearest in time, plus the mount offset. Last resort:
    estimate from the cloud itself (empty-return cluster at the sensor).
    """
    dbg = os.path.join(run, "lidar", f"debug_step_{step:07d}.npz")
    if os.path.exists(dbg):
        d = np.load(dbg, allow_pickle=True)
        M = d["lidar_world_M"]
        fwd = M[0, :2] / max(np.linalg.norm(M[0, :2]), 1e-9)
        return M[3, :3].astype(float), float(np.arctan2(fwd[1], fwd[0])), "debug record"
    if rows:
        steps = np.array([fnum(r.get("step")) for r in rows])
        k = int(np.nanargmin(np.abs(steps - step)))
        r = rows[k]
        b = np.array([fnum(r.get("body_x_m")), fnum(r.get("body_y_m")), fnum(r.get("body_z_m"))])
        f = np.array([fnum(r.get("body_fwd_x")), fnum(r.get("body_fwd_y"))])
        if np.all(np.isfinite(b)) and np.all(np.isfinite(f)):
            f = f / max(np.linalg.norm(f), 1e-9)
            left = np.array([-f[1], f[0]])
            ox, oy, oz = LIDAR_MOUNT_OFFSET_M
            s = np.array([b[0] + ox * f[0] + oy * left[0], b[1] + ox * f[1] + oy * left[1], b[2] + oz])
            return s, float(np.arctan2(f[1], f[0])), f"evenness CSV (step {int(steps[k])})"
    vals, cnt = np.unique(np.round(P, 2), axis=0, return_counts=True)
    print("  ⚠ no pose record for this scan; estimating sensor position from the cloud, heading +x")
    return vals[cnt.argmax()].astype(float), 0.0, "estimated"


def to_robot_frame(P, sensor, heading):
    """World -> robot frame: origin under the sensor, x forward, y left, z up."""
    c, s = np.cos(heading), np.sin(heading)
    d = P - sensor
    x = c * d[:, 0] + s * d[:, 1]
    y = -s * d[:, 0] + c * d[:, 1]
    Q = np.stack([x, y, d[:, 2]], axis=1)
    # z = 0 at the ground under/around Spot: low returns 2.5-5 m out
    rxy = np.hypot(Q[:, 0], Q[:, 1])
    ring = Q[(rxy > 2.0) & (rxy < 5.0), 2]
    z0 = np.percentile(ring, 20) if ring.size > 50 else np.percentile(Q[:, 2], 5)
    Q[:, 2] -= z0
    return Q, -z0


# ----------------------------------------------------- segmentation ----
def segment_ground(Q):
    """Slope-based ground segmentation along each azimuth (spinning-lidar standard).

    For every AZ_BIN_DEG direction, walk outward through the returns sorted by
    range, tracking the last ground point. A return is ground if its height
    change from that point is within GROUND_STEP_M + tan(GROUND_SLOPE_DEG) * gap.
    The allowance grows with the gap, so the metres between the VLP-16's far
    rings are bridged; curbs (~0.15 m) pass as steppable ground; a wall, car
    side or pole rises by much more over almost no range and fails at once.
    Returns (ground_mask, local_ground_z).
    """
    r = np.hypot(Q[:, 0], Q[:, 1])
    az = np.degrees(np.arctan2(Q[:, 1], Q[:, 0]))
    b = np.floor((az + 180.0) / AZ_BIN_DEG).astype(np.int64)
    ground = np.zeros(len(Q), bool)
    gz = np.full(len(Q), np.nan)
    tan_s = np.tan(np.radians(GROUND_SLOPE_DEG))
    order = np.lexsort((r, b))
    prev_b, pr, pz = None, 0.0, 0.0
    for k in order:
        if b[k] != prev_b:                       # new azimuth: restart under Spot
            prev_b, pr, pz = b[k], 0.0, 0.0
        if r[k] < SELF_RADIUS_M:
            continue
        dr = r[k] - pr
        dz = Q[k, 2] - pz
        allow = GROUND_STEP_M + tan_s * dr
        gz[k] = pz + 0.0
        if -allow - 0.10 <= dz <= allow:
            ground[k] = True
            pr, pz = r[k], Q[k, 2]
            gz[k] = Q[k, 2]
    return ground, gz


def cluster(Q, mask):
    """Connected components of non-ground voxels. Returns per-point label (-1 = none)."""
    idx = np.flatnonzero(mask)
    labels = np.full(len(Q), -1)
    if idx.size == 0:
        return labels
    vox = np.floor(Q[idx] / VOXEL_M).astype(np.int64)
    uniq, inv = np.unique(vox, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    vid = {tuple(v): k for k, v in enumerate(uniq.tolist())}
    parent = list(range(len(uniq)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
            if (a, b, c) > (0, 0, 0)]
    for k, v in enumerate(uniq.tolist()):
        for o in offs:
            n = vid.get((v[0] + o[0], v[1] + o[1], v[2] + o[2]))
            if n is not None:
                ra, rb = find(k), find(n)
                if ra != rb:
                    parent[ra] = rb
    root = np.array([find(k) for k in range(len(uniq))])
    _, comp = np.unique(root, return_inverse=True)
    labels[idx] = np.asarray(comp).ravel()[inv]
    return labels


def describe(Qc):
    """Oriented footprint box + size-only label for one cluster."""
    xy = Qc[:, :2]
    c = xy.mean(0)
    if len(xy) >= 3:
        w, v = np.linalg.eigh(np.cov((xy - c).T))
        axes = v[:, ::-1]
    else:
        axes = np.eye(2)
    proj = (xy - c) @ axes
    lo, hi = proj.min(0), proj.max(0)
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]) @ axes.T + c
    L, W = hi - lo
    H = Qc[:, 2].max() - max(Qc[:, 2].min(), 0.0)
    zmin, zmax = Qc[:, 2].min(), Qc[:, 2].max()
    long_side = max(L, W)
    if zmax > 2.5 and long_side > 3.0:
        cls = "wall / building"
    elif long_side < 0.6 and zmax > 1.5:
        cls = "pole"
    elif long_side < 1.0 and 1.0 < zmax < 2.1:
        cls = "person-sized"
    elif 2.5 < long_side < 6.5 and 1.0 < zmax < 2.4:
        cls = "vehicle-sized"
    else:
        cls = "object"
    dist = float(np.min(np.hypot(Qc[:, 0], Qc[:, 1])))
    return {"class": cls, "corners": corners, "zmin": zmin, "zmax": zmax,
            "length_m": float(long_side), "width_m": float(min(L, W)), "height_m": float(H),
            "distance_m": dist, "n_points": int(len(Qc)), "cx": float(c[0]), "cy": float(c[1])}


def run_pipeline(P, sensor, heading, radius):
    Q, _ = to_robot_frame(P, sensor, heading)
    keep = np.hypot(Q[:, 0], Q[:, 1]) <= radius
    Q = Q[keep]
    ground, cell_low = segment_ground(Q)
    labels = cluster(Q, ~ground)

    objs = []
    for lab in np.unique(labels[labels >= 0]):
        m = labels == lab
        Qc = Q[m]
        # Flat leftovers no region reached (ground behind obstacles) -> ground.
        if np.ptp(Qc[:, 2]) < 0.12 and Qc[:, 2].max() < 0.35:
            ground |= m
            labels[m] = -1
            continue
        if m.sum() < MIN_CLUSTER_PTS:
            labels[m] = -1
            continue
        o = describe(Qc)
        o["label"] = int(lab)
        objs.append(o)
    objs.sort(key=lambda o: o["distance_m"])
    for k, o in enumerate(objs):
        o["id"] = k + 1
        o["color"] = PALETTE[k % len(PALETTE)]

    # Corridor check.
    body = (~ground) & (labels >= 0) & (Q[:, 2] > BODY_BAND_M[0]) & (Q[:, 2] < BODY_BAND_M[1])
    inc = body & (Q[:, 0] > 0.6) & (Q[:, 0] < CORRIDOR_LEN_M) & (np.abs(Q[:, 1]) < CORRIDOR_W_M / 2)
    block = float(Q[inc, 0].min()) if inc.any() else None
    return Q, ground, labels, objs, block


# ------------------------------------------------------------ drawing ----
def style_3d(ax, Q, radius):
    """Dark, minimal 3-D axes framed tightly on the data, at true scale."""
    ax.set_facecolor(BG)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0, 0, 0, 0))
        a.line.set_color((1, 1, 1, 0.0))
        a.set_ticks([])
    ax.grid(False)
    ys = Q[:, 1]
    xr = (-2.5, radius)
    yr = (max(np.percentile(ys, 1) - 0.5, -radius * 0.6), min(np.percentile(ys, 99) + 0.5, radius * 0.6))
    zr = (-0.2, float(np.clip(np.percentile(Q[:, 2], 99.5), 2.0, 5.0)))
    ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_zlim(*zr)
    try:                                   # true scale, zoomed to fill the panel
        ax.set_box_aspect((xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]), zoom=1.45)
    except TypeError:                      # matplotlib < 3.6
        ax.set_box_aspect((xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]))
    ax.view_init(elev=22, azim=-140)
    # 1 m ground grid, and a 1 m scale bar beside Spot
    for g in np.arange(np.ceil(xr[0]), xr[1] + 1e-9, 1.0):
        ax.plot([g, g], yr, [0, 0], color=(1, 1, 1, 0.06), lw=0.5)
    for g in np.arange(np.ceil(yr[0]), yr[1] + 1e-9, 1.0):
        ax.plot(xr, [g, g], [0, 0], color=(1, 1, 1, 0.06), lw=0.5)
    ax.plot([-2.0, -1.0], [yr[0] + 0.3] * 2, [0, 0], color=FG, lw=2)
    ax.text(-1.5, yr[0] + 0.3, 0.15, "1 m", color=FG, fontsize=8, ha="center")


def draw_spot(ax):
    """Spot's body footprint and the sensor puck, so scale and pose read at a glance."""
    L, W, H = 1.1, 0.5, 0.55
    x0, y0 = -L / 2 - LIDAR_MOUNT_OFFSET_M[0], -W / 2
    xs = [x0, x0 + L, x0 + L, x0, x0]
    ys = [y0, y0, y0 + W, y0 + W, y0]
    for z in (0.35, H):
        ax.plot(xs, ys, [z] * 5, color="#f4f4f4", lw=1.0)
    for x, y in zip(xs[:4], ys[:4]):
        ax.plot([x, x], [y, y], [0.35, H], color="#f4f4f4", lw=1.0)
    ax.scatter([0], [0], [H + 0.12], s=30, c="#ffd400", depthshade=False)


def scatter(ax, Q, colors, s=1.0):
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c=colors, s=s, linewidths=0, depthshade=False)


def turbo(t):
    t = np.clip(np.nan_to_num(t), 0, 1)[:, None]
    r = np.array([0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943])
    g = np.array([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604])
    b = np.array([0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973])
    T = np.hstack([t ** k for k in range(6)])
    return np.clip(np.stack([T @ r, T @ g, T @ b], axis=1), 0, 1)


def panel_raw(ax, Q, radius):
    style_3d(ax, Q, radius)
    scatter(ax, Q, turbo(0.12 + 0.88 * np.clip(Q[:, 2], 0, 3) / 3))
    draw_spot(ax)


def panel_ground(ax, Q, ground, radius):
    style_3d(ax, Q, radius)
    scatter(ax, Q[ground], GROUND_C, s=1.0)
    scatter(ax, Q[~ground], "#ff8c42", s=1.2)
    draw_spot(ax)


def panel_clusters(ax, Q, ground, labels, objs, radius):
    style_3d(ax, Q, radius)
    scatter(ax, Q[ground], GROUND_C, s=0.8)
    rest = (~ground) & (labels < 0)
    if rest.any():
        scatter(ax, Q[rest], "#444b55", s=0.4)
    for o in objs:
        m = labels == o["label"]
        scatter(ax, Q[m], o["color"], s=1.4)
        c = o["corners"]
        for z in (o["zmin"], o["zmax"]):
            ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], [z] * 5, color=o["color"], lw=0.9)
        for x, y in c:
            ax.plot([x, x], [y, y], [o["zmin"], o["zmax"]], color=o["color"], lw=0.9)
        if o["distance_m"] < radius * 0.95:
            ax.text(o["cx"], o["cy"], o["zmax"] + 0.25, f"{o['id']}", color=o["color"],
                    fontsize=8, ha="center", fontweight="bold")
    draw_spot(ax)


def panel_decision(ax, Q, ground, labels, objs, block, radius):
    ax.set_facecolor(BG)
    ax.scatter(Q[ground, 0], Q[ground, 1], s=0.8, c=GROUND_C, linewidths=0)
    for o in objs:
        m = labels == o["label"]
        ax.scatter(Q[m, 0], Q[m, 1], s=1.2, c=o["color"], linewidths=0)
        ax.add_patch(Polygon(o["corners"], closed=True, fill=False, ec=o["color"], lw=1.1))
        ax.text(o["corners"][:, 0].max() + 0.15, o["corners"][:, 1].max(),
                f"{o['id']} {o['class']}\n{o['distance_m']:.1f} m", color=o["color"],
                fontsize=7, va="top")
    # corridor
    end = block if block is not None else CORRIDOR_LEN_M
    ax.add_patch(Rectangle((0.6, -CORRIDOR_W_M / 2), end - 0.6, CORRIDOR_W_M,
                           fc=(0.0, 0.84, 0.47, 0.22), ec="#06d6a0", lw=1.2))
    if block is not None:
        ax.add_patch(Rectangle((block, -CORRIDOR_W_M / 2), CORRIDOR_LEN_M - block, CORRIDOR_W_M,
                               fc=(0.94, 0.28, 0.44, 0.18), ec="#ef476f", lw=1.2, ls="--"))
        msg, col = f"BLOCKED at {block:.1f} m", "#ef476f"
    else:
        msg, col = f"CLEAR for {CORRIDOR_LEN_M:.0f} m", "#06d6a0"
    ax.text(0.7, CORRIDOR_W_M / 2 + 0.25, msg, color=col, fontsize=10, fontweight="bold")
    # Spot
    ax.add_patch(Rectangle((-0.55 - LIDAR_MOUNT_OFFSET_M[0], -0.25), 1.1, 0.5,
                           fc="#f4f4f4", ec="none"))
    ax.add_patch(FancyArrow(0.5, 0, 0.7, 0, width=0.06, color="#ffd400"))
    ax.set_xlim(-2.5, radius)
    ys = Q[:, 1]
    ax.set_ylim(max(np.percentile(ys, 1) - 0.5, -radius * 0.6), min(np.percentile(ys, 99) + 0.5, radius * 0.6))
    ax.set_aspect("equal")
    ax.set_xlabel("forward (m)", color=MUTED, fontsize=8)
    ax.set_ylabel("left (m)", color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color((1, 1, 1, 0.15))
    ax.grid(color=(1, 1, 1, 0.06), lw=0.5)


def titled(fig, ax, title, sub):
    ax.set_title(f"{title}\n", color=FG, fontsize=12, fontweight="bold", loc="left")
    ax.text2D(0.0, 1.0, sub, transform=ax.transAxes, color=MUTED, fontsize=8, va="bottom") \
        if hasattr(ax, "text2D") else \
        ax.text(0.0, 1.0, sub, transform=ax.transAxes, color=MUTED, fontsize=8, va="bottom")


def render(Q, ground, labels, objs, block, radius, out, run_name, step):
    n = len(Q)
    n_obj_pts = int((labels >= 0).sum())
    stages = [
        ("1 · Raw VLP-16 scan", f"{n:,} points · colour = height above ground (0-3 m)",
         lambda ax: panel_raw(ax, Q, radius), True),
        ("2 · Ground segmentation", f"{ground.mean():.0%} ground (grey) · {(~ground).mean():.0%} above ground (orange)",
         lambda ax: panel_ground(ax, Q, ground, radius), True),
        ("3 · Object clusters", f"{len(objs)} objects · {n_obj_pts:,} points · numbered by distance",
         lambda ax: panel_clusters(ax, Q, ground, labels, objs, radius), True),
        ("4 · Robot's view", f"{CORRIDOR_W_M:.1f} m corridor ahead · objects at body height",
         lambda ax: panel_decision(ax, Q, ground, labels, objs, block, radius), False),
    ]
    names = ["panel_1_raw.png", "panel_2_ground.png", "panel_3_clusters.png", "panel_4_decision.png"]

    def add(fig, spec, stage):
        title, sub, draw, is3d = stage
        spec = spec if isinstance(spec, tuple) else (spec,)
        ax = fig.add_subplot(*spec, projection="3d" if is3d else None)
        draw(ax)
        titled(fig, ax, title, sub)
        return ax

    for stage, name in zip(stages, names):
        fig = plt.figure(figsize=(9, 6.2), facecolor=BG)
        add(fig, 111, stage)
        fig.text(0.99, 0.01, f"{run_name} · step {step}", color=MUTED, fontsize=7, ha="right")
        fig.savefig(os.path.join(out, name), dpi=160, facecolor=BG, bbox_inches="tight")
        plt.close(fig)

    for layout, size, fname in (((1, 4), (30, 6.6), "strip_1x4.png"),
                                ((2, 2), (18, 12.4), "strip_2x2.png")):
        fig = plt.figure(figsize=size, facecolor=BG)
        for k, stage in enumerate(stages):
            add(fig, (layout[0], layout[1], k + 1), stage)
        fig.text(0.995, 0.005, f"{run_name} · step {step} · true scale", color=MUTED,
                 fontsize=8, ha="right")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.03, wspace=0.04, hspace=0.12)
        fig.savefig(os.path.join(out, fname), dpi=130, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)


# --------------------------------------------------------------- main ----
def main():
    args = sys.argv[1:]
    radius = 12.0
    step = None
    if "--radius" in args:
        k = args.index("--radius"); radius = float(args[k + 1]); del args[k:k + 2]
    if "--step" in args:
        k = args.index("--step"); step = int(args[k + 1]); del args[k:k + 2]
    run = args[0] if args else latest_run()
    scans = sorted(glob.glob(os.path.join(run, "lidar", "cloud_world_step_*.npy")), key=step_of)
    if not scans:
        sys.exit(f"no cloud_world_step_*.npy in {run}\\lidar")
    rows = load_rows(run)
    out = os.path.join(run, "lidar", "pipeline")
    os.makedirs(out, exist_ok=True)
    print(f"run: {run}  ({len(scans)} scans)")

    def process(path):
        P = np.load(path).astype(float)
        P = P[np.all(np.isfinite(P), axis=1)]
        s, hdg, src = sensor_pose(run, step_of(path), rows, P)
        return (P, s, hdg, src) + run_pipeline(P, s, hdg, radius)

    if step is not None:
        path = min(scans, key=lambda p: abs(step_of(p) - step))
        res = process(path)
    else:
        # Busiest scan = most separate objects: the most informative slide.
        best, res, path = -1, None, None
        cand = scans if len(scans) <= 40 else scans[:: int(np.ceil(len(scans) / 40))]
        for p in cand:
            r = process(p)
            score = len(r[7]) + 0.01 * len(r[4])
            if score > best:
                best, res, path = score, r, p
    P, s, hdg, src, Q, ground, labels, objs, block = res
    st = step_of(path)
    print(f"  scan step {st}: {len(Q):,} points within {radius:.0f} m · pose from {src}")
    print(f"  ground {ground.mean():.0%} · {len(objs)} objects · corridor "
          f"{'blocked at %.1f m' % block if block is not None else 'clear'}")
    for o in objs[:12]:
        print(f"   {o['id']:>2}. {o['class']:<15} {o['distance_m']:5.1f} m  "
              f"{o['length_m']:.2f} x {o['width_m']:.2f} x {o['height_m']:.2f} m  ({o['n_points']} pts)")

    with open(os.path.join(out, "clusters.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "class", "distance_m", "length_m", "width_m", "height_m",
                    "center_forward_m", "center_left_m", "n_points"])
        for o in objs:
            w.writerow([o["id"], o["class"], f"{o['distance_m']:.2f}", f"{o['length_m']:.2f}",
                        f"{o['width_m']:.2f}", f"{o['height_m']:.2f}", f"{o['cx']:.2f}",
                        f"{o['cy']:.2f}", o["n_points"]])

    render(Q, ground, labels, objs, block, radius, out, os.path.basename(run), st)
    print(f"\n✓ {out}: strip_1x4.png, strip_2x2.png, panel_1..4, clusters.csv")


if __name__ == "__main__":
    main()