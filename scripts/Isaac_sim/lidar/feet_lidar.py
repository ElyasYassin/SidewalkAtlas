"""Unevenness from Spot's foot placements vs the VLP-16 lidar (vs raycast truth).

Offline, plain numpy + matplotlib. Uses what sidewalk_walk_vlp16.py already
logged, so it works on any run that has foot columns in the evenness CSV and a
lidar_heightmap.npz (the autosaved one is fine, so the walk need not finish):

    python compare_feet_lidar.py                  # newest run
    python compare_feet_lidar.py C:/.../run_XXXX

What it measures
----------------
Feet: every contiguous contact phase of a foot is one STANCE. Its world (x, y)
is body position + body-frame foot offset; its z is the foot centre's world z
(median over the stance). Foot-centre height = ground + foot radius, a constant
that cancels in every detrended metric. This is ground height a real Spot
knows from kinematics, sampled only where it actually stepped.

Lidar: the height map sampled at those same stance (x, y) -> the per-point
comparison "does the lidar put the ground where the foot found it?", which is
independent of the raycast entirely.

Windows: the same 1 m trailing windows every 10 cm as the walk script, on four
profiles: raycast (truth, body centreline), feet, lidar at the feet, lidar on
the centreline. Feet span ~0.3 m laterally, so feet and lidar-at-feet windows
fit a PLANE (along-track + lateral), not a line, before peak-to-peak -- a
cross-slope must not read as unevenness.

Expect the feet to under-read narrow defects: ~10 stances per metre means a
crack or lip between footfalls is simply never touched. That is a property of
the sensor, and the summary reports it rather than hiding it.
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAPTURE_DIR = "C:/Users/Elyas/sidewalk_dataset/walk_captures"
FEET = ("fl_foot", "fr_foot", "hl_foot", "hr_foot")
FOOT_COLORS = {"fl_foot": "#1f77b4", "fr_foot": "#ff7f0e",
               "hl_foot": "#2ca02c", "hr_foot": "#d62728"}

# Same definitions as the walk script.
WINDOW_M = 1.0
WINDOW_STEP_M = 0.10
MIN_WINDOW_SPAN_M = 0.25
MIN_ROW_SAMPLES = 20
SMOOTH_MAX_MM, MILD_MAX_MM, UNEVEN_MAX_MM = 3.0, 10.0, 25.0
LIDAR_MIN_POINTS = 3
LIDAR_MIN_COVERAGE = 0.8

# Stance extraction.
MIN_STANCE_SAMPLES = 3      # >= 30 ms of contact at 100 Hz
MIN_STANCES_PER_WINDOW = 8  # a plane fit with fewer is mostly noise
MIN_STANCE_SPAN_M = 0.5

CLASSES = ("smooth", "mild", "uneven", "severe")


# ---------------------------------------------------------------- io ----
def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def latest_run():
    runs = sorted(glob.glob(os.path.join(CAPTURE_DIR, "run_*")))
    runs = [r for r in runs
            if os.path.exists(os.path.join(r, "lidar", "lidar_heightmap.npz"))
            and os.path.exists(os.path.join(r, "sidewalk_evenness_timeseries.csv"))]
    if not runs:
        sys.exit(f"No run under {CAPTURE_DIR} has both lidar_heightmap.npz and "
                 f"sidewalk_evenness_timeseries.csv")
    return runs[-1]


def load_rows(run):
    path = os.path.join(run, "sidewalk_evenness_timeseries.csv")
    if not os.path.exists(path):
        sys.exit(f"missing {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("evenness CSV is empty")
    cols = {k: np.array([fnum(r.get(k)) for r in rows]) for k in rows[0].keys()
            if k not in ("collider",)}
    needed = [f"foot_{f}_{s}" for f in FEET for s in ("x_m", "y_m", "world_z_m", "in_contact")]
    missing = [c for c in needed if c not in cols]
    if missing:
        sys.exit("This run has no foot columns (" + ", ".join(missing[:3]) + " ...).\n"
                 "It was recorded with SENSORS_ENABLED off or by an older script.")
    return cols


class LidarMap:
    """Median-height lookup on the saved 5 cm grid, 3x3 fallback like the walk."""

    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.cell = float(d["cell_m"])
        self.z = {(int(i), int(j)): (float(z), int(n))
                  for i, j, z, n in zip(d["ix"], d["iy"], d["median_z"], d["count"])}

    def at(self, x, y):
        out = np.full(len(x), np.nan)
        for k, (xx, yy) in enumerate(zip(x, y)):
            if not (np.isfinite(xx) and np.isfinite(yy)):
                continue
            i, j = int(np.floor(xx / self.cell)), int(np.floor(yy / self.cell))
            hit = self.z.get((i, j))
            if hit and hit[1] >= LIDAR_MIN_POINTS:
                out[k] = hit[0]
                continue
            zs, ns = [], 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    h = self.z.get((i + di, j + dj))
                    if h:
                        zs.append(h[0])
                        ns += h[1]
            if zs and ns >= LIDAR_MIN_POINTS:
                out[k] = float(np.median(zs))
        return out


# ----------------------------------------------------------- stances ----
def extract_stances(cols):
    """One record per contiguous contact phase of each foot."""
    bx, by = cols["body_x_m"], cols["body_y_m"]
    fx_, fy_ = cols["body_fwd_x"], cols["body_fwd_y"]
    dist = cols["distance_m"]
    on_sw = cols.get("on_sidewalk", np.ones_like(dist))
    out = []
    for foot in FEET:
        c = cols[f"foot_{foot}_in_contact"]
        if not np.isfinite(c).any():
            continue
        contact = np.nan_to_num(c) > 0.5
        edges = np.diff(np.concatenate([[0], contact.astype(int), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        fx = cols[f"foot_{foot}_x_m"]
        fy = cols[f"foot_{foot}_y_m"]
        fz = cols[f"foot_{foot}_world_z_m"]
        for a, b in zip(starts, ends):
            if b - a < MIN_STANCE_SAMPLES:
                continue
            sl = slice(a, b)
            # world xy of the foot: body + fwd * x + left * y, left = (-fwd_y, fwd_x)
            wx = bx[sl] + fx[sl] * fx_[sl] - fy[sl] * fy_[sl]
            wy = by[sl] + fx[sl] * fy_[sl] + fy[sl] * fx_[sl]
            if not np.isfinite(fz[sl]).any():
                continue
            mid = (a + b) // 2
            out.append({
                "foot": foot,
                "row_start": int(a), "row_end": int(b),
                "s_m": float(dist[mid] + np.nanmedian(fx[sl])),
                "lateral_m": float(np.nanmedian(fy[sl])),
                "x_m": float(np.nanmedian(wx)), "y_m": float(np.nanmedian(wy)),
                "foot_z_m": float(np.nanmedian(fz[sl])),
                "slip_mm": float(np.hypot(np.nanmax(wx) - np.nanmin(wx),
                                          np.nanmax(wy) - np.nanmin(wy)) * 1000),
                "on_sidewalk": int(on_sw[mid] > 0.5) if np.isfinite(on_sw[mid]) else 1,
            })
    out.sort(key=lambda r: r["s_m"])
    return out


# ----------------------------------------------------------- metrics ----
def classify(ptp_mm):
    if ptp_mm <= SMOOTH_MAX_MM:
        return "smooth"
    if ptp_mm <= MILD_MAX_MM:
        return "mild"
    if ptp_mm <= UNEVEN_MAX_MM:
        return "uneven"
    return "severe"


def line_metrics(s, z, min_n=MIN_ROW_SAMPLES):
    ok = np.isfinite(s) & np.isfinite(z)
    s, z = s[ok], z[ok]
    if s.size < min_n or s.max() - s.min() < MIN_WINDOW_SPAN_M:
        return None
    r = z - np.polyval(np.polyfit(s, z, 1), s)
    return {"ptp_mm": float(np.ptp(r) * 1000), "rms_mm": float(np.sqrt(np.mean(r ** 2)) * 1000),
            "n": int(s.size)}


def plane_metrics(s, lat, z):
    ok = np.isfinite(s) & np.isfinite(lat) & np.isfinite(z)
    s, lat, z = s[ok], lat[ok], z[ok]
    if s.size < MIN_STANCES_PER_WINDOW or s.max() - s.min() < MIN_STANCE_SPAN_M:
        return None
    A = np.stack([np.ones_like(s), s - s.mean(), lat], axis=1)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    r = z - A @ coef
    return {"ptp_mm": float(np.ptp(r) * 1000), "rms_mm": float(np.sqrt(np.mean(r ** 2)) * 1000),
            "n": int(s.size)}


def agreement(windows, ref, test):
    pairs = [w for w in windows if w[f"{ref}_class"] in CLASSES and w[f"{test}_class"] in CLASSES]
    if not pairs:
        return {"n_windows": 0}
    r = np.array([w[f"{ref}_ptp_mm"] for w in pairs])
    t = np.array([w[f"{test}_ptp_mm"] for w in pairs])
    idx = {c: k for k, c in enumerate(CLASSES)}
    rc = [w[f"{ref}_class"] for w in pairs]
    tc = [w[f"{test}_class"] for w in pairs]
    conf = {a: {b: 0 for b in CLASSES} for a in CLASSES}
    for a, b in zip(rc, tc):
        conf[a][b] += 1
    rp, tp_ = r > MILD_MAX_MM, t > MILD_MAX_MM
    tp, fp, fn = int(np.sum(rp & tp_)), int(np.sum(~rp & tp_)), int(np.sum(rp & ~tp_))
    return {
        "n_windows": len(pairs),
        "class_agreement_exact": float(np.mean([a == b for a, b in zip(rc, tc)])),
        "class_agreement_within_one": float(np.mean([abs(idx[a] - idx[b]) <= 1
                                                     for a, b in zip(rc, tc)])),
        "ptp_pearson_r": (float(np.corrcoef(r, t)[0, 1])
                          if len(r) > 2 and np.std(r) > 0 and np.std(t) > 0 else None),
        "ptp_bias_mm": float(np.mean(t - r)),
        "ptp_mae_mm": float(np.mean(np.abs(t - r))),
        "uneven_or_worse_detection": {
            "ref_positive": int(rp.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        },
        "confusion_rows_ref_cols_test": conf,
    }


# -------------------------------------------------------------- main ----
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run = args[0] if args else latest_run()
    out = os.path.join(run, "lidar", "feet_vs_lidar")
    os.makedirs(out, exist_ok=True)
    print(f"run: {run}")

    cols = load_rows(run)
    lmap = LidarMap(os.path.join(run, "lidar", "lidar_heightmap.npz"))
    stances = [s for s in extract_stances(cols) if s["on_sidewalk"]]
    if len(stances) < MIN_STANCES_PER_WINDOW:
        sys.exit(f"only {len(stances)} stances found; is foot contact being logged "
                 f"(foot_*_in_contact column all NaN/0)?")
    per_foot = ", ".join(f"{f}: {sum(1 for s in stances if s['foot'] == f)}" for f in FEET)
    print(f"  {len(stances)} on-sidewalk stances ({per_foot})")

    ss = np.array([s["s_m"] for s in stances])
    sl = np.array([s["lateral_m"] for s in stances])
    sz = np.array([s["foot_z_m"] for s in stances])
    sx = np.array([s["x_m"] for s in stances])
    sy = np.array([s["y_m"] for s in stances])
    lz_feet = lmap.at(sx, sy)
    for s, z in zip(stances, lz_feet):
        s["lidar_z_m"] = z

    # Raycast centreline + lidar on the same centreline points.
    on_sw = cols.get("on_sidewalk", np.ones_like(cols["distance_m"])) > 0.5
    d = cols["distance_m"][on_sw]
    gz = cols["ground_z_m"][on_sw]
    lz_center = lmap.at(cols["ground_x_m"][on_sw], cols["ground_y_m"][on_sw])

    # ---- point level: foot vs lidar at the same (x, y) ----------------
    ok = np.isfinite(lz_feet)
    diff = (sz[ok] - lz_feet[ok]) * 1000.0
    radius_mm = float(np.median(diff)) if ok.any() else None
    resid = diff - radius_mm if ok.any() else np.array([])
    point = {
        "stances_with_lidar": int(ok.sum()),
        "stance_coverage": float(ok.mean()),
        "foot_centre_minus_lidar_median_mm": radius_mm,
        "note": "median ~ foot radius (foot z is the foot centre); spread is lidar+foot height error",
        "residual_std_mm": float(np.std(resid)) if resid.size else None,
        "residual_mae_mm": float(np.mean(np.abs(resid))) if resid.size else None,
        "residual_p95_abs_mm": float(np.percentile(np.abs(resid), 95)) if resid.size else None,
        "median_stance_slip_mm": float(np.median([s["slip_mm"] for s in stances])),
    }

    # ---- windows ------------------------------------------------------
    windows = []
    if d.size:
        for end in np.arange(d.min() + WINDOW_M, d.max() + 1e-9, WINDOW_STEP_M):
            w = {"window_end_m": float(end)}
            a, b = np.searchsorted(d, end - WINDOW_M), np.searchsorted(d, end, side="right")
            m = line_metrics(d[a:b], gz[a:b])
            w["raycast_ptp_mm"], w["raycast_class"] = (m["ptp_mm"], classify(m["ptp_mm"])) if m else (np.nan, "no_data")
            v = np.isfinite(lz_center[a:b])
            m = line_metrics(d[a:b][v], lz_center[a:b][v]) if b > a and v.mean() >= LIDAR_MIN_COVERAGE else None
            w["lidar_center_ptp_mm"], w["lidar_center_class"] = (m["ptp_mm"], classify(m["ptp_mm"])) if m else (np.nan, "no_data")

            k = (ss > end - WINDOW_M) & (ss <= end)
            w["n_stances"] = int(k.sum())
            m = plane_metrics(ss[k], sl[k], sz[k])
            w["feet_ptp_mm"], w["feet_class"] = (m["ptp_mm"], classify(m["ptp_mm"])) if m else (np.nan, "no_data")
            v = np.isfinite(lz_feet[k])
            m = (plane_metrics(ss[k][v], sl[k][v], lz_feet[k][v])
                 if k.any() and v.mean() >= LIDAR_MIN_COVERAGE else None)
            w["lidar_feet_ptp_mm"], w["lidar_feet_class"] = (m["ptp_mm"], classify(m["ptp_mm"])) if m else (np.nan, "no_data")
            windows.append(w)

    with open(os.path.join(out, "stances.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(stances[0].keys()))
        wr.writeheader()
        wr.writerows(stances)
    if windows:
        with open(os.path.join(out, "windows.csv"), "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(windows[0].keys()))
            wr.writeheader()
            wr.writerows(windows)

    comparisons = {
        "raycast_vs_feet": agreement(windows, "raycast", "feet"),
        "raycast_vs_lidar_center": agreement(windows, "raycast", "lidar_center"),
        "feet_vs_lidar_at_feet": agreement(windows, "feet", "lidar_feet"),
        "raycast_vs_lidar_at_feet": agreement(windows, "raycast", "lidar_feet"),
    }
    summary = {
        "run": os.path.basename(run),
        "stances": len(stances),
        "stances_per_m": len(stances) / max(float(ss.max() - ss.min()), 1e-6),
        "point_level_feet_vs_lidar": point,
        "windows": len(windows),
        "thresholds_mm": {"smooth_max": SMOOTH_MAX_MM, "mild_max": MILD_MAX_MM,
                          "uneven_max": UNEVEN_MAX_MM},
        "comparisons": comparisons,
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # ---- console ------------------------------------------------------
    print(f"  stances/m: {summary['stances_per_m']:.1f}")
    if radius_mm is not None:
        print(f"  foot vs lidar at the same spot: offset {radius_mm:.1f} mm (≈ foot radius), "
              f"residual std {point['residual_std_mm']:.1f} mm, "
              f"p95 {point['residual_p95_abs_mm']:.1f} mm  (n={point['stances_with_lidar']})")
    for key, label in (("raycast_vs_feet", "raycast vs feet         "),
                       ("raycast_vs_lidar_center", "raycast vs lidar (path) "),
                       ("feet_vs_lidar_at_feet", "feet vs lidar (at feet) "),
                       ("raycast_vs_lidar_at_feet", "raycast vs lidar@feet   ")):
        a = comparisons[key]
        if not a.get("n_windows"):
            print(f"  {label}: no overlapping windows")
            continue
        det = a["uneven_or_worse_detection"]
        r = a["ptp_pearson_r"]
        rec = "n/a" if det["recall"] is None else f"{det['recall']:.0%}"
        pre = "n/a" if det["precision"] is None else f"{det['precision']:.0%}"
        print(f"  {label}: class agree {a['class_agreement_exact']:.0%} "
              f"(±1 {a['class_agreement_within_one']:.0%})  ptp r={'n/a' if r is None else f'{r:.2f}'}  "
              f"bias {a['ptp_bias_mm']:+.1f} mm  uneven+ recall {rec} precision {pre}  "
              f"(n={a['n_windows']})")

    # ---- plots --------------------------------------------------------
    off = (radius_mm or 0.0) / 1000.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax1.plot(d, gz * 1000, color="k", lw=0.8, label="raycast (truth, centreline)")
    for f in FEET:
        k = np.array([s["foot"] == f for s in stances])
        ax1.scatter(ss[k], (sz[k] - off) * 1000, s=10, color=FOOT_COLORS[f], label=f"{f} stance")
    ax1.scatter(ss, lz_feet * 1000, s=12, marker="x", color="#888", label="lidar at stance")
    ax1.set_ylabel("ground z (mm, world; feet minus foot radius)")
    ax1.legend(fontsize=8, ncol=3)
    if windows:
        we = [w["window_end_m"] for w in windows]
        for key, lab, sty in (("raycast_ptp_mm", "raycast", "-k"),
                              ("feet_ptp_mm", "feet (plane fit)", "-"),
                              ("lidar_feet_ptp_mm", "lidar at feet", "--"),
                              ("lidar_center_ptp_mm", "lidar on path", ":")):
            ax2.plot(we, [w[key] for w in windows], sty, label=lab, lw=1)
        for thr in (SMOOTH_MAX_MM, MILD_MAX_MM, UNEVEN_MAX_MM):
            ax2.axhline(thr, color="gray", ls=":", lw=0.7)
        ax2.set_ylabel("1 m window ptp (mm)")
        ax2.legend(fontsize=8)
    ax2.set_xlabel("path distance (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "profiles.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(cols["ground_x_m"], cols["ground_y_m"], color="#bbb", lw=0.6, label="body path")
    resid_all = (sz - off - lz_feet) * 1000
    sc = ax.scatter(sx, sy, c=resid_all, cmap="RdBu_r", vmin=-15, vmax=15, s=14,
                    edgecolors="k", linewidths=0.2)
    fig.colorbar(sc, ax=ax, label="foot − lidar height at stance (mm)")
    ax.set_aspect("equal")
    ax.set_title("stances, coloured by foot vs lidar disagreement")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stance_map.png"), dpi=130)
    plt.close(fig)

    print(f"\n✓ outputs in {out}: stances.csv, windows.csv, summary.json, "
          f"profiles.png, stance_map.png")


if __name__ == "__main__":
    main()