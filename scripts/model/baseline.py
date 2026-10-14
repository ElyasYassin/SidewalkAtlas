"""
baseline.py

First end-to-end model: predict sidewalk unevenness from proprioception alone.

Runs OFFLINE. Loads walk CSVs produced by
spot_walk_with_evenness_autosave.py, windows them by distance, extracts
features from ONLY the signals a real Spot could report, and regresses the
raycast-derived roughness label.

WHAT THIS IS
------------
Pipeline validation, not a scientific result. The corpus is ~42 m of walking
across two Meshy sidewalk sections, which is roughly 40 independent 1 m
windows. That is far too small to conclude anything about detectability. What
it CAN do, today:

  - prove the sensor -> feature -> label path works end to end
  - show whether the signal exists at all
  - surface feature-extraction bugs now rather than after 500 collection runs
  - establish the honest-evaluation harness before there is any incentive
    to be dishonest with it

LEAKAGE IS THE WHOLE GAME
-------------------------
The label comes from a raycast at the robot's own body position. Several
logged columns are therefore on the LABEL side of the problem and must never
become features -- a model given them scores near-perfectly and has learned
nothing:

  ground_{x,y,z}_m        the label source itself
  body_{x,y,z}_m          world pose; a real robot does not know its world Z
  body_clearance_m        body_z - ground_z, i.e. the label minus a constant
  foot_*_world_z_m        world-frame foot height IS terrain height
  body_vertical_accel     double-differenced body pose, same provenance
  distance_m, collider    indexing / bookkeeping

The subtle one is foot_*_world_z_m. It looks like a foot sensor, it sits in
the foot block, and it is a direct readout of the ground the label measures.
Body-RELATIVE foot position is the legitimate version, and is what the real
SDK reports.

Usage:
    python scripts/model/baseline.py
    python scripts/model/baseline.py --window 1.0 --stride 0.25
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# =====================================================================
# CONFIG
# =====================================================================

WALK_CAPTURES = "C:/Users/Elyas/sidewalk_dataset/walk_captures"

FEET = ("fl_foot", "fr_foot", "hl_foot", "hr_foot")
JOINTS = (
    "fl_hx", "fl_hy", "fl_kn", "fr_hx", "fr_hy", "fr_kn",
    "hl_hx", "hl_hy", "hl_kn", "hr_hx", "hr_hy", "hr_kn",
)

# Channels a real Spot could report. Everything else is excluded by default;
# see the module docstring on why that matters more than usual here.
IMU_CHANNELS = [
    "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z",
    "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z",
]
FOOT_KINEMATIC_CHANNELS = [
    f"foot_{f}_{axis}" for f in FEET for axis in ("x_m", "y_m", "z_m")
]
FOOT_CONTACT_CHANNELS = (
    [f"foot_{f}_in_contact" for f in FEET]
    + [f"foot_{f}_force_n" for f in FEET]
)
JOINT_CHANNELS = [
    f"joint_{j}_{q}" for j in JOINTS
    for q in ("pos_rad", "vel_rads", "effort_nm")
]

# Explicitly named so a future reader sees the exclusions as a decision
# rather than an oversight.
FORBIDDEN_SUBSTRINGS = (
    "ground_", "body_x", "body_y", "body_z", "body_clearance",
    "world_z", "body_vertical_accel", "distance_m", "collider",
    "on_sidewalk", "time_s", "step",
)

# ---------------------------------------------------------------------
# What a real Spot can actually report
# ---------------------------------------------------------------------
# Checked against bosdyn/api/robot_state.proto rather than assumed:
#
#   JointState.position / .velocity / .load      -> our 36 joint channels
#   FootState.foot_position_rt_body              -> our 12 foot kinematics
#   FootState.contact (CONTACT_MADE/LOST)        -> our 4 contact flags
#   ImuState.Packet.acceleration_rt_odom_in_link_frame
#   ImuState.Packet.angular_velocity_rt_odom_in_link_frame
#                                                -> our 6 IMU channels
#
# FootState has NO contact-force field. It carries contact state and a
# TerrainState (friction estimate, slip, ground normal, visual ground
# penetration), but no force magnitude. So per-foot force is simulation-only:
# a model that depends on it cannot be deployed.
#
# CONVENTION WARNING: our sim IMU reports PROPER acceleration -- 9.81 on Z at
# rest, gravity included, as a physical accelerometer does. Spot's field is
# named acceleration_rt_odom_in_link_frame, which reads as gravity-COMPENSATED
# (~0 on Z at rest). If that is right, sim and real differ by gravity rotated
# into the body frame. The packet ships odom_rot_link, so it is convertible --
# but it has to be done deliberately. Verify on first contact with hardware.
SIM_ONLY_CHANNELS = [f"foot_{f}_force_n" for f in FEET]

REAL_AVAILABLE_CHANNELS = [
    c for c in (IMU_CHANNELS + FOOT_KINEMATIC_CHANNELS
                + FOOT_CONTACT_CHANNELS + JOINT_CHANNELS)
    if c not in SIM_ONLY_CHANNELS
]

MODALITIES = {
    "imu": IMU_CHANNELS,
    "foot_kinematics": FOOT_KINEMATIC_CHANNELS,
    "foot_contact": FOOT_CONTACT_CHANNELS,
    "joints": JOINT_CHANNELS,
    "DEPLOYABLE (real-available)": REAL_AVAILABLE_CHANNELS,
}


# =====================================================================
# LOADING
# =====================================================================

def load_run(path: str) -> Optional[Dict[str, np.ndarray]]:
    """Read one run CSV into a dict of float arrays (collider kept as str)."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    out: Dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        if key == "collider":
            out[key] = np.array([r.get(key, "") for r in rows], dtype=object)
            continue
        values = []
        for r in rows:
            try:
                values.append(float(r.get(key, "")))
            except (TypeError, ValueError):
                values.append(np.nan)
        out[key] = np.asarray(values, dtype=float)
    return out


def run_is_usable(run: Dict[str, np.ndarray]) -> Tuple[bool, str]:
    """Reject runs recorded before the sensor fixes landed.

    These are the physical assertions the plan calls for: gravity on a
    walking robot averages ~9.81 on Z, and four feet carrying a ~25 kg robot
    sum to a few hundred newtons. Both failed silently at various points
    during bring-up, which is exactly why they are checked rather than
    assumed.
    """
    if "imu_lin_acc_z" not in run:
        return False, "no IMU columns"
    az = float(np.nanmean(run["imu_lin_acc_z"]))
    if not np.isfinite(az) or abs(az - 9.81) > 1.0:
        return False, f"IMU gravity {az:.2f} != 9.81"

    total = np.zeros_like(run["imu_lin_acc_z"])
    for f in FEET:
        key = f"foot_{f}_force_n"
        if key not in run:
            return False, "no contact force columns"
        total = total + np.nan_to_num(run[key])
    mean_force = float(np.nanmean(total))
    if mean_force < 100.0:
        return False, f"contact force {mean_force:.1f} N too low"

    return True, f"gravity {az:.2f}, force {mean_force:.0f} N"


def sidewalk_mask(run: Dict[str, np.ndarray]) -> np.ndarray:
    """Samples whose raycast landed on sidewalk rather than city street.

    Older runs predate the on_sidewalk flag, so fall back to the collider
    path, which every run has.
    """
    n = len(run["time_s"])
    if "on_sidewalk" in run:
        flags = run["on_sidewalk"]
        if np.isfinite(flags).any():
            return np.nan_to_num(flags, nan=1.0) > 0.5
    if "collider" in run:
        return np.array(
            ["assembly_City/ce_Context_City_Streets" not in str(c)
             and str(c) != "" for c in run["collider"]]
        )
    return np.ones(n, dtype=bool)


def section_of(run: Dict[str, np.ndarray],
               walkway: Optional[Dict] = None,
               origin_x: float = 0.0) -> np.ndarray:
    """Which independent surface each sample sits on -- the split key.

    Two windows from the same surface share the same physical defects, so
    they are not independent however far apart along the path they fall.

    On PARAMETRIC terrain the right unit is the generated segment, not the
    collider prim. Collider prims are 20 m import chunks chosen for PhysX's
    convenience; they cut across segment boundaries arbitrarily. Segments are
    the parameter draws -- one severity, one set of authored faults -- which
    is what independence actually means here. Splitting by chunk would put
    the same parameter draw on both sides of the split and quietly inflate
    the score.

    On the city scene there are no segments and the collider prim IS the
    distinct surface, so the path component is correct there.
    """
    n = len(run["time_s"])

    if walkway is not None and "ground_x_m" in run:
        segments = walkway["segments"]
        wx = run["ground_x_m"] - origin_x
        out = []
        for x in wx:
            hit = "unknown"
            for seg in segments:
                if seg["x_start_m"] <= x < seg["x_end_m"]:
                    hit = seg["segment_id"]
                    break
            out.append(hit)
        return np.asarray(out, dtype=object)

    if "collider" not in run:
        return np.array(["unknown"] * n, dtype=object)
    out = []
    for c in run["collider"]:
        parts = str(c).split("/")
        out.append(parts[3] if len(parts) > 3 else "unknown")
    return np.asarray(out, dtype=object)


# =====================================================================
# LABEL
# =====================================================================

def detrended_ptp_rms(distance_m: np.ndarray,
                      ground_z_m: np.ndarray) -> Tuple[float, float]:
    """Roughness after removing local grade -- mirrors the definition used by
    both the walker's live classifier and the terrain generator's labels."""
    if distance_m.size < 4:
        return np.nan, np.nan
    span = float(distance_m[-1] - distance_m[0])
    if span < 0.25:
        return np.nan, np.nan
    coeff = np.polyfit(distance_m, ground_z_m, 1)
    residual = (ground_z_m - np.polyval(coeff, distance_m)) * 1000.0
    return float(np.ptp(residual)), float(np.sqrt(np.mean(residual ** 2)))


# =====================================================================
# WINDOWING + FEATURES
# =====================================================================

def channel_features(values: np.ndarray, name: str) -> Dict[str, float]:
    """Summary statistics for one channel over one window."""
    v = values[np.isfinite(values)]
    if v.size < 3:
        return {}
    feats = {
        f"{name}__mean": float(v.mean()),
        f"{name}__std": float(v.std()),
        f"{name}__ptp": float(np.ptp(v)),
        f"{name}__rms": float(np.sqrt(np.mean(v ** 2))),
    }
    # Contact channels carry gait timing, which plain moments miss entirely:
    # how often the foot is down, and how often it changes state.
    if name.endswith("_in_contact"):
        binary = (v > 0.5).astype(float)
        feats[f"{name}__duty"] = float(binary.mean())
        feats[f"{name}__transitions"] = float(
            np.abs(np.diff(binary)).sum() / max(len(binary), 1))
    return feats


def build_windows(run: Dict[str, np.ndarray],
                  channels: Sequence[str],
                  window_m: float,
                  stride_m: float,
                  walkway: Optional[Dict] = None,
                  origin_x: float = 0.0) -> Tuple[List[Dict[str, float]],
                                            List[float], List[float],
                                            List[str], List[float]]:
    """Slice one run into distance windows; return features, labels, sections."""
    keep = sidewalk_mask(run)
    if keep.sum() < 10:
        return [], [], [], [], []

    distance = run["distance_m"][keep]
    ground = run["ground_z_m"][keep]
    time_s = run["time_s"][keep]
    sections = section_of(run, walkway, origin_x)[keep]
    sub = {c: run[c][keep] for c in channels if c in run}

    feats_list, ptp_list, rms_list, sect_list, speed_list = [], [], [], [], []

    start_d = float(distance[0])
    end_d = float(distance[-1])
    d = start_d
    while d + window_m <= end_d:
        lo, hi = d, d + window_m
        idx = np.where((distance >= lo) & (distance < hi))[0]
        d += stride_m
        if idx.size < 20:
            continue

        # A window must sit on ONE physical surface. One spanning two meshes
        # measures the join between them, not a property of either.
        win_sections = sections[idx]
        uniq = set(win_sections.tolist())
        if len(uniq) != 1:
            continue

        ptp, rms = detrended_ptp_rms(distance[idx], ground[idx])
        if not np.isfinite(ptp):
            continue

        # Positive control: forward speed over the window. Gait cadence
        # encodes this directly, so a feature pipeline that cannot recover it
        # is broken -- and no amount of extra data would fix that. Separating
        # "features are wrong" from "not enough data" is the whole point.
        span_t = float(time_s[idx][-1] - time_s[idx][0])
        speed = (float(distance[idx][-1] - distance[idx][0]) / span_t
                 if span_t > 1e-6 else np.nan)

        feats: Dict[str, float] = {}
        for c, values in sub.items():
            feats.update(channel_features(values[idx], c))
        if not feats:
            continue

        feats_list.append(feats)
        ptp_list.append(ptp)
        rms_list.append(rms)
        speed_list.append(speed)
        sect_list.append(next(iter(uniq)))

    return feats_list, ptp_list, rms_list, sect_list, speed_list


def vectorize(feature_dicts: Sequence[Dict[str, float]]
              ) -> Tuple[np.ndarray, List[str]]:
    names = sorted({k for d in feature_dicts for k in d})
    X = np.zeros((len(feature_dicts), len(names)), dtype=float)
    for i, d in enumerate(feature_dicts):
        for j, n in enumerate(names):
            X[i, j] = d.get(n, np.nan)
    # Column-median imputation; a channel missing for a whole window should
    # not silently become a zero, which is a real value for most of these.
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad]) if (~bad).any() else 0.0
    return X, names


# =====================================================================
# MODELS
# =====================================================================

def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge. Deliberately the first model: with ~40 independent
    windows and hundreds of features, anything fancier is fitting noise."""
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
    A[-1, -1] -= alpha  # do not penalise the intercept
    return np.linalg.solve(A, Xb.T @ y)


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ w


def standardize(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return ((train - mu) / sd,) + tuple((o - mu) / sd for o in others)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    resid = y_true - y_pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "mae_mm": float(np.abs(resid).mean()),
        "rmse_mm": float(np.sqrt((resid ** 2).mean())),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"),
        "baseline_mae_mm": float(np.abs(y_true - y_true.mean()).mean()),
    }


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", default=WALK_CAPTURES)
    ap.add_argument("--window", type=float, default=1.0, help="metres")
    ap.add_argument("--stride", type=float, default=0.25, help="metres")
    ap.add_argument("--alpha", type=float, default=10.0, help="ridge penalty")
    ap.add_argument("--walkway", default=None,
                    help="parametric walkway sample dir; splits by generated "
                         "segment instead of by import chunk")
    ap.add_argument("--walkway-origin-x", type=float, default=1100.0,
                    help="world X the walkway was imported at")
    args = ap.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(
        args.captures, "run_*", "sidewalk_evenness_timeseries.csv")))
    if not csv_paths:
        print(f"No run CSVs under {args.captures}")
        return 1

    walkway = None
    if args.walkway:
        with open(os.path.join(args.walkway, "labels.json"),
                  encoding="utf-8") as fh:
            walkway = json.load(fh)
        print(f"walkway: {len(walkway['segments'])} segments")

    all_channels = (IMU_CHANNELS + FOOT_KINEMATIC_CHANNELS
                    + FOOT_CONTACT_CHANNELS + JOINT_CHANNELS)

    # Guard the leakage rule in code, not just in the docstring.
    for c in all_channels:
        assert not any(bad in c for bad in FORBIDDEN_SUBSTRINGS), \
            f"feature channel {c} matches a forbidden pattern"

    print("=== runs ===")
    feats, ptps, rmss, sects, speeds, origins = [], [], [], [], [], []
    for path in csv_paths:
        run_name = os.path.basename(os.path.dirname(path))
        run = load_run(path)
        if run is None:
            print(f"  skip {run_name}: empty")
            continue
        ok, why = run_is_usable(run)
        if not ok:
            print(f"  skip {run_name}: {why}")
            continue

        f, p, r, s, sp = build_windows(run, all_channels, args.window,
                                       args.stride, walkway,
                                       args.walkway_origin_x)
        print(f"  use  {run_name}: {len(f):4d} windows  ({why})")
        feats += f
        ptps += p
        rmss += r
        sects += s
        speeds += sp
        origins += [run_name] * len(f)

    if not feats:
        print("\nNo usable windows.")
        return 1

    X, names = vectorize(feats)
    y = np.asarray(ptps, dtype=float)
    sections = np.asarray(sects, dtype=object)
    origins = np.asarray(origins, dtype=object)

    print(f"\n=== dataset ===")
    print(f"  windows {X.shape[0]}, features {X.shape[1]}")
    print(f"  label (detrended ptp): mean {y.mean():.2f} mm, "
          f"std {y.std():.2f}, range {y.min():.2f}-{y.max():.2f} mm")
    independent = 0.0
    for s in set(sections.tolist()):
        m = sections == s
        independent += float(np.ptp(np.arange(m.sum())) * args.stride / args.window)
    print(f"  sections: {dict((s, int((sections == s).sum())) for s in set(sections.tolist()))}")
    print(f"  NOTE windows overlap {100 * (1 - args.stride / args.window):.0f}%; "
          f"independent samples are far fewer than the window count.")

    uniq_sections = sorted(set(sections.tolist()))
    if len(uniq_sections) < 2:
        print("\nOnly one sidewalk section present -- no honest split is "
              "possible. Collect a second surface before trusting any number.")
        return 1

    print(f"\n=== evaluation ===")
    print("Two splits are reported on purpose. The gap between them IS the "
          "leakage you would otherwise report as a result.\n")

    # --- dishonest split: random ------------------------------------
    rng = np.random.default_rng(0)
    perm = rng.permutation(X.shape[0])
    cut = int(0.75 * X.shape[0])
    tr, te = perm[:cut], perm[cut:]
    Xtr, Xte = standardize(X[tr], X[te])
    w = ridge_fit(Xtr, y[tr], args.alpha)
    m_random = evaluate(y[te], ridge_predict(Xte, w))
    print(f"  RANDOM split   (leaky: overlapping windows on both sides)")
    print(f"    MAE {m_random['mae_mm']:6.2f} mm   RMSE {m_random['rmse_mm']:6.2f}   "
          f"R2 {m_random['r2']:+.3f}   (predict-the-mean MAE "
          f"{m_random['baseline_mae_mm']:.2f})")

    # --- honest split: hold out a whole physical surface -------------
    print(f"\n  HELD-OUT SECTION (train on one mesh, test on another)")
    for held in uniq_sections:
        te_mask = sections == held
        tr_mask = ~te_mask
        if tr_mask.sum() < 20 or te_mask.sum() < 10:
            print(f"    {held:16s} skipped (too few windows)")
            continue
        Xtr, Xte = standardize(X[tr_mask], X[te_mask])
        w = ridge_fit(Xtr, y[tr_mask], args.alpha)
        m = evaluate(y[te_mask], ridge_predict(Xte, w))
        print(f"    hold out {held:14s} n_test={int(te_mask.sum()):4d}  "
              f"MAE {m['mae_mm']:6.2f} mm   R2 {m['r2']:+.3f}   "
              f"(predict-the-mean MAE {m['baseline_mae_mm']:.2f})")

    # --- modality ablation, on the honest split ---------------------
    print(f"\n=== modality ablation (held-out section, MAE mm) ===")
    held = uniq_sections[-1]
    te_mask = sections == held
    tr_mask = ~te_mask
    if tr_mask.sum() >= 20 and te_mask.sum() >= 10:
        for label, chans in list(MODALITIES.items()) + [("all", all_channels)]:
            cols = [i for i, n in enumerate(names)
                    if any(n.startswith(c + "__") for c in chans)]
            if not cols:
                continue
            Xs = X[:, cols]
            Xtr, Xte = standardize(Xs[tr_mask], Xs[te_mask])
            w = ridge_fit(Xtr, y[tr_mask], args.alpha)
            m = evaluate(y[te_mask], ridge_predict(Xte, w))
            print(f"  {label:18s} {len(cols):4d} feats   "
                  f"MAE {m['mae_mm']:6.2f}   R2 {m['r2']:+.3f}")
        print(f"  {'predict-the-mean':18s} {'':4s}        "
              f"MAE {np.abs(y[te_mask] - y[tr_mask].mean()).mean():6.2f}")

    # --- positive control -------------------------------------------
    # Can these same features recover forward speed? Gait cadence encodes it
    # directly, so a strong result here means the feature pipeline works and
    # the roughness failure is a data-quantity problem. A weak result means
    # the features themselves are broken, and collecting more data would
    # just produce more of the same nothing.
    print(f"\n=== positive control: predict forward speed ===")
    spd = np.asarray(speeds, dtype=float)
    ok = np.isfinite(spd)
    if ok.sum() > 40 and float(np.nanstd(spd)) > 1e-3:
        held = uniq_sections[-1]
        te_mask = (sections == held) & ok
        tr_mask = (~(sections == held)) & ok
        if tr_mask.sum() >= 20 and te_mask.sum() >= 10:
            Xtr, Xte = standardize(X[tr_mask], X[te_mask])
            w = ridge_fit(Xtr, spd[tr_mask], args.alpha)
            m = evaluate(spd[te_mask], ridge_predict(Xte, w))
            print(f"  held-out section, speed std {spd[ok].std():.3f} m/s")
            print(f"    MAE {m['mae_mm']:.4f} m/s   R2 {m['r2']:+.3f}   "
                  f"(predict-the-mean MAE {m['baseline_mae_mm']:.4f})")
            if m["r2"] > 0.3:
                print("  -> features DO carry information. The roughness "
                      "result is a data-quantity problem, not a broken "
                      "pipeline.")
            else:
                print("  -> features fail even on an easy target. Fix feature "
                      "extraction before collecting more data.")
    else:
        print("  skipped: not enough speed variation in this corpus")

    # Size the corpus honestly, from the data rather than from a stale
    # sentence. Overlapping windows are not independent samples, and the
    # ratio of independent samples to features is what decides whether any
    # of the numbers above can mean anything.
    overlap_factor = max(args.window / max(args.stride, 1e-9), 1.0)
    independent = X.shape[0] / overlap_factor
    ratio = independent / max(X.shape[1], 1)
    print("\n=== how much to believe the numbers above ===")
    print(f"  {X.shape[0]} windows at {100 * (1 - args.stride / args.window):.0f}% "
          f"overlap  ->  ~{independent:.0f} independent samples")
    print(f"  against {X.shape[1]} features  ->  {ratio:.2f} samples per feature")
    print(f"  across {len(uniq_sections)} independent surfaces")
    if ratio < 1.0:
        need = int(np.ceil(X.shape[1] * 5 * overlap_factor / max(X.shape[0], 1)))
        print(f"  UNDERDETERMINED. Roughly {need}x more walking would reach a")
        print("  usable 5 samples per feature -- or use far fewer features.")
        print("  Held-out-surface scores here are dominated by that, not by")
        print("  whether unevenness is detectable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
