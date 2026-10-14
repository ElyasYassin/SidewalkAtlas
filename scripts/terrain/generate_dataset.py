"""
generate_dataset.py

Batch-generate parametric sidewalk surfaces with exact ground-truth labels.

Runs OFFLINE -- no Isaac needed. Produces, per sample:

  <out>/<sample_id>/height.npy     (nx, ny) heights in mm
  <out>/<sample_id>/labels.json    params + authored features + window labels
  <out>/<sample_id>/mesh.obj       triangulated surface for Isaac import
  <out>/<sample_id>/preview.png    (first N samples only)

plus a dataset-level <out>/manifest.json carrying the class balance.

Severity is sampled on a stratified grid rather than uniformly at random, so
the dataset spans pristine-to-severe evenly instead of clustering wherever the
default parameters happen to sit. Class balance is the point: a model trained
on a corpus that is 90% compliant learns to answer "compliant".

Usage:
    python scripts/terrain/generate_dataset.py --n 200 --out data/terrain
    python scripts/terrain/generate_dataset.py --n 8 --preview 8   # quick look
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sidewalk_terrain import (  # noqa: E402
    ADA_CLASSES,
    concatenate_surfaces,
    seam_steps_mm,
    tag_windows,
    SidewalkSurface,
    compute_labels,
    generate_surface,
    label_summary,
    sample_params,
    write_obj,
)


# =====================================================================
# PREVIEW
# =====================================================================

def write_preview(surface: SidewalkSurface,
                  records: List[Dict],
                  path: str) -> None:
    """Top-down height map, centreline profile, and the discontinuity trace
    against the ADA bands. Cheap insurance against a generator that looks
    fine in the numbers and wrong on the screen."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sidewalk_terrain import ADA_BARRIER_MM, ADA_BEVEL_MM

    h = surface.height_mm
    x, y = surface.x_m, surface.y_m

    # Detrend for display so grade doesn't wash out the colour scale.
    xx = np.repeat(x[:, None], h.shape[1], axis=1)
    yy = np.repeat(y[None, :], h.shape[0], axis=0)
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(h.size)])
    coeff, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)
    detrended = (h.ravel() - A @ coeff).reshape(h.shape)

    lim = float(np.percentile(np.abs(detrended), 99)) or 1.0

    fig, axes = plt.subplots(3, 1, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [2, 1, 1]})

    im = axes[0].imshow(
        detrended.T, origin="lower", aspect="auto", cmap="RdBu_r",
        vmin=-lim, vmax=lim,
        extent=[x[0], x[-1], y[0], y[-1]],
    )
    axes[0].set_title(
        f"Detrended surface (mm)  --  seed {surface.seed}, "
        f"{len(surface.features)} features"
    )
    axes[0].set_ylabel("cross track (m)")
    fig.colorbar(im, ax=axes[0], label="mm", fraction=0.03)

    for f in surface.features:
        if f["type"] == "joint_fault":
            axes[0].axvline(f["x_m"], color="k", lw=0.7, ls="--", alpha=0.55)

    axes[1].plot(x, surface.lane_profile(0.0), lw=0.8, color="#333")
    axes[1].set_title("Centreline profile (mm, raw)")
    axes[1].set_ylabel("mm")
    axes[1].grid(alpha=0.3)

    cx = [r["window_centre_m"] for r in records]
    disc = [r["discontinuity_mm"] for r in records]
    ptp = [r["ptp_mm_max"] for r in records]

    axes[2].axhspan(0, ADA_BEVEL_MM, color="#2e7d32", alpha=0.13,
                    label="compliant")
    axes[2].axhspan(ADA_BEVEL_MM, ADA_BARRIER_MM, color="#f9a825", alpha=0.16,
                    label="bevel required")
    top = max(max(disc, default=0), max(ptp, default=0), ADA_BARRIER_MM) * 1.1
    axes[2].axhspan(ADA_BARRIER_MM, top, color="#c62828", alpha=0.13,
                    label="barrier")
    axes[2].plot(cx, disc, lw=1.6, color="#111", label="discontinuity")
    axes[2].plot(cx, ptp, lw=1.0, color="#1565c0", ls="--",
                 label="detrended ptp")
    axes[2].set_ylim(0, top)
    axes[2].set_xlabel("along track (m)")
    axes[2].set_ylabel("mm")
    axes[2].set_title("Ground-truth labels vs ADA thresholds")
    axes[2].legend(loc="upper right", fontsize=8, ncol=5)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================

def run(args: argparse.Namespace) -> int:
    os.makedirs(args.out, exist_ok=True)
    master = np.random.default_rng(args.seed)

    # Stratified severity: an even sweep, jittered, rather than iid uniform.
    severities = (np.arange(args.n) + master.uniform(0.0, 1.0, args.n)) / args.n
    master.shuffle(severities)

    manifest: Dict = {
        "n_samples": args.n,
        "seed": args.seed,
        "length_m": args.length,
        "width_m": args.width,
        "cell_m": args.cell,
        "mesh_stride": args.mesh_stride,
        "samples": [],
    }

    totals = {c: 0 for c in ADA_CLASSES}
    total_windows = 0

    for i in range(args.n):
        sample_id = f"sw_{i:05d}"
        sample_dir = os.path.join(args.out, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        seed = int(master.integers(0, 2 ** 31 - 1))
        rng = np.random.default_rng(seed)

        if args.segments <= 1:
            params = sample_params(
                rng,
                length_m=args.length,
                width_m=args.width,
                cell_m=args.cell,
                severity=float(severities[i]),
            )
            surface = generate_surface(params, rng, seed=seed)
            segments = [{
                "segment_id": "seg_000",
                "source_seed": seed,
                "x_start_m": 0.0,
                "x_end_m": float(surface.x_m[-1]),
                "params": surface.params.to_dict(),
                "features": surface.features,
            }]
            seams = []
        else:
            # One walkway of many slabs, which is what a real sidewalk is --
            # and what the robot should meet: a single continuous strip walked
            # once, instead of N short strips with a stage rebuild between
            # each. Severity is swept ACROSS the segments so a single walk
            # covers the full range rather than one condition.
            parts = []
            seg_sev = (np.arange(args.segments) + 0.5) / args.segments
            rng.shuffle(seg_sev)
            for k in range(args.segments):
                sub_seed = int(rng.integers(0, 2 ** 31 - 1))
                sub_rng = np.random.default_rng(sub_seed)
                sub_params = sample_params(
                    sub_rng,
                    length_m=args.length,
                    width_m=args.width,
                    cell_m=args.cell,
                    severity=float(seg_sev[k]),
                )
                parts.append(generate_surface(sub_params, sub_rng,
                                              seed=sub_seed))
            surface, segments = concatenate_surfaces(
                parts,
                running_grade_pct=float(rng.uniform(-5.0, 5.0)),
                cross_slope_pct=float(rng.uniform(-2.0, 2.0)),
            )
            seams = seam_steps_mm(surface, segments)

        records = compute_labels(
            surface,
            window_m=args.window,
            stride_m=args.stride,
        )
        records = tag_windows(records, segments)
        summary = label_summary(records)

        np.save(os.path.join(sample_dir, "height.npy"),
                surface.height_mm.astype(np.float32))

        mesh_info = None
        if not args.no_mesh:
            mesh_info = write_obj(
                surface,
                os.path.join(sample_dir, "mesh.obj"),
                stride=args.mesh_stride,
            )

        with open(os.path.join(sample_dir, "labels.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({
                "sample_id": sample_id,
                "seed": seed,
                "severity": float(severities[i]),
                "params": surface.params.to_dict(),
                "features": surface.features,
                "segments": [
                    {k: v for k, v in seg.items() if k != "features"}
                    for seg in segments
                ],
                "seam_discontinuity_mm": seams,
                "label_config": {
                    "window_m": args.window,
                    "stride_m": args.stride,
                },
                "summary": summary,
                "windows": records,
            }, fh, indent=2)

        for c in ADA_CLASSES:
            totals[c] += summary["ada_class_counts"][c]
        total_windows += summary["n_windows"]

        if i < args.preview:
            write_preview(surface, records,
                          os.path.join(sample_dir, "preview.png"))

        manifest["samples"].append({
            "sample_id": sample_id,
            "seed": seed,
            "severity": float(severities[i]),
            "n_windows": summary["n_windows"],
            "ada_class_counts": summary["ada_class_counts"],
            "max_discontinuity_mm": summary["discontinuity_mm"]["max"],
            "max_ptp_mm": summary["ptp_mm"]["max"],
            "n_features": len(surface.features),
            "n_segments": len(segments),
            "worst_seam_mm": (max(seams) if seams else 0.0),
            "mesh": mesh_info,
        })

        if (i + 1) % max(1, args.n // 10) == 0 or i == args.n - 1:
            print(f"  {i + 1}/{args.n} generated")

    manifest["totals"] = {
        "windows": total_windows,
        "ada_class_counts": totals,
        "ada_class_fractions": {
            c: (totals[c] / total_windows if total_windows else 0.0)
            for c in ADA_CLASSES
        },
    }

    with open(os.path.join(args.out, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nWrote {args.n} samples to {args.out}")
    print(f"  windows: {total_windows}")
    for c in ADA_CLASSES:
        frac = totals[c] / total_windows if total_windows else 0.0
        print(f"    {c:16s} {totals[c]:7d}  ({frac * 100:5.1f}%)")
    if args.preview:
        print(f"  previews: first {min(args.preview, args.n)} samples")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=50, help="number of surfaces")
    p.add_argument("--out", default="data/terrain", help="output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--length", type=float, default=12.0, help="metres")
    p.add_argument("--width", type=float, default=1.8, help="metres")
    p.add_argument("--cell", type=float, default=0.01, help="label grid, metres")
    p.add_argument("--window", type=float, default=1.0, help="label window, metres")
    p.add_argument("--stride", type=float, default=0.10, help="label stride, metres")
    p.add_argument("--mesh-stride", type=int, default=2,
                   help="decimate mesh export by this factor (labels unaffected)")
    p.add_argument("--no-mesh", action="store_true", help="skip OBJ export")
    p.add_argument("--segments", type=int, default=1,
                   help="slabs concatenated into ONE walkway per sample; "
                        ">1 produces a single long continuous strip")
    p.add_argument("--preview", type=int, default=6,
                   help="render previews for the first N samples")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
