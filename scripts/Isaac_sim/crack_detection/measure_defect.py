"""
measure_defect.py

Run inside Isaac Sim after opening the USD scene.

Purpose
-------
Measure a sidewalk defect from the actual PhysX collision surface:

1. Get the candidate prim's world-space bounding box.
2. Cast a dense top-down ray grid through that region.
3. Fit the normal sidewalk plane using samples near the grid border.
4. Compute signed surface displacement from that plane.
5. Segment the defect using a depth/height threshold.
6. Measure:
   - length: major horizontal extent of the segmented defect
   - width: minor horizontal extent
   - max_depth: deepest point below the reference plane
   - max_height: highest point above the reference plane
   - vertical_range: max_height + max_depth
   - area: approximate projected defect area

Requirements
------------
- The sidewalk/defect must have PhysX collision enabled.
- The scene should be Z-up.
- Defects represented only by textures/decals cannot have geometric depth.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import carb
import numpy as np
import omni.physx
import omni.usd
from pxr import Gf, Usd, UsdGeom


@dataclass
class MeasurementConfig:
    grid_resolution: int = 151
    bbox_padding_m: float = 0.25
    ray_start_height_m: float = 1.5
    ray_distance_m: float = 4.0

    # Points within this fraction of an edge are used to fit the intact sidewalk.
    border_fraction: float = 0.18

    # A sample is considered defective when its absolute deviation from the
    # reference plane reaches this threshold.
    defect_threshold_m: float = 0.008

    # Removes tiny disconnected noise regions.
    min_component_cells: int = 12

    # Reject extreme ray samples before plane fitting.
    reference_outlier_sigma: float = 3.0

    # The candidate bbox may include a large slab. Set this to crop around the
    # strongest connected defect region after segmentation.
    keep_largest_component: bool = True


@dataclass
class DefectMeasurement:
    prim_path: str
    defect_class: str
    status: str
    message: str

    length_m: Optional[float] = None
    width_m: Optional[float] = None
    max_depth_m: Optional[float] = None
    mean_depth_m: Optional[float] = None
    max_height_m: Optional[float] = None
    mean_height_m: Optional[float] = None
    vertical_range_m: Optional[float] = None
    projected_area_m2: Optional[float] = None

    center_world_m: Optional[List[float]] = None
    major_axis_world_xy: Optional[List[float]] = None
    minor_axis_world_xy: Optional[List[float]] = None

    valid_ray_count: int = 0
    defect_cell_count: int = 0
    meters_per_unit: float = 1.0


def _get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is currently open.")
    return stage


def _meters_per_unit(stage) -> float:
    value = UsdGeom.GetStageMetersPerUnit(stage)
    if not value or value <= 0:
        carb.log_warn("Stage metersPerUnit is invalid; assuming 1 stage unit = 1 meter.")
        return 1.0
    return float(value)


def _world_bbox(stage, prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Prim does not exist: {prim_path}")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    minimum = np.array(aligned.GetMin(), dtype=np.float64)
    maximum = np.array(aligned.GetMax(), dtype=np.float64)

    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError(f"Could not calculate a finite world bbox for {prim_path}")
    return minimum, maximum


def _raycast_closest(origin: Sequence[float],
                     direction: Sequence[float],
                     distance: float) -> Optional[np.ndarray]:
    query = omni.physx.get_physx_scene_query_interface()
    hit = query.raycast_closest(
        carb.Float3(float(origin[0]), float(origin[1]), float(origin[2])),
        carb.Float3(float(direction[0]), float(direction[1]), float(direction[2])),
        float(distance),
    )
    if not hit or not hit.get("hit", False):
        return None

    position = hit.get("position")
    if position is not None:
        return np.array([position[0], position[1], position[2]], dtype=np.float64)

    # Compatibility fallback for bindings that expose only distance.
    hit_distance = float(hit["distance"])
    return np.asarray(origin, dtype=np.float64) + hit_distance * np.asarray(direction, dtype=np.float64)


def _sample_surface(stage,
                    prim_path: str,
                    cfg: MeasurementConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mpu = _meters_per_unit(stage)
    bbox_min, bbox_max = _world_bbox(stage, prim_path)

    padding_u = cfg.bbox_padding_m / mpu
    x_min, y_min = bbox_min[0] - padding_u, bbox_min[1] - padding_u
    x_max, y_max = bbox_max[0] + padding_u, bbox_max[1] + padding_u

    n = int(cfg.grid_resolution)
    if n < 15:
        raise ValueError("grid_resolution should be at least 15.")

    xs = np.linspace(x_min, x_max, n)
    ys = np.linspace(y_min, y_max, n)
    z_start = bbox_max[2] + cfg.ray_start_height_m / mpu
    ray_distance = cfg.ray_distance_m / mpu

    z_grid = np.full((n, n), np.nan, dtype=np.float64)
    hit_grid = np.zeros((n, n, 3), dtype=np.float64)

    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            hit = _raycast_closest((x, y, z_start), (0.0, 0.0, -1.0), ray_distance)
            if hit is not None:
                z_grid[row, col] = hit[2]
                hit_grid[row, col] = hit

    return xs, ys, z_grid, hit_grid


def _border_mask(shape: Tuple[int, int], fraction: float) -> np.ndarray:
    rows, cols = shape
    border_rows = max(2, int(round(rows * fraction)))
    border_cols = max(2, int(round(cols * fraction)))

    mask = np.zeros(shape, dtype=bool)
    mask[:border_rows, :] = True
    mask[-border_rows:, :] = True
    mask[:, :border_cols] = True
    mask[:, -border_cols:] = True
    return mask


def _fit_plane_robust(x: np.ndarray,
                      y: np.ndarray,
                      z: np.ndarray,
                      sigma: float) -> np.ndarray:
    """Fit z = ax + by + c with iterative residual rejection."""
    points = np.column_stack([x, y, np.ones_like(x)])
    keep = np.isfinite(z)

    if np.count_nonzero(keep) < 12:
        raise ValueError("Too few valid border samples to fit the sidewalk plane.")

    coeff = np.zeros(3)
    for _ in range(4):
        coeff, *_ = np.linalg.lstsq(points[keep], z[keep], rcond=None)
        residual = z - points @ coeff
        center = np.nanmedian(residual[keep])
        mad = np.nanmedian(np.abs(residual[keep] - center))
        robust_sigma = max(1.4826 * mad, 1e-12)
        new_keep = keep & (np.abs(residual - center) <= sigma * robust_sigma)
        if np.array_equal(new_keep, keep) or np.count_nonzero(new_keep) < 12:
            break
        keep = new_keep

    return coeff


def _connected_components(mask: np.ndarray) -> List[np.ndarray]:
    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[np.ndarray] = []

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            cells = []

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))

            component = np.zeros_like(mask, dtype=bool)
            rr, cc = zip(*cells)
            component[np.asarray(rr), np.asarray(cc)] = True
            components.append(component)

    return components


def _largest_valid_component(mask: np.ndarray, min_cells: int) -> np.ndarray:
    components = [c for c in _connected_components(mask) if int(c.sum()) >= min_cells]
    if not components:
        return np.zeros_like(mask, dtype=bool)
    return max(components, key=lambda c: int(c.sum()))


def _pca_extents_xy(points_xy_m: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    if len(points_xy_m) < 3:
        raise ValueError("Too few defect points for PCA.")

    center = points_xy_m.mean(axis=0)
    centered = points_xy_m - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]

    major = axes[:, 0]
    minor = axes[:, 1]
    major_projection = centered @ major
    minor_projection = centered @ minor

    length = float(np.ptp(major_projection))
    width = float(np.ptp(minor_projection))
    return length, width, major, minor


def infer_class(prim_path: str) -> str:
    name = prim_path.lower()
    if "collapsed" in name:
        return "collapsed"
    if "uneven" in name:
        return "uneven"
    if "crack" in name:
        return "crack"
    return "unknown"


def measure_defect(prim_path: str,
                   defect_class: Optional[str] = None,
                   cfg: Optional[MeasurementConfig] = None) -> DefectMeasurement:
    cfg = cfg or MeasurementConfig()
    defect_class = defect_class or infer_class(prim_path)
    stage = _get_stage()
    mpu = _meters_per_unit(stage)

    result = DefectMeasurement(
        prim_path=prim_path,
        defect_class=defect_class,
        status="error",
        message="Measurement did not complete.",
        meters_per_unit=mpu,
    )

    try:
        xs, ys, z_grid, _ = _sample_surface(stage, prim_path, cfg)
        valid = np.isfinite(z_grid)
        result.valid_ray_count = int(valid.sum())

        if result.valid_ray_count < max(100, cfg.grid_resolution):
            raise ValueError(
                "Too few ray hits. Ensure the sidewalk and defect have PhysX colliders "
                "and that simulation/scene queries are active."
            )

        xx, yy = np.meshgrid(xs, ys)
        border = _border_mask(z_grid.shape, cfg.border_fraction) & valid
        coeff = _fit_plane_robust(
            xx[border], yy[border], z_grid[border], cfg.reference_outlier_sigma
        )
        plane_z = coeff[0] * xx + coeff[1] * yy + coeff[2]

        # Positive value = surface below the fitted sidewalk plane.
        depth_m = (plane_z - z_grid) * mpu
        height_m = (z_grid - plane_z) * mpu

        defect_mask = valid & (np.abs(depth_m) >= cfg.defect_threshold_m)

        # Avoid selecting noisy samples in the outer reference ring.
        defect_mask &= ~_border_mask(z_grid.shape, max(0.04, cfg.border_fraction * 0.45))

        if cfg.keep_largest_component:
            defect_mask = _largest_valid_component(defect_mask, cfg.min_component_cells)

        result.defect_cell_count = int(defect_mask.sum())
        if result.defect_cell_count < cfg.min_component_cells:
            raise ValueError(
                "No sufficiently large geometric defect was detected. Reduce "
                "defect_threshold_m, enlarge bbox_padding_m, or verify that the defect "
                "has real geometry rather than only a texture/decal."
            )

        points_xy_m = np.column_stack([xx[defect_mask], yy[defect_mask]]) * mpu
        length_m, width_m, major_axis, minor_axis = _pca_extents_xy(points_xy_m)

        defect_depth = depth_m[defect_mask]
        defect_height = height_m[defect_mask]
        positive_depth = defect_depth[defect_depth > 0]
        positive_height = defect_height[defect_height > 0]

        dx_m = float(abs(xs[1] - xs[0]) * mpu)
        dy_m = float(abs(ys[1] - ys[0]) * mpu)
        center_xy_m = points_xy_m.mean(axis=0)
        center_z_m = float(np.nanmedian(z_grid[defect_mask]) * mpu)

        result.status = "ok"
        result.message = "Measured from raycast surface relative to fitted border plane."
        result.length_m = length_m
        result.width_m = width_m
        result.max_depth_m = float(positive_depth.max()) if positive_depth.size else 0.0
        result.mean_depth_m = float(positive_depth.mean()) if positive_depth.size else 0.0
        result.max_height_m = float(positive_height.max()) if positive_height.size else 0.0
        result.mean_height_m = float(positive_height.mean()) if positive_height.size else 0.0
        result.vertical_range_m = result.max_depth_m + result.max_height_m
        result.projected_area_m2 = float(result.defect_cell_count * dx_m * dy_m)
        result.center_world_m = [float(center_xy_m[0]), float(center_xy_m[1]), center_z_m]
        result.major_axis_world_xy = [float(major_axis[0]), float(major_axis[1])]
        result.minor_axis_world_xy = [float(minor_axis[0]), float(minor_axis[1])]

    except Exception as exc:
        result.message = str(exc)

    return result


def save_results_json(results: Iterable[DefectMeasurement], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump([asdict(item) for item in results], handle, indent=2)


def save_results_csv(results: Iterable[DefectMeasurement], output_path: str) -> None:
    rows = [asdict(item) for item in results]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not rows:
        return

    for row in rows:
        for key in ("center_world_m", "major_axis_world_xy", "minor_axis_world_xy"):
            row[key] = json.dumps(row[key]) if row[key] is not None else ""

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_measurement(result: DefectMeasurement) -> None:
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    # Change this path before running directly.
    PRIM_PATH = "/World/assembly_City/crack_1"

    CONFIG = MeasurementConfig(
        grid_resolution=151,
        bbox_padding_m=0.25,
        defect_threshold_m=0.008,
        border_fraction=0.18,
    )

    measurement = measure_defect(PRIM_PATH, cfg=CONFIG)
    print_measurement(measurement)
