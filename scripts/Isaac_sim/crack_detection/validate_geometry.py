"""
validate_geometry.py

Quick diagnostic script. It checks whether each candidate has enough top-down
ray hits and whether its surface contains measurable vertical variation.

This is useful for detecting defects that are only textures/decals or that lack
PhysX collision.
"""

import importlib.util
import os
import sys

SCRIPT_DIR = r"C:\Users\Elyas\sidewalk_dataset\scripts"
MEASURE_SCRIPT = os.path.join(SCRIPT_DIR, "measure_defect.py")

spec = importlib.util.spec_from_file_location("measure_defect", MEASURE_SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules["measure_defect"] = m
spec.loader.exec_module(m)

CANDIDATES = [
    "/World/assembly_City/collapsed_1",
    "/World/assembly_City/collapsed_2",
    "/World/assembly_City/collapsed_3",
    "/World/assembly_City/collapsed_4",
    "/World/assembly_City/crack_1",
    "/World/assembly_City/crack_2",
    "/World/assembly_City/crack_3",
    "/World/assembly_City/crack_4",
    "/World/assembly_City/uneven_1",
    "/World/assembly_City/uneven_2",
    "/World/assembly_City/uneven_3",
    "/World/assembly_City/uneven_4",
    "/World/assembly_City/uneven_5",
    "/World/assembly_City/uneven_6",
    "/World/assembly_City/uneven_7",
    "/World/procedural_cracks",
]

cfg = m.MeasurementConfig(
    grid_resolution=51,
    bbox_padding_m=0.20,
    defect_threshold_m=0.003,
    min_component_cells=4,
)

for path in CANDIDATES:
    result = m.measure_defect(path, cfg=cfg)
    if result.status == "ok":
        print(
            f"✓ {path}\n"
            f"  depth={result.max_depth_m:.4f}m, "
            f"height={result.max_height_m:.4f}m, "
            f"LxW={result.length_m:.3f}x{result.width_m:.3f}m, "
            f"rays={result.valid_ray_count}"
        )
    else:
        print(f"✗ {path}\n  {result.message}")
