"""
batch_measure_defects.py

Run in Isaac Sim's Script Editor after measure_defect.py is available.
Edit SELECTED_DEFECTS to contain only candidates whose previews are useful.
"""

import importlib.util
import os
import sys

# Update these paths for your machine.
SCRIPT_DIR = r"C:\Users\Elyas\sidewalk_dataset\scripts"
OUTPUT_DIR = r"C:\Users\Elyas\sidewalk_dataset\ground_truth"

MEASURE_SCRIPT = os.path.join(SCRIPT_DIR, "measure_defect.py")

spec = importlib.util.spec_from_file_location("measure_defect", MEASURE_SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules["measure_defect"] = module
spec.loader.exec_module(module)

MeasurementConfig = module.MeasurementConfig
measure_defect = module.measure_defect
save_results_csv = module.save_results_csv
save_results_json = module.save_results_json
print_measurement = module.print_measurement


SELECTED_DEFECTS = [
    {"prim_path": "/World/assembly_City/collapsed_1", "class": "collapsed"},
    {"prim_path": "/World/assembly_City/collapsed_2", "class": "collapsed"},
    {"prim_path": "/World/assembly_City/collapsed_3", "class": "collapsed"},
    {"prim_path": "/World/assembly_City/collapsed_4", "class": "collapsed"},

    {"prim_path": "/World/assembly_City/crack_1", "class": "crack"},
    {"prim_path": "/World/assembly_City/crack_2", "class": "crack"},
    {"prim_path": "/World/assembly_City/crack_3", "class": "crack"},
    {"prim_path": "/World/assembly_City/crack_4", "class": "crack"},

    {"prim_path": "/World/assembly_City/uneven_1", "class": "uneven"},
    {"prim_path": "/World/assembly_City/uneven_2", "class": "uneven"},
    {"prim_path": "/World/assembly_City/uneven_3", "class": "uneven"},
    {"prim_path": "/World/assembly_City/uneven_4", "class": "uneven"},
    {"prim_path": "/World/assembly_City/uneven_5", "class": "uneven"},
    {"prim_path": "/World/assembly_City/uneven_6", "class": "uneven"},
    {"prim_path": "/World/assembly_City/uneven_7", "class": "uneven"},

    # This prim is extremely dense. Enable only if its preview is useful.
    # {"prim_path": "/World/procedural_cracks", "class": "crack"},
]


CONFIG_BY_CLASS = {
    "crack": MeasurementConfig(
        grid_resolution=201,
        bbox_padding_m=0.20,
        defect_threshold_m=0.004,
        border_fraction=0.18,
        min_component_cells=10,
    ),
    "uneven": MeasurementConfig(
        grid_resolution=151,
        bbox_padding_m=0.30,
        defect_threshold_m=0.010,
        border_fraction=0.20,
        min_component_cells=15,
    ),
    "collapsed": MeasurementConfig(
        grid_resolution=151,
        bbox_padding_m=0.35,
        defect_threshold_m=0.015,
        border_fraction=0.22,
        min_component_cells=15,
    ),
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    for index, item in enumerate(SELECTED_DEFECTS, start=1):
        prim_path = item["prim_path"]
        defect_class = item["class"]
        print(f"\n[{index}/{len(SELECTED_DEFECTS)}] Measuring {prim_path}")

        result = measure_defect(
            prim_path=prim_path,
            defect_class=defect_class,
            cfg=CONFIG_BY_CLASS[defect_class],
        )
        print_measurement(result)
        results.append(result)

        # Save after every item so partial progress is not lost.
        save_results_json(results, os.path.join(OUTPUT_DIR, "defect_ground_truth.json"))
        save_results_csv(results, os.path.join(OUTPUT_DIR, "defect_ground_truth.csv"))

    successful = sum(result.status == "ok" for result in results)
    print(
        f"\nDone: {successful}/{len(results)} measured successfully.\n"
        f"JSON: {os.path.join(OUTPUT_DIR, 'defect_ground_truth.json')}\n"
        f"CSV:  {os.path.join(OUTPUT_DIR, 'defect_ground_truth.csv')}"
    )


main()
