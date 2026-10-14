# Sidewalk Defect Ground-Truth Scripts

These scripts measure geometric sidewalk defects in an open Isaac Sim scene.

## Files

- `measure_defect.py`: core raycast, plane-fitting, segmentation, and measurement logic.
- `batch_measure_defects.py`: measures a selected list and saves JSON/CSV.
- `validate_geometry.py`: fast diagnostic pass before the full-resolution batch.

## Installation

Copy the three Python files to:

```text
C:\Users\Elyas\sidewalk_dataset\scripts
```

No additional pip packages are required beyond Isaac Sim's Python environment and NumPy.

## Required scene setup

1. Open the USD scene in Isaac Sim.
2. Ensure the sidewalk and defects have PhysX colliders.
3. Start or initialize physics so scene queries can detect those colliders.
4. Keep the scene Z-up.
5. Run scripts through Isaac Sim's Script Editor, not ordinary system Python.

PhysX raycasts only detect collider geometry. A crack represented solely by a
texture, decal, normal map, or shader will report no geometric depth.

## Recommended workflow

### 1. Validate candidates

Open and run `validate_geometry.py`.

Candidates marked with `✓` produced measurable surface variation. Candidates
marked with `✗` may lack collision, may be texture-only, or may need parameter
adjustment.

### 2. Inspect preview images

Keep only defects that are visually clear and represent the intended class.
Remove unsuitable paths from `SELECTED_DEFECTS` in `batch_measure_defects.py`.

### 3. Run full measurement

Run `batch_measure_defects.py`.

Results are written to:

```text
C:\Users\Elyas\sidewalk_dataset\ground_truth\defect_ground_truth.json
C:\Users\Elyas\sidewalk_dataset\ground_truth\defect_ground_truth.csv
```

## Output interpretation

For cracks:

- `length_m`: major PCA extent of the segmented geometric crack
- `width_m`: minor PCA extent
- `max_depth_m`: deepest point below the fitted intact sidewalk plane
- `mean_depth_m`: average depth of points below the plane

For uneven/collapsed regions:

- `max_height_m`: highest point above the reference plane
- `max_depth_m`: deepest point below the reference plane
- `vertical_range_m`: `max_height_m + max_depth_m`

All output measurements are converted using the USD stage's `metersPerUnit`.

## Tuning

### Defect not detected

Lower:

```python
defect_threshold_m=0.004
```

Increase:

```python
grid_resolution=201
bbox_padding_m=0.30
```

### Surrounding sidewalk is being classified as defective

Increase `defect_threshold_m`, increase `border_fraction`, or reduce
`bbox_padding_m`.

### Wrong connected region selected

The algorithm keeps the largest connected thresholded region. If the candidate
bbox contains several cracks, set:

```python
keep_largest_component=False
```

This will measure their combined PCA extent, or split the candidate into separate
prims/regions for cleaner instance-level ground truth.

## Important limitation

PCA length is the end-to-end major-axis extent. For a strongly curved or
branching crack, it is not the full centerline/arc length. A later version can
skeletonize the segmented grid and compute geodesic crack length if that metric
is required.
