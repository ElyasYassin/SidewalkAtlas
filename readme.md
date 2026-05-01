# SidewalkAtlas

AI-assisted sidewalk condition assessment using simulation, perception, geometry estimation, and condition scoring.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Scripts

### Sidewalk Segmentation

Runs SegFormer (Cityscapes-pretrained) on images in `images/` and outputs segmentation overlays and sidewalk masks.

```powershell
python scripts/segment_sidewalk.py
```

Results are saved to `images/segmentation_output/`.

### Defect Placement (Isaac Sim)

Places Meshy-generated defect assets onto sidewalk surfaces in the AECO city scene.

Before running, set `CITY_USD_PATH` in the script if it differs from the default.

```powershell
C:\isaac-sim\python.bat scripts\place_defects.py
```

Output is saved as `World_CityDemopack_with_defects.usd` alongside the original city file.

> Isaac Sim dependencies (`isaacsim`, `omni`, `pxr`) are bundled with Isaac Sim and are not in `requirements.txt`. Use `C:\isaac-sim\python.bat` to run Isaac Sim scripts, not the `.venv` Python.

## Project Structure

```
SidewalkAtlas/
├── images/                  # Input images and segmentation output
├── scripts/
│   ├── segment_sidewalk.py  # SegFormer segmentation
│   └── place_defects.py     # Isaac Sim defect placement
├── sidewalks/               # GLB defect assets (crack, collapsed, pothole, uneven)
│   └── usd_cache/           # Auto-generated USD conversions
├── requirements.txt
└── readme.md
```
