"""
Batch-generate synthetic cracked-sidewalk scenes (offline, no Isaac).

    python scripts/crack_data/generate_crack_scenes.py --n 250
    python scripts/crack_data/generate_crack_scenes.py --n 4 --start 9000 --out <dir>

Each scene_XXXXX/ gets albedo.jpg, roughness.jpg, normal.png,
height_mesh.npy, labels.npz, scene.json (measurements + centrelines) and
preview.jpg. render_crack_dataset.py renders them in Isaac.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crack_scene import generate_scene, save_scene  # noqa: E402


# =====================================================================
# CONFIG
# =====================================================================

OUT_DIR = "D:/sidewalk_crack_synth/scenes"   # C: is nearly full


def _one(args):
    seed, out = args
    d = os.path.join(out, f"scene_{seed:05d}")
    if os.path.exists(os.path.join(d, "scene.json")):
        return seed, -1
    meta = save_scene(generate_scene(seed), d)
    return seed, len(meta["cracks"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() // 2, 1))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    jobs = [(s, a.out) for s in range(a.start, a.start + a.n)]
    n_cracks = 0
    with Pool(a.workers) as pool:
        for i, (seed, nc) in enumerate(pool.imap_unordered(_one, jobs), 1):
            n_cracks += max(nc, 0)
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} scenes, {n_cracks} cracks, "
                      f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
