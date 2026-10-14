"""
Contact sheet of rendered frames: RGB with crack-mask outlines, for eyeballing
that masks trace the visible cracks.

    python scripts/crack_data/contact_sheet.py --render D:/sidewalk_crack_synth/render --n 12
"""

import argparse
import os
import random

import cv2
import numpy as np


def overlay(rgb, mask):
    """Crack outlines red; IGNORE (65535, too thin to resolve) yellow."""
    out = rgb.copy()
    k = np.ones((3, 3), np.uint8)
    for sel, col in (((mask > 0) & (mask < 65535), (0, 0, 255)),
                     (mask == 65535, (0, 230, 255))):
        edge = cv2.morphologyEx(sel.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0
        out[edge] = col
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default="D:/sidewalk_crack_synth/render")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", action="store_true", help="RGB only, no outlines")
    a = ap.parse_args()
    files = sorted(os.listdir(os.path.join(a.render, "rgb")))
    random.Random(a.seed).shuffle(files)
    tiles = []
    for f in files[: a.n]:
        rgb = cv2.imread(os.path.join(a.render, "rgb", f))
        mask = cv2.imread(os.path.join(a.render, "mask", f), cv2.IMREAD_UNCHANGED)
        img = rgb if a.raw else overlay(rgb, mask)
        img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f[:-4], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        tiles.append(img)
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
    out = a.out or os.path.join(a.render, "contact_sheet.jpg")
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(out)


if __name__ == "__main__":
    main()
