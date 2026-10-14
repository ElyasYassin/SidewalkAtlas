"""
Download the CC0 ambientCG texture sets the crack scenes use (2K JPG), unzip
to TEX_DIR/<AssetId>/. Re-runnable: skips assets already present.

    python scripts/crack_data/fetch_textures.py
"""

import io
import os
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

TEX_DIR = "D:/textures/acg"
RES = "2K-JPG"

ASSETS = {
    # walking-surface concrete (checked by eye: no cracks of their own)
    "concrete": ["Concrete048", "Concrete047A", "Concrete034", "Concrete030",
                 "Concrete046", "Concrete025", "Concrete016", "Concrete014",
                 "Concrete017", "Concrete015", "Concrete006", "Concrete002",
                 "Concrete003", "Concrete037", "Concrete040", "Concrete018",
                 "Concrete044A", "Concrete041A", "Concrete043A", "Concrete026",
                 "Concrete038"],
    "asphalt": ["Asphalt033", "Asphalt031", "Asphalt025C", "Asphalt023S",
                "Asphalt026C", "Asphalt032", "Road012A", "Road008A",
                "Road013A", "Road009C", "Road011A", "Road014A", "Road015A"],
    "pavers": ["PavingStones151", "PavingStones150", "PavingStones138",
               "PavingStones128", "PavingStones142", "PavingStones146",
               "PavingStones147", "PavingStones070", "PavingStones149",
               "PavingStones136", "PavingStones127", "PavingStones131",
               "PavingStones141", "PavingStones122", "PavingStones111",
               "PavingStones081", "PavingStones139", "PavingStones130"],
    "bricks": ["Bricks105", "Bricks104", "Bricks097", "Bricks102",
               "Bricks075A", "Bricks101", "Bricks085", "Bricks094",
               "Bricks092", "Bricks066"],
    "metal": ["DiamondPlate009", "DiamondPlate008A", "DiamondPlate006A",
              "DiamondPlate001", "DiamondPlate003", "Grate001", "Grate002"],
    "tactile": ["TactilePaving001", "TactilePaving002", "TactilePaving003",
                "TactilePaving004", "TactilePaving005", "TactilePaving006",
                "TactilePaving007"],
    "ground": ["Grass001", "Grass004", "Grass005", "Ground037", "Ground003",
               "Gravel023", "Gravel040", "Gravel022", "Ground039",
               "Ground062S"],
}


def fetch(asset):
    out = os.path.join(TEX_DIR, asset)
    if os.path.isdir(out) and any(f.endswith(".jpg") for f in os.listdir(out)):
        return asset, "skip"
    url = f"https://ambientcg.com/get?file={asset}_{RES}.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "SidewalkAtlas/1.0"})
    try:
        data = urllib.request.urlopen(req, timeout=180).read()
        os.makedirs(out, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for n in z.namelist():
                if n.lower().endswith((".jpg", ".png")):
                    with open(os.path.join(out, os.path.basename(n)), "wb") as fh:
                        fh.write(z.read(n))
        return asset, f"{len(data) / 1e6:.0f} MB"
    except Exception as e:  # keep going; a missing set just isn't sampled
        return asset, f"FAILED {e}"


def main():
    os.makedirs(TEX_DIR, exist_ok=True)
    todo = [a for v in ASSETS.values() for a in v]
    with ThreadPoolExecutor(6) as ex:
        for asset, status in ex.map(fetch, todo):
            print(f"{asset:20s} {status}", flush=True)


if __name__ == "__main__":
    main()
