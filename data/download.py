"""
data/download.py
Downloads Galaxy Zoo 2 catalog and fetches SDSS image cutouts.
Run this once before training: python -m data.download
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent
IMAGE_DIR  = DATA_DIR / "images"
LABEL_FILE = DATA_DIR / "labels.csv"

IMAGE_DIR.mkdir(exist_ok=True)

# ── Galaxy Zoo 2 vote-fraction thresholds ─────────────────────────────────────
# GZ2 labels each galaxy with crowd-sourced vote fractions.
# We map them to 3 coarse classes:
#   0 = Smooth   (elliptical / lenticular)
#   1 = Disk     (spiral with features)
#   2 = Irregular / Merger / Artifact
#
# Column names from the GZ2 catalog:
#   t01_smooth_or_features_a01_smooth_frac        → P(smooth)
#   t01_smooth_or_features_a02_features_or_disk_frac → P(disk)
#   t01_smooth_or_features_a03_star_or_artifact_frac → P(artifact/irreg)

SMOOTH_COL   = "t01_smooth_or_features_a01_smooth_weighted_fraction"
DISK_COL     = "t01_smooth_or_features_a02_features_or_disk_weighted_fraction"
ARTIFACT_COL = "t01_smooth_or_features_a03_star_or_artifact_weighted_fraction"

# Only keep galaxies where ≥60 % of voters agree on one class
CONFIDENCE_THRESHOLD = 0.60

# SDSS cutout service
SDSS_CUTOUT = (
    "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"
    "?ra={ra}&dec={dec}&scale=0.2&width=128&height=128"
)


def download_gz2_catalog(catalog_path: Path) -> pd.DataFrame:
    """
    Download the GZ2 morphology catalog from Kaggle / the GZ data release.
    If you already have it locally, point catalog_path at your CSV.

    The official GZ2 catalog is available at:
      https://data.galaxyzoo.org/  (gz2_hart16.csv or gz2_filename_mapping.csv)

    For the Kaggle Galaxy Zoo challenge version, download from:
      https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data
    """
    if catalog_path.exists():
        print(f"[catalog] Found existing catalog at {catalog_path}")
        return pd.read_csv(catalog_path)

    raise FileNotFoundError(
        f"Catalog not found at {catalog_path}.\n"
        "Please download gz2_hart16.csv from https://data.galaxyzoo.org/ "
        "and place it in the data/ folder."
    )


def assign_label(row: pd.Series) -> int | None:
    """Return 0/1/2 if one class exceeds the confidence threshold, else None."""
    fracs = {
        0: row[SMOOTH_COL],
        1: row[DISK_COL],
        2: row[ARTIFACT_COL],
    }
    best_class, best_frac = max(fracs.items(), key=lambda x: x[1])
    if best_frac >= CONFIDENCE_THRESHOLD:
        return best_class
    return None


def fetch_sdss_image(ra: float, dec: float, objid: str) -> bool:
    """Download a 128×128 SDSS JPEG for a galaxy. Returns True on success."""
    out_path = IMAGE_DIR / f"{objid}.jpg"
    if out_path.exists():
        return True  # already downloaded

    url = SDSS_CUTOUT.format(ra=ra, dec=dec)
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and len(r.content) > 1000:
            out_path.write_bytes(r.content)
            return True
    except requests.RequestException:
        pass
    return False


def prepare_dataset(
    catalog_path: Path,
    max_per_class: int = 5000,
    sleep_between: float = 0.05,
) -> None:
    """
    Build a balanced dataset of galaxy images.

    Args:
        catalog_path:   Path to the GZ2 CSV catalog.
        max_per_class:  Cap per morphology class (keeps dataset balanced).
        sleep_between:  Seconds to wait between SDSS requests (be polite).
    """
    print("[prepare] Loading catalog …")
    df = download_gz2_catalog(catalog_path)

    # Assign coarse labels
    df["label"] = df.apply(assign_label, axis=1)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print(f"[prepare] Confident labels: {len(df):,}")
    print(df["label"].value_counts().rename({0: "Smooth", 1: "Disk", 2: "Irregular"}))

    # Balance classes
    balanced_groups = []
    for label_class, group in df.groupby("label"):
        sampled = group.sample(min(len(group), max_per_class), random_state=42)
        balanced_groups.append(sampled)
    balanced = pd.concat(balanced_groups, ignore_index=True)
    print(f"[prepare] After balancing: {len(balanced):,} galaxies")

    # Download images
    records = []
    for _, row in tqdm(balanced.iterrows(), total=len(balanced), desc="Fetching images"):
        objid = str(int(row["dr7objid"]))
        ok = fetch_sdss_image(ra=row["ra"], dec=row["dec"], objid=objid)
        if ok:
            records.append({"objid": objid, "label": row["label"]})
        time.sleep(sleep_between)

    labels_df = pd.DataFrame(records)
    labels_df.to_csv(LABEL_FILE, index=False)
    print(f"[prepare] Saved {len(labels_df):,} image labels → {LABEL_FILE}")


if __name__ == "__main__":
    catalog_path = DATA_DIR / "gz2_hart16.csv"
    prepare_dataset(catalog_path, max_per_class=5000)
