"""
Download the raw IMDB dataset and create 500-item splits used by the project.

Run this once before run_baseline_search.py if the data is missing.

Usage:
    python search_approach_comparison/prepare_data.py
"""

import os
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data.imdb.reduced_imdb import create_dataset_splits

IMDB_URL     = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_DIR     = "/mount-fs/poodle/labeled-data/imdb"
ACLIMDB_ROOT = os.path.join(IMDB_DIR, "aclImdb")
SPLIT_SIZE   = 250   # 250 pos + 250 neg = 500 items per split


def download_and_extract():
    # Download to /tmp to avoid permission issues on the mount, then extract in place
    archive_path = "/tmp/aclImdb_v1.tar.gz"
    os.makedirs(IMDB_DIR, exist_ok=True)

    if not os.path.exists(archive_path):
        print(f"Downloading IMDB dataset from {IMDB_URL} …")
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            pct = downloaded / total_size * 100 if total_size > 0 else 0
            print(f"\r  {pct:.1f}%  ({downloaded // 1_000_000} / {total_size // 1_000_000} MB)", end="", flush=True)
        urllib.request.urlretrieve(IMDB_URL, archive_path, reporthook=_progress)
        print()
    else:
        print(f"Archive already present at {archive_path}, skipping download.")

    if not os.path.isdir(ACLIMDB_ROOT):
        print(f"Extracting {archive_path} → /tmp …")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall("/tmp")
        # Move from /tmp/aclImdb to the target location
        import shutil
        print(f"Moving /tmp/aclImdb → {ACLIMDB_ROOT} …")
        shutil.move("/tmp/aclImdb", ACLIMDB_ROOT)
        print("Extraction complete.")
    else:
        print(f"aclImdb already extracted at {ACLIMDB_ROOT}, skipping extraction.")


def raw_data_present():
    for split in ("train", "test"):
        for sentiment in ("pos", "neg"):
            if not os.path.isdir(os.path.join(ACLIMDB_ROOT, split, sentiment)):
                return False
    return True


if __name__ == "__main__":
    # --- Step 1: ensure raw data exists ------------------------------------
    if not raw_data_present():
        download_and_extract()

    for split in ("train", "test"):
        for sentiment in ("pos", "neg"):
            path = os.path.join(ACLIMDB_ROOT, split, sentiment)
            print(f"  {split}/{sentiment}: {len(os.listdir(path))} files")

    # --- Step 2: create 500-item splits ------------------------------------
    for split in ("train", "test"):
        first_split = os.path.join(ACLIMDB_ROOT, f"{split}-500-0")
        if os.path.isdir(first_split):
            n = len([d for d in os.listdir(ACLIMDB_ROOT) if d.startswith(f"{split}-500-")])
            print(f"\n[SKIP] {split} splits already exist ({n} splits found)")
            continue

        split_path = os.path.join(ACLIMDB_ROOT, split)
        print(f"\nCreating {split} splits (500 items each) from {split_path} …")
        create_dataset_splits(split_path, SPLIT_SIZE)

        n = len([d for d in os.listdir(ACLIMDB_ROOT) if d.startswith(f"{split}-500-")])
        print(f"  Created {n} {split} splits.")

    print("\nDone. You can now run run_baseline_search.py")
