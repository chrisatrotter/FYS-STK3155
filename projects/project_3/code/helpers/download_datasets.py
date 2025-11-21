# code/helpers/download_datasets.py
"""
Download and extract the two required datasets for Project 3.
Only runs when the final CSV files are missing.
Called automatically from project3.py.
Can be run manually: python -m helpers.download_datasets
"""

from pathlib import Path
import urllib.request
import zipfile
import shutil
from tqdm import tqdm
import sys

# ──────────────────────────────────────────────────────────────
# Import your beautiful divider
# ──────────────────────────────────────────────────────────────
from utils import breakpoint

# ──────────────────────────────────────────────────────────────
# Paths – correct project root (data/ is at the root, not inside code/)
# ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # project root
DATA_DIR = ROOT_DIR / "code/data"
DATA_DIR.mkdir(exist_ok=True)

POWER_CSV = DATA_DIR / "household_power_consumption.csv"
TRADE_CSV = DATA_DIR / "Dyadic_COW_4.0.csv"


def _download_with_progress(url: str, dest_path: Path):
    """Download with nice tqdm progress bar."""
    if dest_path.exists():
        return

    print(f"Downloading {dest_path.name} ...")
    class ProgressBar(tqdm):
        def update_to(self, block_num=1, block_size=1, total_size=None):
            if total_size is not None:
                self.total = total_size
            self.update(block_num * block_size - self.n)

    with ProgressBar(
        unit='B', unit_scale=True, unit_divisor=1024,
        miniters=1, desc=dest_path.name, leave=False
    ) as t:
        urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)


def download_power_dataset():
    if POWER_CSV.exists():
        print("Power dataset already exists: household_power_consumption.csv")
        return

    zip_path = DATA_DIR / "power_consumption.zip"
    _download_with_progress(
        "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
        zip_path
    )

    print("Extracting power consumption dataset...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)  # extracts to data/ + possible subfolder

    # Find the extracted .txt file (it's usually directly in data/)
    txt_files = list(DATA_DIR.glob("household_power_consumption*.txt"))
    if not txt_files:
        print("Could not find household_power_consumption.txt after extraction!")
        sys.exit(1)

    txt_files[0].rename(POWER_CSV)

    # Clean up any leftover folders (sometimes UCI zip has a subdir)
    for folder in DATA_DIR.iterdir():
        if folder.is_dir() and folder.name.startswith(("individual+household", "__MACOSX")):
            shutil.rmtree(folder, ignore_errors=True)

    zip_path.unlink(missing_ok=True)
    print("Power dataset ready → household_power_consumption.csv\n")


def download_trade_dataset():
    if TRADE_CSV.exists():
        print("Trade dataset already exists: Dyadic_COW_4.0.csv")
        return

    zip_path = DATA_DIR / "cow_trade.zip"
    _download_with_progress(
        "https://correlatesofwar.org/wp-content/uploads/COW_Trade_4.0.zip",
        zip_path
    )

    print("Extracting COW trade dataset...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)

    # Find the dyadic CSV
    candidates = list(DATA_DIR.rglob("*[Dd]yadic*.csv"))
    if not candidates:
        print("Could not locate Dyadic COW CSV after extraction!")
        sys.exit(1)

    candidates[0].rename(TRADE_CSV)

    # Clean up extracted folders
    for folder in DATA_DIR.glob("Dyadic*"):
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
    for folder in DATA_DIR.glob("COW_Trade*"):
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)

    zip_path.unlink(missing_ok=True)
    print("Trade dataset ready → Dyadic_COW_4.0.csv\n")


def ensure_datasets():
    """Called from project3.py – ensures datasets exist before running."""
    breakpoint()
    print("Checking required datasets in data/ ...")
    breakpoint()

    missing = []
    if not POWER_CSV.exists():
        missing.append("household_power_consumption.csv")
    if not TRADE_CSV.exists():
        missing.append("Dyadic_COW_4.0.csv")

    if not missing:
        print("All required datasets are already present!")
        print("→ household_power_consumption.csv")
        print("→ Dyadic_COW_4.0.csv")
        breakpoint()
        return

    print(f"Missing {len(missing)} dataset(s): {', '.join(missing)}")
    print("Downloading and extracting automatically...\n")
    breakpoint()

    download_power_dataset()
    download_trade_dataset()

    breakpoint()
    print("ALL DATASETS ARE NOW READY!")
    print("You can safely run the full project.")
    breakpoint()


if __name__ == "__main__":
    ensure_datasets()