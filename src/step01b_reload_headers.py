"""
STEP 01B - Reload Files with Correct Headers
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step01b_reload_headers.py

Purpose:
Step 01 showed that all three CMS files have disclaimer/title rows before
the real header row. This script skips those rows and reloads each file
with the real column names, then prints everything in full (no truncation)
so we can confirm exact column names before writing Step 02.
"""

from pathlib import Path
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 40)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
RAW_DIR = BASE_DIR / "data" / "raw"
if not RAW_DIR.exists():
    RAW_DIR = BASE_DIR
RVU_DIR = RAW_DIR / "RVU23A"

PPRRVU_PATH = RVU_DIR / "PPRRVU23_JAN.csv"
GPCI_PATH = RVU_DIR / "GPCI2023.csv"
LOCALITY_PATH = RVU_DIR / "23LOCCO.csv"

OUTPUT_DIR = BASE_DIR / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_raw_lines(path, n_lines=10):
    """Print the first n raw lines of a file exactly as they appear,
    so we can see the disclaimer rows and find the true header row."""
    print(f"\nRaw first {n_lines} lines of {path.name}:")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= n_lines:
                    break
                print(f"  [row {i}] {line.strip()[:200]}")
    except Exception as e:
        print(f"  Could not read raw lines: {e}")


# ============================================================
# FILE 1: PPRRVU -- reload with header on row 9 (0-indexed)
# ============================================================
print_section("FILE 1: PPRRVU23_JAN.csv -- RAW LINES")
print_raw_lines(PPRRVU_PATH, n_lines=10)

print_section("FILE 1: PPRRVU23_JAN.csv -- RELOADED WITH CORRECT HEADER")
try:
    pprrvu_df = pd.read_csv(
        PPRRVU_PATH,
        skiprows=9,
        encoding="utf-8",
        low_memory=False,
    )
    pprrvu_df = pprrvu_df.dropna(how="all")
    print(f"Shape: {pprrvu_df.shape[0]:,} rows x {pprrvu_df.shape[1]} columns")
    print("\nColumns:")
    for c in pprrvu_df.columns:
        print(f"  - {c}")
    print("\nFirst 10 rows:")
    print(pprrvu_df.head(10).to_string())
    pprrvu_df.to_csv(OUTPUT_DIR / "pprrvu_reloaded_preview.csv", index=False)
except Exception as e:
    print(f"Error reloading PPRRVU: {e}")
    pprrvu_df = None

# ============================================================
# FILE 2: GPCI -- reload with header on row 2 (0-indexed)
# ============================================================
print_section("FILE 2: GPCI2023.csv -- RELOADED WITH CORRECT HEADER")
try:
    gpci_df = pd.read_csv(GPCI_PATH, skiprows=2, encoding="utf-8")
    gpci_df = gpci_df.dropna(how="all")
    print(f"Shape: {gpci_df.shape[0]:,} rows x {gpci_df.shape[1]} columns")
    print("\nColumns:")
    for c in gpci_df.columns:
        print(f"  - {c}")
    print("\nFirst 10 rows:")
    print(gpci_df.head(10).to_string())
    gpci_df.to_csv(OUTPUT_DIR / "gpci_reloaded_preview.csv", index=False)
except Exception as e:
    print(f"Error reloading GPCI: {e}")
    gpci_df = None

# ============================================================
# FILE 3: LOCALITY CROSSWALK -- reload with header on row 2 (0-indexed)
# ============================================================
print_section("FILE 3: 23LOCCO.csv -- RELOADED WITH CORRECT HEADER")
try:
    locality_df = pd.read_csv(LOCALITY_PATH, skiprows=2, encoding="utf-8")
    locality_df = locality_df.dropna(how="all")
    print(f"Shape: {locality_df.shape[0]:,} rows x {locality_df.shape[1]} columns")
    print("\nColumns:")
    for c in locality_df.columns:
        print(f"  - {c}")
    print("\nFirst 10 rows:")
    print(locality_df.head(10).to_string())
    locality_df.to_csv(OUTPUT_DIR / "locality_reloaded_preview.csv", index=False)
except Exception as e:
    print(f"Error reloading locality crosswalk: {e}")
    locality_df = None

print_section("STEP 01B COMPLETE")
print("""
All three CMS files have now been loaded using their correct header rows.
Review the printed columns and saved preview files before starting Step 02.
""")
