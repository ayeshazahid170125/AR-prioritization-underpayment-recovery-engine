"""
STEP 01 - Data Load & Exploration (Project 2)
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step_01_eda.py

Purpose:
Explore the three fee-schedule source files before building the
expected-payment calculation in Step 02. We only look at the data here --
nothing is calculated, joined, or cleaned yet.
"""

from pathlib import Path
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION - adjust these paths if your folder names differ
# ============================================================
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
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def explore_file(path, label, skiprows=0, n_preview=8):
    """Load a CSV and print its shape, columns, dtypes, and a few sample rows.
    Tries a couple of encodings since CMS files are sometimes not UTF-8."""
    print_section(f"{label} -- {path.name}")

    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        print("Check the path in the CONFIGURATION section above.")
        return None

    df = None
    last_error = None
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(
                path,
                skiprows=skiprows,
                encoding=encoding,
                low_memory=False,
            )
            df = df.dropna(how="all")
            print(f"Loaded successfully with encoding: {encoding}")
            break
        except Exception as e:
            last_error = e
            continue

    if df is None:
        print(f"Could not load file with any common encoding. Last error: {last_error}")
        return None

    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\nColumn names and dtypes:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:>3}. {col:<35} {str(df[col].dtype)}")

    print(f"\nFirst {n_preview} rows:")
    print(df.head(n_preview).to_string())

    print("\nMissing values per column (top 10 by count):")
    nulls = df.isna().sum().sort_values(ascending=False)
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("  No missing values found.")
    else:
        print(nulls.head(10).to_string())

    return df


# ============================================================
# 1. PPRRVU FILE -- RVU values per HCPCS code
# ============================================================
pprrvu_df = explore_file(PPRRVU_PATH, "FILE 1: PROCEDURE RVUs", skiprows=9)

if pprrvu_df is not None:
    print_section("PPRRVU -- LOOKING FOR KEY COLUMNS")
    candidate_keywords = ["hcpcs", "work", "rvu", "pe", "mp", "malpractice", "status", "modifier"]
    for col in pprrvu_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in candidate_keywords):
            print(f"  Possible match: '{col}'")
            print(f"    Sample values: {pprrvu_df[col].dropna().unique()[:5].tolist()}")


# ============================================================
# 2. GPCI FILE -- Geographic adjustment by locality
# ============================================================
gpci_df = explore_file(
    GPCI_PATH,
    "FILE 2: GEOGRAPHIC PRACTICE COST INDEX (GPCI)",
    skiprows=2,
)

if gpci_df is not None:
    print_section("GPCI -- LOOKING FOR KEY COLUMNS")
    candidate_keywords = ["locality", "carrier", "state", "gpci", "work", "pe", "mp"]
    for col in gpci_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in candidate_keywords):
            print(f"  Possible match: '{col}'")
            print(f"    Sample values: {gpci_df[col].dropna().unique()[:5].tolist()}")


# ============================================================
# 3. LOCALITY CROSSWALK FILE -- bridges locality code to state/county
# ============================================================
locality_df = explore_file(
    LOCALITY_PATH,
    "FILE 3: LOCALITY-TO-COUNTY CROSSWALK",
    skiprows=2,
)

if locality_df is not None:
    print_section("LOCALITY CROSSWALK -- LOOKING FOR KEY COLUMNS")
    candidate_keywords = ["locality", "carrier", "state", "county", "area"]
    for col in locality_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in candidate_keywords):
            print(f"  Possible match: '{col}'")
            print(f"    Sample values: {locality_df[col].dropna().unique()[:5].tolist()}")


# ============================================================
# SAVE COLUMN PROFILES FOR EACH FILE
# ============================================================
print_section("SAVING COLUMN PROFILES")

for df, name in [(pprrvu_df, "pprrvu"), (gpci_df, "gpci"), (locality_df, "locality")]:
    if df is not None:
        profile = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_null_count": [df[c].notna().sum() for c in df.columns],
            "unique_count": [df[c].nunique(dropna=True) for c in df.columns],
            "sample_value": [
                df[c].dropna().iloc[0] if df[c].notna().any() else None
                for c in df.columns
            ],
        })
        out_path = OUTPUT_DIR / f"column_profile_{name}.csv"
        profile.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

# ============================================================
# SUMMARY -- WHAT WE STILL NEED TO CONFIRM BEFORE STEP 02
# ============================================================
print_section("WHAT TO CONFIRM BEFORE STEP 02")
print("""
1. In the PPRRVU file: confirm exact column names for
   - HCPCS code
   - Work RVU
   - Practice Expense (PE) RVU (non-facility and facility versions)
   - Malpractice (MP) RVU
   - Status indicator (some codes are not separately payable)

2. In the GPCI file: confirm exact column names for
   - Locality code (must match the format used elsewhere)
   - Work GPCI, PE GPCI, MP GPCI

3. In the locality crosswalk file: confirm how locality code
   maps to state, since the CMS Provider Payment PUF identifies
   providers by state, not by CMS locality code.

4. The 2023 Conversion Factor dollar value -- this is a single
   fixed number published in the RVU23A-508.pdf documentation,
   not a column in any of these files.

Review the printed column lists and saved column_profile_*.csv files,
then we will write Step 02 using the confirmed column names.
""")

print_section("STEP 01 COMPLETE")
