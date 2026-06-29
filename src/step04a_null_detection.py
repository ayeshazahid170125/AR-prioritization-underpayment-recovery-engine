"""
STEP 04A - Null Values Detection
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: python step04a_null_detection.py

Purpose:
Detect every type of missing/null value in claims_with_variance.csv (Step 03
output) before any cleaning decisions are made. This mirrors the Project 1
Step 02 approach: detect everything first, decide what to do about it later.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
INPUT_PATH = BASE_DIR / "variance_outputs" / "claims_with_variance.csv"

OUTPUT_DIR = BASE_DIR / "cleaning_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = OUTPUT_DIR / "null_detection_summary.csv"


def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================
print_section("LOAD claims_with_variance.csv")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"File not found: {INPUT_PATH}\nRun step03_join_actual_expected.py first.")

df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"File: {INPUT_PATH}")


# ============================================================
# NULL PATTERNS -- same comprehensive list used in Project 1
# ============================================================
NULL_PATTERNS = [
    None, np.nan, 'nan', 'NaN', 'NAN', 'none', 'None', 'NONE',
    'null', 'NULL', 'NA', 'na', 'N/A', 'n/a', '', ' ', '  ',
    '-', '--', '.', '?', 'unknown', 'Unknown', 'UNKNOWN',
    'missing', 'Missing', 'MISSING', '#N/A', '#NA',
    # "0" is intentionally excluded: a $0 gap or $0 allowed amount can be a
    # real, valid value in this dataset (e.g. truly $0 services), so zero
    # values are flagged separately as review-only signals below.
]


def detect_all_nulls(series, dtype):
    """Detect every type of null/missing pattern in a single column."""
    results = {}
    total = len(series)

    standard_null = series.isnull().sum()
    results['Standard NaN/None'] = standard_null

    if dtype == 'object' or str(dtype) == 'str':
        for pattern in NULL_PATTERNS:
            if pattern is None or (isinstance(pattern, float) and np.isnan(pattern)):
                continue
            count = (series.astype(str).str.strip() == str(pattern).strip()).sum()
            if count > 0:
                results[f'String "{pattern}"'] = count

        whitespace_count = series.dropna().astype(str).str.strip().eq('').sum()
        if whitespace_count > 0:
            results['Whitespace Only'] = whitespace_count

    if dtype in ['int64', 'float64']:
        zero_count = (series == 0).sum()
        if zero_count > 0:
            results['Zero (0) values - review only'] = zero_count

        neg_count = (series < 0).sum()
        if neg_count > 0:
            results['Negative values - review only'] = neg_count

        if dtype == 'float64':
            inf_count = np.isinf(series).sum()
            if inf_count > 0:
                results['Inf / -Inf'] = inf_count

    return results, total


# ============================================================
# RUN DETECTION ACROSS ALL COLUMNS
# ============================================================
print_section("NULL DETECTION -- ALL COLUMNS")
print(f"{'Column':<35} {'Dtype':<10} {'Null Type':<32} {'Count':>10} {'%':>8}")
print("-" * 97)

summary_rows = []
cols_with_no_nulls = []

for col in df.columns:
    dtype = df[col].dtype
    null_results, total = detect_all_nulls(df[col], dtype)

    has_true_null = any(
        v > 0 and "review only" not in k for k, v in null_results.items()
    )
    has_any_flag = any(v > 0 for v in null_results.values())

    if not has_true_null:
        cols_with_no_nulls.append(col)
    if not has_any_flag:
        continue

    first = True
    for null_type, count in null_results.items():
        if count > 0:
            pct = count / total * 100
            col_display = col if first else ''
            dtype_display = str(dtype) if first else ''
            print(f"{col_display:<35} {dtype_display:<10} {null_type:<32} {count:>10,} {pct:>7.2f}%")
            summary_rows.append({
                'Column': col, 'Dtype': str(dtype), 'Null_Type': null_type,
                'Flag_Category': 'Review Only' if 'review only' in null_type else 'Null / Missing',
                'Count': count, 'Percentage': round(pct, 2),
            })
            first = False
    print("-" * 97)


# ============================================================
# SPECIAL CHECK -- COLUMNS WHERE NULL IS EXPECTED/MEANINGFUL
# ============================================================
print_section("CONTEXT CHECK -- IS_UNDERPAID AND DOLLAR FIELDS")
print("""
Payment_Gap, Payment_Gap_Pct, and Total_Dollar_Gap can legitimately be
negative (that is the entire point of this analysis -- negative means
underpaid). Negative values in these specific columns are NOT a data
quality problem and will not be flagged for correction in Step 04c.
""")

for col in ["Payment_Gap", "Payment_Gap_Pct", "Total_Dollar_Gap"]:
    if col in df.columns:
        neg_count = (df[col] < 0).sum()
        neg_pct = neg_count / len(df) * 100
        print(f"  {col:<25} negative (expected): {neg_count:,} ({neg_pct:.2f}%)")


# ============================================================
# COLUMNS WITH ZERO NULLS
# ============================================================
print_section("COLUMNS WITH ZERO TRUE NULLS")
if cols_with_no_nulls:
    for col in cols_with_no_nulls:
        print(f"  OK - {col}")
else:
    print("  Every column has some null/missing signal.")


# ============================================================
# SUMMARY TABLE
# ============================================================
print_section("SUMMARY -- NULL COUNT BY COLUMN (Sorted)")
summary_df = pd.DataFrame(summary_rows)

if not summary_df.empty:
    true_null_df = summary_df[summary_df['Flag_Category'] == 'Null / Missing']
    if not true_null_df.empty:
        col_summary = (
            true_null_df.groupby('Column')['Count'].sum().reset_index()
            .sort_values('Count', ascending=False)
        )
        col_summary['Percentage'] = (col_summary['Count'] / len(df) * 100).round(2)
        print(col_summary.to_string(index=False))
    else:
        print("No true null/missing values found across any column.")


# ============================================================
# OVERALL STATS + SAVE
# ============================================================
print_section("OVERALL NULL STATISTICS")
total_cells = df.shape[0] * df.shape[1]
standard_nulls = df.isnull().sum().sum()
print(f"Total Rows            : {df.shape[0]:,}")
print(f"Total Columns          : {df.shape[1]}")
print(f"Total Cells             : {total_cells:,}")
print(f"Standard NaN/None       : {standard_nulls:,} ({standard_nulls/total_cells*100:.4f}%)")
print(f"Columns WITH nulls      : {df.shape[1] - len(cols_with_no_nulls)}")
print(f"Columns WITHOUT nulls   : {len(cols_with_no_nulls)}")

if not summary_df.empty:
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSummary saved: {SUMMARY_PATH}")

print_section("STEP 04A COMPLETE")
print("""
Next: Step 04b will run statistical and domain-specific outlier detection
on the numeric columns (Payment_Gap, Payment_Gap_Pct, Total_Dollar_Gap),
including visual charts, before any rows are flagged or cleaned.
""")