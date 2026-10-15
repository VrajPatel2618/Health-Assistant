"""
preprocessing.py
================
Data loading and cleaning pipeline for the Cardiovascular Disease dataset.

All cleaning steps operate on a working copy of the raw dataframe so the
original DataFrame is never mutated.  Every removal or modification step is
logged to a ``cleaning_log`` list that callers can inspect or print.

Usage (from notebook or train_model.py):
    from src.preprocessing import load_and_clean

    df_clean, cleaning_log = load_and_clean("data/cardio_train.csv")
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants (domain-informed thresholds)
# ---------------------------------------------------------------------------
AGE_DAYS_TO_YEARS = 365.25          # days → years conversion factor

# Blood pressure plausibility bounds (mmHg)
AP_HI_MIN, AP_HI_MAX = 60, 250     # systolic
AP_LO_MIN, AP_LO_MAX = 40, 180     # diastolic

# Height plausibility bounds (cm)
HEIGHT_MIN, HEIGHT_MAX = 100, 220

# Weight plausibility bounds (kg)
WEIGHT_MIN, WEIGHT_MAX = 30, 200

# BMI plausibility bounds (kg/m²)
BMI_MIN, BMI_MAX = 10, 70


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _log(cleaning_log: list, message: str) -> None:
    """Append a timestamped message to the cleaning log."""
    cleaning_log.append(message)
    print(f"  [CLEAN] {message}")


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def load_and_clean(filepath: str) -> tuple[pd.DataFrame, list[str]]:
    """Load the CSV, inspect it, clean it, and return (df_clean, cleaning_log).

    Parameters
    ----------
    filepath : str
        Path to the raw cardio_train.csv file (semicolon-separated).

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned working copy.  The ``id`` column is dropped.
        Age has been converted to years.  BMI column added.
    cleaning_log : list of str
        Human-readable record of every cleaning decision made.
    """
    cleaning_log: list[str] = []

    # ------------------------------------------------------------------
    # Step 1 – Load
    # ------------------------------------------------------------------
    print("\n=== Step 1: Loading dataset ===")
    df_raw = pd.read_csv(filepath, sep=";")
    _log(cleaning_log, f"Loaded raw dataset: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    _log(cleaning_log, f"Separator: semicolon (';')")
    _log(cleaning_log, f"Columns: {df_raw.columns.tolist()}")

    # ------------------------------------------------------------------
    # Step 2 – Inspect
    # ------------------------------------------------------------------
    print("\n=== Step 2: Inspection ===")
    _log(cleaning_log, f"Data types:\n{df_raw.dtypes.to_string()}")
    _log(cleaning_log, f"Missing values per column:\n{df_raw.isnull().sum().to_string()}")
    _log(cleaning_log, f"Total missing cells: {df_raw.isnull().sum().sum()}")

    # Work on a copy so the raw DataFrame stays unchanged
    df = df_raw.copy()

    # Drop the meaningless id column
    df.drop(columns=["id"], inplace=True)
    _log(cleaning_log, "Dropped column 'id' (surrogate key, not a feature).")

    # ------------------------------------------------------------------
    # Step 3 – Remove duplicates
    # ------------------------------------------------------------------
    print("\n=== Step 3: Duplicate removal ===")
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_dupes = n_before - len(df)
    _log(cleaning_log, f"Duplicate rows removed: {n_dupes} ({n_dupes/n_before*100:.2f}%)")

    # ------------------------------------------------------------------
    # Step 4 – Convert age from days to years
    # ------------------------------------------------------------------
    print("\n=== Step 4: Age conversion ===")
    df["age"] = (df["age"] / AGE_DAYS_TO_YEARS).round(1)
    _log(cleaning_log,
         f"Converted 'age' from days to years (div {AGE_DAYS_TO_YEARS}). "
         f"Range now: {df['age'].min():.1f} – {df['age'].max():.1f} years.")

    # ------------------------------------------------------------------
    # Step 5 – Blood pressure cleaning
    # ------------------------------------------------------------------
    print("\n=== Step 5: Blood pressure cleaning ===")
    n_before = len(df)

    # Remove non-positive values
    bp_nonpos = df[(df["ap_hi"] <= 0) | (df["ap_lo"] <= 0)]
    _log(cleaning_log, f"Non-positive BP readings: {len(bp_nonpos)}")

    # Remove implausible extremes
    bp_extreme = df[
        (df["ap_hi"] < AP_HI_MIN) | (df["ap_hi"] > AP_HI_MAX) |
        (df["ap_lo"] < AP_LO_MIN) | (df["ap_lo"] > AP_LO_MAX)
    ]
    _log(cleaning_log,
         f"Implausible BP (systolic outside {AP_HI_MIN}–{AP_HI_MAX} mmHg "
         f"or diastolic outside {AP_LO_MIN}–{AP_LO_MAX} mmHg): {len(bp_extreme)}")

    # Remove rows where diastolic >= systolic (physiologically impossible)
    bp_inverted = df[df["ap_lo"] >= df["ap_hi"]]
    _log(cleaning_log, f"Inverted BP (diastolic >= systolic): {len(bp_inverted)}")

    # Apply all BP filters together
    valid_bp = (
        (df["ap_hi"] > 0) & (df["ap_lo"] > 0) &
        (df["ap_hi"] >= AP_HI_MIN) & (df["ap_hi"] <= AP_HI_MAX) &
        (df["ap_lo"] >= AP_LO_MIN) & (df["ap_lo"] <= AP_LO_MAX) &
        (df["ap_lo"] < df["ap_hi"])
    )
    df = df[valid_bp].copy()
    n_removed_bp = n_before - len(df)
    _log(cleaning_log, f"Total rows removed for BP issues: {n_removed_bp} "
                        f"({n_removed_bp/n_before*100:.2f}%)")

    # ------------------------------------------------------------------
    # Step 6 – Height / Weight outliers
    # ------------------------------------------------------------------
    print("\n=== Step 6: Height and weight cleaning ===")
    n_before = len(df)

    hw_invalid = df[
        (df["height"] < HEIGHT_MIN) | (df["height"] > HEIGHT_MAX) |
        (df["weight"] < WEIGHT_MIN) | (df["weight"] > WEIGHT_MAX)
    ]
    _log(cleaning_log,
         f"Implausible height (outside {HEIGHT_MIN}–{HEIGHT_MAX} cm) "
         f"or weight (outside {WEIGHT_MIN}–{WEIGHT_MAX} kg): {len(hw_invalid)}")

    valid_hw = (
        (df["height"] >= HEIGHT_MIN) & (df["height"] <= HEIGHT_MAX) &
        (df["weight"] >= WEIGHT_MIN) & (df["weight"] <= WEIGHT_MAX)
    )
    df = df[valid_hw].copy()
    n_removed_hw = n_before - len(df)
    _log(cleaning_log, f"Total rows removed for height/weight issues: {n_removed_hw} "
                        f"({n_removed_hw/n_before*100:.2f}%)")

    # ------------------------------------------------------------------
    # Step 7 – Calculate BMI
    # ------------------------------------------------------------------
    print("\n=== Step 7: BMI calculation ===")
    df["bmi"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(2)
    _log(cleaning_log,
         f"BMI computed: weight(kg) / height(m)². "
         f"Range: {df['bmi'].min():.1f} – {df['bmi'].max():.1f}")

    # ------------------------------------------------------------------
    # Step 8 – BMI plausibility filter (catches remaining height/weight
    #          combination oddities, e.g. weight=30 kg, height=100 cm)
    # ------------------------------------------------------------------
    print("\n=== Step 8: BMI plausibility filter ===")
    n_before = len(df)
    df = df[(df["bmi"] >= BMI_MIN) & (df["bmi"] <= BMI_MAX)].copy()
    n_removed_bmi = n_before - len(df)
    _log(cleaning_log,
         f"Rows removed for implausible BMI (outside {BMI_MIN}–{BMI_MAX}): "
         f"{n_removed_bmi}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_removed = df_raw.shape[0] - len(df)
    _log(cleaning_log,
         f"\nCLEANING SUMMARY: started with {df_raw.shape[0]:,} rows, "
         f"removed {total_removed:,} rows total "
         f"({total_removed/df_raw.shape[0]*100:.2f}%), "
         f"retained {len(df):,} rows.")
    _log(cleaning_log, f"Final cleaned dataset shape: {df.shape}")

    return df, cleaning_log


def get_feature_columns(df: pd.DataFrame, target: str = "cardio") -> tuple[list, str]:
    """Return (feature_columns, target_column) from a cleaned DataFrame."""
    feature_cols = [c for c in df.columns if c != target]
    return feature_cols, target


if __name__ == "__main__":
    # Quick smoke-test when run directly
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "cardio_train.csv")
    df_clean, log = load_and_clean(path)
    print(f"\nCleaned shape: {df_clean.shape}")
    print(df_clean.describe().to_string())
