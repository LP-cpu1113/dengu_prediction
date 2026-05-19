"""
features.py
───────────
Feature selection and engineering.

To add a new feature:
  1. Add a function that takes df and returns df with the new column.
  2. Add the column name to FEATURE_COLS.
  3. Call the function inside build_features().
"""

import numpy as np

# ── Which columns go into the model ──────────────────────────────────────────
# Edit this list to change what the model sees.

FEATURE_COLS = [
    "city",
    "year",
    "weekofyear",
    "ndvi_ne",
    "ndvi_nw",
    "ndvi_se",
    "ndvi_sw",
    "precipitation_amt_mm",
    "reanalysis_air_temp_k",
    "reanalysis_avg_temp_k",
    "reanalysis_dew_point_temp_k",
    "reanalysis_max_air_temp_k",
    "reanalysis_min_air_temp_k",
    "reanalysis_precip_amt_kg_per_m2",
    "reanalysis_relative_humidity_percent",
    "reanalysis_sat_precip_amt_mm",
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_tdtr_k",
    "station_avg_temp_c",
    "station_diur_temp_rng_c",
    "station_max_temp_c",
    "station_min_temp_c",
    "station_precip_mm",
]


# ── Optional engineering functions ────────────────────────────────────────────
# Uncomment or add your own below, then add the new column to FEATURE_COLS.

# def add_ndvi_mean(df):
#     df = df.copy()
#     ndvi_cols = [c for c in df.columns if c.startswith("ndvi_")]
#     df["ndvi_mean"] = df[ndvi_cols].mean(axis=1)
#     return df


def build_features(df):
    """Apply all feature engineering steps and return the modified df."""
    # df = add_ndvi_mean(df)   # example: uncomment to activate
    return df


def get_feature_cols(df):
    """Return only the feature columns that actually exist in df."""
    return [c for c in FEATURE_COLS if c in df.columns]
