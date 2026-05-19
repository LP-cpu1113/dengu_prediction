"""
features.py
───────────
Feature selection and engineering.

To add a new feature:
  1. Add a function that takes df and returns df with the new column.
  2. Add the column name to FEATURE_COLS.
  3. Call the function inside build_features().
"""

# ── Lag windows ───────────────────────────────────────────────────────────────

LAGS = list(range(1, 9))   # [1, 2, 3, 4, 5, 6, 7, 8] weeks — applied to all signals

# ── Which columns go into the model ──────────────────────────────────────────

FEATURE_COLS = [
    "city",
    "year",
    "weekofyear",
    # NDVI — four raw directions + mean + lagged mean
    "ndvi_ne",
    "ndvi_nw",
    "ndvi_se",
    "ndvi_sw",
    "ndvi_mean",
    *[f"ndvi_mean_lag{l}"              for l in LAGS],
    # Precipitation — raw + 4w rolling + lags + lags of rolling
    "station_precip_mm",
    "station_precip_roll4",
    *[f"station_precip_lag{l}"         for l in LAGS],
    *[f"station_precip_roll4_lag{l}"   for l in LAGS],
    # Temperature — avg + diurnal range + min + max + dew point, each with roll4/lags
    "reanalysis_avg_temp_k",
    "reanalysis_tdtr_k",
    "avg_temp_roll4",
    *[f"avg_temp_lag{l}"               for l in LAGS],
    *[f"avg_temp_roll4_lag{l}"         for l in LAGS],
    "reanalysis_min_air_temp_k",
    "min_temp_roll4",
    *[f"min_temp_lag{l}"               for l in LAGS],
    *[f"min_temp_roll4_lag{l}"         for l in LAGS],
    "reanalysis_max_air_temp_k",
    "max_temp_roll4",
    *[f"max_temp_lag{l}"               for l in LAGS],
    *[f"max_temp_roll4_lag{l}"         for l in LAGS],
    "reanalysis_dew_point_temp_k",
    "dew_point_roll4",
    *[f"dew_point_lag{l}"              for l in LAGS],
    *[f"dew_point_roll4_lag{l}"        for l in LAGS],
    # Humidity — raw + 4w rolling + lags + lags of rolling
    "reanalysis_specific_humidity_g_per_kg",
    "specific_humidity_roll4",
    *[f"specific_humidity_lag{l}"      for l in LAGS],
    *[f"specific_humidity_roll4_lag{l}" for l in LAGS],
]

# Expected total: 3 + 13 + 18 + 19 + 18 + 18 + 18 + 18 = 125


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling(df, col, window):
    """Per-city rolling mean. min_periods=1 so first rows use partial windows."""
    return (
        df.groupby("city")[col]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )

def _lag(df, col, lag):
    """Per-city lag with backfill to handle NaNs at the start of each city's series."""
    shifted = df.groupby("city")[col].shift(lag)
    return shifted.groupby(df["city"]).transform(lambda s: s.bfill())


# ── Engineering functions ─────────────────────────────────────────────────────

def add_ndvi_features(df):
    df = df.copy()
    ndvi_cols = [c for c in df.columns if c.startswith("ndvi_")]
    df["ndvi_mean"] = df[ndvi_cols].mean(axis=1)
    for l in LAGS:
        df[f"ndvi_mean_lag{l}"] = _lag(df, "ndvi_mean", l)
    return df


def add_precipitation_features(df):
    df = df.copy()
    col = "station_precip_mm"
    df["station_precip_roll4"] = _rolling(df, col, 4)
    for l in LAGS:
        df[f"station_precip_lag{l}"]       = _lag(df, col, l)
        df[f"station_precip_roll4_lag{l}"] = _lag(df, "station_precip_roll4", l)
    return df


def add_temp_features(df):
    df = df.copy()
    for col, prefix in [
        ("reanalysis_avg_temp_k",       "avg_temp"),
        ("reanalysis_min_air_temp_k",   "min_temp"),
        ("reanalysis_max_air_temp_k",   "max_temp"),
        ("reanalysis_dew_point_temp_k", "dew_point"),
    ]:
        df[f"{prefix}_roll4"] = _rolling(df, col, 4)
        for l in LAGS:
            df[f"{prefix}_lag{l}"]       = _lag(df, col, l)
            df[f"{prefix}_roll4_lag{l}"] = _lag(df, f"{prefix}_roll4", l)
    return df


def add_humidity_features(df):
    df = df.copy()
    col = "reanalysis_specific_humidity_g_per_kg"
    df["specific_humidity_roll4"] = _rolling(df, col, 4)
    for l in LAGS:
        df[f"specific_humidity_lag{l}"]       = _lag(df, col, l)
        df[f"specific_humidity_roll4_lag{l}"] = _lag(df, "specific_humidity_roll4", l)
    return df


def build_features(df):
    df = add_ndvi_features(df)
    df = add_precipitation_features(df)
    df = add_temp_features(df)
    df = add_humidity_features(df)
    return df


def get_feature_cols(df):
    """Return only the feature columns that actually exist in df."""
    return [c for c in FEATURE_COLS if c in df.columns]
