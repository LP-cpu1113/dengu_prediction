"""
cleaning.py
───────────
All data cleaning steps live here as standalone functions.

To add a new cleaning step:
  1. Write a function that takes df and returns df.
  2. Call it in main.py inside clean().
"""

import pandas as pd


def drop_unused_columns(df):
    return df.drop(columns=["week_start_date"], errors="ignore")


def encode_city(df):
    """Map city strings to integers so the model can use it as a feature."""
    df = df.copy()
    df["city"] = df["city"].map({"sj": 0, "iq": 1})
    return df


def impute_missing(df):
    """
    Fill missing feature values using per-city linear interpolation.

    Interpolation respects the temporal structure of the data — a missing week's
    climate value is estimated as a smooth progression between the surrounding
    known values, which is physically sensible for temperature, humidity, etc.
    Any remaining NaNs at the edges of each city's series (where there is no
    surrounding value to interpolate from) are filled with ffill then bfill.
    """
    df = df.copy()

    skip      = {"total_cases", "type", "city", "year", "weekofyear"}
    feat_cols = [c for c in df.columns if c not in skip]
    cols_null = [c for c in feat_cols if df[c].isnull().any()]

    df[cols_null] = (
        df.groupby("city")[cols_null]
        .transform(lambda s: s.interpolate(method="linear").ffill().bfill())
    )

    return df


# ── Compose all cleaning steps ────────────────────────────────────────────────

def clean(df):
    df = drop_unused_columns(df)
    df = encode_city(df)
    df = impute_missing(df)
    return df
