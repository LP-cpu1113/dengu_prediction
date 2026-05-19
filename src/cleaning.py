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
    Fill missing feature values using the training set mean.

    Why train mean only? Using the full dataset mean would leak information
    from the test set into training — a subtle but real form of data leakage.
    """
    df = df.copy()

    target     = "total_cases"
    skip       = {target, "type", "city", "year", "weekofyear"}
    feat_cols  = [c for c in df.columns if c not in skip]
    cols_null  = [c for c in feat_cols if df[c].isnull().any()]

    train_mean = df.loc[df["type"] == "train", cols_null].mean()
    df[cols_null] = df[cols_null].fillna(train_mean)

    return df


# ── Compose all cleaning steps ────────────────────────────────────────────────

def clean(df):
    df = drop_unused_columns(df)
    df = encode_city(df)
    df = impute_missing(df)
    return df
