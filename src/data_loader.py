"""
data_loader.py
──────────────
Loads the raw CSVs and returns a single combined DataFrame.

The returned df has a 'type' column: 'train' or 'test'.
Test rows have NaN for total_cases.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_data(data_dir=DATA_DIR):
    data_dir = Path(data_dir)

    features_train = pd.read_csv(data_dir / "dengue_features_train.csv")
    labels_train   = pd.read_csv(data_dir / "dengue_labels_train.csv")
    features_test  = pd.read_csv(data_dir / "dengue_features_test.csv")

    train = features_train.merge(labels_train, on=["city", "year", "weekofyear"])
    train["type"] = "train"

    features_test["type"] = "test"

    df = pd.concat([train, features_test], ignore_index=True)
    return df
