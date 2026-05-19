"""
tune.py
───────
Hyperparameter search using training data only.

    python tune.py

Loads the training set, engineers features, then runs GridSearchCV per city
with TimeSeriesSplit (train on past, validate on future — no leakage).
Results are printed as a ranked table and saved to output/tuning_results.csv.
"""

import os
import pandas as pd
from src.data_loader import load_data
from src.cleaning    import clean
from src.features    import build_features, get_feature_cols
from src.tuning      import run_search, PARAM_GRID

OUTPUT_DIR = "output"
CITIES     = [(0, "San Juan"), (1, "Iquitos")]


def print_table(city_name, results):
    print(f"\n{'─' * 60}")
    print(f"  {city_name} — top 10 parameter combinations (MAE ↑ = worse)")
    print(f"{'─' * 60}")
    display = results.head(10).copy()
    display.index = range(1, len(display) + 1)
    display.columns = ["max_features", "min_samples_leaf", "max_depth", "MAE", "std"]
    display["MAE"] = display["MAE"].round(3)
    display["std"] = display["std"].round(3)
    print(display.to_string())
    print()


def tune():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()

    print("Cleaning...")
    df = clean(df)

    print("Building features...")
    df = build_features(df)
    feat_cols = get_feature_cols(df)
    print(f"  Using {len(feat_cols)} features\n")

    print(f"Parameter grid: {PARAM_GRID}\n")

    all_results = []

    for city_code, city_name in CITIES:
        print(f"{'=' * 60}")
        print(f"  Searching: {city_name}")
        print(f"{'=' * 60}")

        city_train = df[(df["city"] == city_code) & (df["type"] == "train")]
        X = city_train[feat_cols]
        y = city_train["total_cases"]
        print(f"  Training rows: {len(X)}\n")

        results = run_search(X, y)
        results.insert(0, "city", city_name)
        all_results.append(results)

        print_table(city_name, results.drop(columns="city"))

    combined = pd.concat(all_results, ignore_index=True)
    out_path = f"{OUTPUT_DIR}/tuning_results.csv"
    combined.to_csv(out_path, index=False)
    print(f"Full results saved: {out_path}")


if __name__ == "__main__":
    tune()
