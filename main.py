"""
main.py
───────
Runs the full pipeline top to bottom.

    python main.py

Outputs (written to the output/ folder):
    output/predictions.csv   — submission ready to upload to DrivenData
    output/report.html       — EDA plots for preliminary investigation
"""

import os
import pandas as pd
from src.data_loader import load_data
from src.cleaning    import clean
from src.features    import build_features, get_feature_cols
from src.models      import train_model, predict
from src.report      import generate_report

OUTPUT_DIR = "output"


def run():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} rows  ({df['type'].value_counts().to_dict()})")

    # ── 2. EDA report (on raw data, before any cleaning) ─────────────────────
    print("Generating EDA report...")
    generate_report(df, output_path=f"{OUTPUT_DIR}/report.html")

    # ── 3. Clean ──────────────────────────────────────────────────────────────
    print("Cleaning...")
    df = clean(df)

    # ── 4. Feature engineering ────────────────────────────────────────────────
    print("Building features...")
    df = build_features(df)

    feat_cols = get_feature_cols(df)
    print(f"  Using {len(feat_cols)} features: {feat_cols}")

    # ── 5. Train and predict — one model per city ─────────────────────────────
    all_preds = []

    for city_code, city_name in [(0, "San Juan"), (1, "Iquitos")]:
        print(f"\n  {city_name}...")

        city_df   = df[df["city"] == city_code]
        train_df  = city_df[city_df["type"] == "train"]
        test_df   = city_df[city_df["type"] == "test"]

        X_train   = train_df[feat_cols]
        y_train   = train_df["total_cases"]
        X_test    = test_df[feat_cols]

        model     = train_model(X_train, y_train)
        preds     = predict(model, X_test)

        result    = test_df[["city", "year", "weekofyear"]].copy()
        result["city"]        = result["city"].map({0: "sj", 1: "iq"})
        result["total_cases"] = preds
        all_preds.append(result)

        print(f"    predictions: min={preds.min()}, max={preds.max()}, mean={preds.mean():.1f}")

    # ── 6. Save submission ────────────────────────────────────────────────────
    submission = pd.concat(all_preds, ignore_index=True)
    submission.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
    print(f"\nSubmission saved: {OUTPUT_DIR}/predictions.csv  ({len(submission)} rows)")


if __name__ == "__main__":
    run()
