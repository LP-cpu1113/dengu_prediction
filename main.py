"""
main.py
───────
DengAI pipeline — single entry point.

    python main.py              # run pipeline → predictions.csv + report.html
    python main.py --tune       # hyperparameter search → tuning_results.csv
    python main.py --present    # generate presentation → presentation.html
"""

import argparse
import os
import textwrap

import pandas as pd

from src.data_loader import load_data
from src.cleaning    import clean
from src.features    import build_features, get_feature_cols
from src.models      import train_model, predict
from src.report      import generate_report

OUTPUT_DIR = "output"
CITIES     = [(0, "San Juan"), (1, "Iquitos")]


# ── Shared setup ──────────────────────────────────────────────────────────────

def _prepare(verbose=True):
    """Load → clean → build features. Returns (df, feat_cols)."""
    print("Loading data...")
    df = load_data()
    if verbose:
        print(f"  {len(df)} rows  ({df['type'].value_counts().to_dict()})")

    print("Cleaning...")
    df = clean(df)

    print("Building features...")
    df = build_features(df)
    feat_cols = get_feature_cols(df)
    if verbose:
        print(f"  Using {len(feat_cols)} features")
    return df, feat_cols


# ── Mode 1: pipeline ──────────────────────────────────────────────────────────

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df_raw = load_data()
    print(f"  {len(df_raw)} rows  ({df_raw['type'].value_counts().to_dict()})")

    print("Generating EDA report...")
    generate_report(df_raw, output_path=f"{OUTPUT_DIR}/report.html")

    print("Cleaning...")
    df = clean(df_raw)

    print("Building features...")
    df = build_features(df)
    feat_cols = get_feature_cols(df)
    print(f"  Using {len(feat_cols)} features: {feat_cols}")

    all_preds = []
    for city_code, city_name in CITIES:
        print(f"\n  {city_name}...")
        city_df  = df[df["city"] == city_code]
        train_df = city_df[city_df["type"] == "train"]
        test_df  = city_df[city_df["type"] == "test"]

        X_train = train_df[feat_cols]
        y_train = train_df["total_cases"]
        X_test  = test_df[feat_cols]

        model = train_model(X_train, y_train)
        preds = predict(model, X_test)

        result = test_df[["city", "year", "weekofyear"]].copy()
        result["city"]        = result["city"].map({0: "sj", 1: "iq"})
        result["total_cases"] = preds
        all_preds.append(result)

        print(f"    predictions: min={preds.min()}, max={preds.max()}, mean={preds.mean():.1f}")

    submission = pd.concat(all_preds, ignore_index=True)
    submission.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
    print(f"\nSubmission saved: {OUTPUT_DIR}/predictions.csv  ({len(submission)} rows)")


# ── Mode 2: hyperparameter tuning ─────────────────────────────────────────────

def run_tuning():
    from src.tuning import run_search, PARAM_GRID

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, feat_cols = _prepare()

    print(f"\nParameter grid: {PARAM_GRID}\n")

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

        print(f"\n{'─' * 60}")
        print(f"  {city_name} — top 10 combinations (MAE ↑ = worse)")
        print(f"{'─' * 60}")
        display = results.drop(columns="city").head(10).copy()
        display.index = range(1, len(display) + 1)
        display.columns = ["max_features", "min_samples_leaf", "max_depth", "MAE", "std"]
        display["MAE"] = display["MAE"].round(3)
        display["std"] = display["std"].round(3)
        print(display.to_string())
        print()

    combined = pd.concat(all_results, ignore_index=True)
    out_path  = f"{OUTPUT_DIR}/tuning_results.csv"
    combined.to_csv(out_path, index=False)
    print(f"Full results saved: {out_path}")


# ── Mode 3: presentation ──────────────────────────────────────────────────────

def run_presentation():
    from src.presentation import (
        _fig_to_base64,
        _param_cards_html,
        plot_dataset_overview,
        plot_missing_values_raw,
        plot_cleaning_interpolation,
        plot_feature_engineering,
        plot_tuning_results,
        plot_feature_importance,
        plot_predictions,
    )

    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "presentation.html")
    TUNING_CSV  = os.path.join(OUTPUT_DIR, "tuning_results.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── load data at each pipeline stage ──────────────────────────────────────
    print("[1/7] Loading raw data...")
    df_raw = load_data()
    print(f"      {len(df_raw)} rows  ({df_raw['type'].value_counts().to_dict()})")

    print("[2/7] Cleaning...")
    df_clean = clean(df_raw)

    print("[3/7] Building features...")
    df_feat   = build_features(df_clean)
    feat_cols = get_feature_cols(df_feat)
    print(f"      {len(feat_cols)} feature columns")

    def _img(fig, alt=""):
        return (f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" '
                f'alt="{alt}" style="max-width:100%;border:1px solid #e2e8f0;'
                f'border-radius:8px;margin-top:0.5rem;display:block;" />')

    def _section(sid, title, body):
        return textwrap.dedent(f"""\
            <section id="{sid}">
              <h2>{title}</h2>
              {body}
            </section>""")

    # ── plots ─────────────────────────────────────────────────────────────────
    print("[4/7] Generating plots...")

    print("      Section 1 — Dataset overview")
    s1 = (
        _img(plot_dataset_overview(df_raw), "Dataset overview") +
        "<p>1,456 training weeks across two cities. "
        "Target: weekly dengue case count. Metric: MAE.</p>"
    )

    print("      Section 2 — Missing values")
    fig2 = plot_missing_values_raw(df_raw)
    if fig2:
        counts  = (df_raw
                   .drop(columns=["type","total_cases","city","year",
                                  "weekofyear","week_start_date"], errors="ignore")
                   .isnull().sum())
        counts  = counts[counts > 0].sort_values(ascending=False)
        top_str = ", ".join(f"<code>{c}</code>" for c in counts.index[:4])
        s2 = (_img(fig2, "Missing values") +
              f"<p>Most affected: {top_str}. All filled before modelling.</p>")
    else:
        s2 = "<p>No missing values in raw data.</p>"

    print("      Section 3 — Cleaning / interpolation")
    fig3 = plot_cleaning_interpolation(df_raw, df_clean)
    s3 = (
        (_img(fig3, "Interpolation") if fig3 else "") +
        f"<p>{int(df_raw['ndvi_ne'].isnull().sum())} missing values in "
        "<code>ndvi_ne</code> filled with per-city linear interpolation. "
        "The shaded regions show where NaN gaps existed. "
        "The bottom panel shows the smooth, gap-free series used by the model.</p>"
    )

    print("      Section 4 — Feature engineering")
    s4 = (
        _img(plot_feature_engineering(df_feat), "Feature engineering") +
        "<p><strong>Lag features</strong> shift a climate signal backward in time, "
        "capturing the delayed biological effect between an environmental event "
        "and the resulting spike in reported cases (lags 1–8 weeks added per signal). "
        "<strong>Rolling means</strong> capture accumulated environmental conditions — "
        "a sustained 4-week humid period is more predictive than a single wet day.</p>"
    )

    print("      Section 5 — Hyperparameter tuning")
    tuning = plot_tuning_results(TUNING_CSV)
    if tuning[0] is not None:
        fig_lines, fig_heat, top5_html = tuning
        s5 = (
            _param_cards_html() +
            "<h3 style='margin:1.5rem 0 0.5rem;color:#1e293b;'>Effect of min_samples_leaf</h3>" +
            _img(fig_lines, "MAE vs min_samples_leaf") +
            "<h3 style='margin:1.5rem 0 0.5rem;color:#1e293b;'>Interaction: min_samples_leaf × max_features</h3>" +
            _img(fig_heat, "Tuning heatmap") +
            "<p><code>min_samples_leaf</code> is the dominant regulariser. "
            "The default of 1 allows trees to memorise individual training weeks; "
            "a value of 20 gives the best validation MAE for both cities.</p>" +
            '<div class="tuning-tables">' + top5_html + "</div>"
        )
    else:
        _, _, msg = tuning
        s5 = _param_cards_html() + msg

    print("      Section 6 — Feature importance (training RF models — ~30 s)")
    s6 = (
        _img(plot_feature_importance(df_feat, feat_cols, top_n=20),
             "Feature importance") +
        "<p>Mean decrease in node impurity across all trees, normalised to sum to 1. "
        "Both cities lean on temperature and humidity lags; "
        "Iquitos relies more on recent precipitation.</p>"
    )

    print("      Section 7 — Predictions (re-training production models)")
    s7 = (
        _img(plot_predictions(df_feat, feat_cols), "Predictions") +
        "<p>Solid line: training history. Dashed: test-period forecast. "
        "Dotted vertical line marks the train/test boundary. "
        "Both cities track seasonal patterns; Iquitos shows a flatter forecast "
        "consistent with its lower baseline case counts.</p>"
    )

    # ── assemble HTML ─────────────────────────────────────────────────────────
    print("[5/7] Assembling HTML...")

    toc_items = [
        ("sec1", "1. The Dataset"),
        ("sec2", "2. Missing Values"),
        ("sec3", "3. Cleaning: Interpolation"),
        ("sec4", "4. Feature Engineering"),
        ("sec5", "5. Hyperparameter Tuning"),
        ("sec6", "6. Feature Importance"),
        ("sec7", "7. Predictions"),
    ]
    toc_html = "<ol>\n" + "\n".join(
        f'  <li><a href="#{sid}">{label}</a></li>'
        for sid, label in toc_items
    ) + "\n</ol>"

    sections = "\n".join([
        _section("sec1", "1. The Dataset",               s1),
        _section("sec2", "2. Missing Values",             s2),
        _section("sec3", "3. Cleaning: Interpolation",    s3),
        _section("sec4", "4. Feature Engineering",        s4),
        _section("sec5", "5. Hyperparameter Tuning",      s5),
        _section("sec6", "6. Feature Importance",         s6),
        _section("sec7", "7. Predictions",                s7),
    ])

    html = textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>DengAI — Pipeline Walkthrough</title>
          <style>
            *, *::before, *::after {{ box-sizing: border-box; }}
            body {{
              font-family: system-ui, -apple-system, sans-serif;
              width: 92%;
              margin: 0 auto;
              padding: 0 0 3rem;
              color: #1e293b;
              background: #f8fafc;
            }}
            header {{
              background: #1e293b; color: #f8fafc;
              padding: 2rem 1.5rem; margin-bottom: 2rem;
              border-bottom: 4px solid #4C72B0;
            }}
            header h1 {{ margin: 0 0 0.4rem; font-size: 2rem; }}
            header p  {{ margin: 0; opacity: 0.7; }}
            nav {{
              background: #fff; border: 1px solid #e2e8f0;
              border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 2rem;
            }}
            nav h3 {{
              margin: 0 0 0.5rem; color: #4C72B0; font-size: 0.9rem;
              text-transform: uppercase; letter-spacing: 0.05em;
            }}
            nav ol {{ margin: 0; padding-left: 1.4rem; }}
            nav li {{ margin: 0.25rem 0; }}
            nav a  {{ color: #4C72B0; text-decoration: none; }}
            nav a:hover {{ text-decoration: underline; }}
            section {{
              background: #fff; border: 1px solid #e2e8f0;
              border-radius: 10px; padding: 1.5rem 1.8rem; margin-bottom: 2rem;
            }}
            h2 {{
              font-size: 1.35rem; color: #4C72B0; margin-top: 0;
              border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem;
            }}
            p {{ line-height: 1.65; margin: 0.8rem 0 0; }}
            img {{ max-width: 100%; display: block; }}
            /* Parameter description cards */
            .param-grid {{
              display: grid;
              grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
              gap: 0.8rem; margin: 1rem 0;
            }}
            .param-card {{
              background: #f1f5f9; border: 1px solid #e2e8f0;
              border-radius: 8px; padding: 0.9rem 1rem;
            }}
            .param-name {{
              font-weight: 700; font-size: 0.9rem;
              color: #4C72B0; margin-bottom: 0.4rem; font-family: monospace;
            }}
            .param-desc {{ font-size: 0.85rem; line-height: 1.55; color: #334155; }}
            /* Tuning tables */
            .tuning-tables {{ margin-top: 1rem; }}
            .tuning-tables h3 {{ font-size: 1rem; color: #1e293b; margin-bottom: 0.4rem; }}
            table.tuning-table {{
              border-collapse: collapse; font-size: 0.85rem; margin-bottom: 1rem;
            }}
            table.tuning-table th, table.tuning-table td {{
              border: 1px solid #e2e8f0; padding: 0.35rem 0.65rem;
            }}
            table.tuning-table th {{
              background: #1e293b; color: #f8fafc; font-weight: 600;
            }}
            table.tuning-table tr:nth-child(even) td {{ background: #f1f5f9; }}
            footer {{ margin-top: 3rem; font-size: 0.8rem; color: #94a3b8; text-align: center; }}
          </style>
        </head>
        <body>
          <header>
            <h1>DengAI — Pipeline Walkthrough</h1>
            <p>End-to-end journey from raw data to competition predictions.</p>
          </header>
          <nav>
            <h3>Contents</h3>
            {toc_html}
          </nav>
          {sections}
          <footer>Generated by python main.py --present</footer>
        </body>
        </html>
    """)

    print(f"[6/7] Writing {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(OUTPUT_PATH) // 1024
    print(f"[7/7] Done — {OUTPUT_PATH}  ({size_kb:,} KB)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DengAI pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python main.py              run pipeline (default)
              python main.py --tune       hyperparameter search
              python main.py --present    generate presentation
        """),
    )
    parser.add_argument("--tune",    action="store_true", help="Run hyperparameter search")
    parser.add_argument("--present", action="store_true", help="Generate presentation")
    args = parser.parse_args()

    if args.tune:
        run_tuning()
    elif args.present:
        run_presentation()
    else:
        run_pipeline()
