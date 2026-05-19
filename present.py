"""
present.py
──────────
Pipeline walkthrough presentation generator.

    python present.py

Runs the full pipeline, captures data at each stage, and writes
output/presentation.html — a self-contained HTML file with base64-encoded
images and a table of contents.
"""

import os
import textwrap

import pandas as pd

from src.data_loader  import load_data
from src.cleaning     import clean
from src.features     import build_features, get_feature_cols
from src.presentation import (
    _fig_to_base64,
    plot_dataset_overview,
    plot_missing_values_raw,
    plot_cleaning_interpolation,
    plot_feature_engineering,
    plot_tuning_results,
    plot_feature_importance,
    plot_predictions,
)

OUTPUT_DIR  = "output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "presentation.html")
TUNING_CSV  = os.path.join(OUTPUT_DIR, "tuning_results.csv")


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _img_tag(b64, alt=""):
    return (f'<img src="data:image/png;base64,{b64}" '
            f'alt="{alt}" style="max-width:100%;border:1px solid #e2e8f0;'
            f'border-radius:8px;margin-top:0.5rem;" />')


def _section(section_id, title, body_html):
    return textwrap.dedent(f"""\
        <section id="{section_id}">
          <h2>{title}</h2>
          {body_html}
        </section>""")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load raw data ────────────────────────────────────────────────
    print("[1/7] Loading raw data...")
    df_raw = load_data()
    print(f"      {len(df_raw)} rows  "
          f"({df_raw['type'].value_counts().to_dict()})")

    # ── Step 2: Clean ────────────────────────────────────────────────────────
    print("[2/7] Cleaning...")
    df_clean = clean(df_raw)

    # ── Step 3: Build features ───────────────────────────────────────────────
    print("[3/7] Building features...")
    df_feat   = build_features(df_clean)
    feat_cols = get_feature_cols(df_feat)
    print(f"      {len(feat_cols)} feature columns")

    # ── Step 4: Generate plots ───────────────────────────────────────────────
    print("[4/7] Generating plots...")

    # Section 1
    print("      Section 1 — Dataset overview")
    fig1 = plot_dataset_overview(df_raw)
    s1_body = (
        _img_tag(_fig_to_base64(fig1), "Dataset overview") +
        "<p>1,456 training weeks across two cities. "
        "Target: weekly dengue case count.</p>"
    )

    # Section 2
    print("      Section 2 — Missing values")
    fig2 = plot_missing_values_raw(df_raw)
    if fig2 is not None:
        # Identify top missing columns for the description
        counts = (
            df_raw
            .drop(columns=["type", "total_cases", "city", "year",
                            "weekofyear", "week_start_date"], errors="ignore")
            .isnull()
            .sum()
        )
        counts = counts[counts > 0].sort_values(ascending=False)
        top_cols = ", ".join(f"<code>{c}</code>" for c in counts.index[:4])
        s2_body = (
            _img_tag(_fig_to_base64(fig2), "Missing values") +
            f"<p>Most affected columns: {top_cols}. "
            "All missing values are filled before modelling.</p>"
        )
    else:
        s2_body = "<p>No missing values detected in the raw data.</p>"

    # Section 3
    print("      Section 3 — Cleaning / interpolation")
    fig3 = plot_cleaning_interpolation(df_raw, df_clean)
    if fig3 is not None:
        n_missing = int(df_raw["ndvi_ne"].isnull().sum())
        s3_body = (
            _img_tag(_fig_to_base64(fig3), "Interpolation") +
            f"<p>{n_missing} missing values in <code>ndvi_ne</code> filled "
            "using per-city linear interpolation. "
            "The shaded region on the top panel shows where NaN gaps existed. "
            "The bottom panel shows the smooth, gap-free series used by the model.</p>"
        )
    else:
        s3_body = "<p><code>ndvi_ne</code> column not found.</p>"

    # Section 4
    print("      Section 4 — Feature engineering")
    fig4 = plot_feature_engineering(df_feat)
    s4_body = (
        _img_tag(_fig_to_base64(fig4), "Feature engineering") +
        "<p>"
        "<strong>Lag features</strong> shift a climate signal backward in time, "
        "capturing the delayed biological effect between an environmental event "
        "(e.g. a humid week) and the resulting spike in reported cases. "
        "Lags 1–8 weeks are added for every key signal. "
        "<strong>Rolling means</strong> smooth out week-to-week noise and "
        "capture accumulated environmental conditions — a 4-week wet period is "
        "more predictive than a single wet day."
        "</p>"
    )

    # Section 5
    print("      Section 5 — Hyperparameter tuning")
    tuning_result = plot_tuning_results(TUNING_CSV)
    if tuning_result[0] is not None:
        fig5, top5_html = tuning_result
        s5_body = (
            _img_tag(_fig_to_base64(fig5), "Tuning results") +
            "<p><code>min_samples_leaf</code> is the dominant regulariser. "
            "The default value of 1 allows the trees to memorise individual "
            "training weeks, which hurts generalisation on this small dataset "
            "(≈700 rows per city). A value of 20 gives the best validation MAE.</p>" +
            '<div class="tuning-tables">' + top5_html + "</div>"
        )
    else:
        _, msg = tuning_result
        s5_body = msg

    # Section 6
    print("      Section 6 — Feature importance (training RF models — may take ~30 s)")
    fig6 = plot_feature_importance(df_feat, feat_cols, top_n=20)
    s6_body = (
        _img_tag(_fig_to_base64(fig6), "Feature importance") +
        "<p>Which signals the Random Forest relied on most. "
        "The importance score is the mean decrease in node impurity across all "
        "trees, normalised to sum to 1. Higher = more predictive. "
        "Both cities lean heavily on temperature and humidity, but the exact "
        "ranking differs — Iquitos relies more on recent precipitation.</p>"
    )

    # Section 7
    print("      Section 7 — Predictions (re-training production models)")
    fig7 = plot_predictions(df_feat, feat_cols)
    s7_body = (
        _img_tag(_fig_to_base64(fig7), "Predictions") +
        "<p>Final predictions for the competition submission period. "
        "The solid line is the training history; the dashed extension is the "
        "model's forecast. The dotted vertical line marks the train/test boundary. "
        "Both cities show the model tracking seasonal patterns, with Iquitos "
        "exhibiting a flatter, lower-amplitude forecast as expected given its "
        "smaller baseline case counts.</p>"
    )

    # ── Step 5: Assemble HTML ────────────────────────────────────────────────
    print("[5/7] Assembling HTML...")

    toc_items = [
        ("sec1", "1. The Dataset"),
        ("sec2", "2. Missing Values (Raw Data)"),
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

    sections_html = "\n".join([
        _section("sec1", "1. The Dataset",                  s1_body),
        _section("sec2", "2. Missing Values (Raw Data)",    s2_body),
        _section("sec3", "3. Cleaning: Interpolation",      s3_body),
        _section("sec4", "4. Feature Engineering",          s4_body),
        _section("sec5", "5. Hyperparameter Tuning",        s5_body),
        _section("sec6", "6. Feature Importance",           s6_body),
        _section("sec7", "7. Predictions",                  s7_body),
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
              max-width: 960px;
              margin: 0 auto;
              padding: 0 1.5rem 3rem;
              color: #1e293b;
              background: #f8fafc;
            }}
            header {{
              background: #1e293b;
              color: #f8fafc;
              padding: 2rem 1.5rem;
              margin: 0 -1.5rem 2rem;
              border-bottom: 4px solid #4C72B0;
            }}
            header h1 {{
              margin: 0 0 0.4rem;
              font-size: 2rem;
              font-weight: 700;
            }}
            header p {{
              margin: 0;
              opacity: 0.7;
              font-size: 1rem;
            }}
            nav {{
              background: #fff;
              border: 1px solid #e2e8f0;
              border-radius: 10px;
              padding: 1.2rem 1.5rem;
              margin-bottom: 2.5rem;
            }}
            nav h3 {{
              margin: 0 0 0.6rem;
              color: #4C72B0;
              font-size: 1rem;
              text-transform: uppercase;
              letter-spacing: 0.05em;
            }}
            nav ol {{
              margin: 0;
              padding-left: 1.4rem;
            }}
            nav ol li {{
              margin: 0.25rem 0;
            }}
            nav a {{
              color: #4C72B0;
              text-decoration: none;
            }}
            nav a:hover {{
              text-decoration: underline;
            }}
            section {{
              background: #fff;
              border: 1px solid #e2e8f0;
              border-radius: 10px;
              padding: 1.5rem 1.8rem;
              margin-bottom: 2rem;
            }}
            h2 {{
              font-size: 1.35rem;
              color: #4C72B0;
              margin-top: 0;
              border-bottom: 2px solid #e2e8f0;
              padding-bottom: 0.5rem;
            }}
            p {{
              line-height: 1.65;
              margin: 0.8rem 0 0;
            }}
            img {{
              max-width: 100%;
              display: block;
            }}
            /* Tuning tables */
            .tuning-tables {{ margin-top: 1rem; }}
            .tuning-tables h3 {{
              font-size: 1rem;
              color: #1e293b;
              margin-bottom: 0.4rem;
            }}
            table.tuning-table {{
              border-collapse: collapse;
              font-size: 0.85rem;
              margin-bottom: 1rem;
            }}
            table.tuning-table th, table.tuning-table td {{
              border: 1px solid #e2e8f0;
              padding: 0.35rem 0.65rem;
            }}
            table.tuning-table th {{
              background: #1e293b;
              color: #f8fafc;
              font-weight: 600;
            }}
            table.tuning-table tr:nth-child(even) td {{
              background: #f1f5f9;
            }}
            footer {{
              margin-top: 3rem;
              font-size: 0.8rem;
              color: #94a3b8;
              text-align: center;
            }}
          </style>
        </head>
        <body>
          <header>
            <h1>DengAI — Pipeline Walkthrough</h1>
            <p>End-to-end journey from raw data to competition predictions.</p>
          </header>
          <nav>
            <h3>Table of Contents</h3>
            {toc_html}
          </nav>
          {sections_html}
          <footer>Generated by present.py</footer>
        </body>
        </html>
    """)

    # ── Step 6: Write file ───────────────────────────────────────────────────
    print(f"[6/7] Writing {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(OUTPUT_PATH) // 1024
    print(f"[7/7] Done — {OUTPUT_PATH}  ({size_kb:,} KB)")


if __name__ == "__main__":
    run()
