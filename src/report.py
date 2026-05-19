"""
report.py
─────────
Generates a self-contained HTML report with EDA plots.

To add a new plot:
  1. Write a function that takes df and returns a matplotlib Figure.
  2. Add it to the PLOTS list at the bottom of this file.
"""

import io, base64, textwrap
import matplotlib
matplotlib.use("Agg")   # no display needed — we're writing to a file
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ── Utility ───────────────────────────────────────────────────────────────────

def _fig_to_base64(fig):
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_missing_values(df):
    """Bar chart of missing value rate per column."""
    rates = df.drop(columns=["type"], errors="ignore").isnull().mean().mul(100)
    rates = rates[rates > 0].sort_values(ascending=False)

    if rates.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    rates.plot.bar(ax=ax, color="#DD8452", edgecolor="white")
    ax.set_title("Missing Value Rate per Column (%)", fontweight="bold")
    ax.set_ylabel("% missing")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def plot_cases_over_time(df):
    """Line chart of total_cases over time, one panel per city."""
    train    = df[df["type"] == "train"].copy()
    cities   = {"sj": "San Juan", "iq": "Iquitos"}
    colors   = {"sj": "#4C72B0", "iq": "#DD8452"}

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    for ax, (city_code, city_name) in zip(axes, cities.items()):
        subset = train[train["city"] == city_code]["total_cases"].values
        ax.fill_between(range(len(subset)), subset, alpha=0.2, color=colors[city_code])
        ax.plot(subset, color=colors[city_code], linewidth=0.9)
        ax.axhline(subset.mean(), color="grey", linestyle="--", linewidth=1,
                   label=f"Mean = {subset.mean():.0f}")
        ax.set_title(f"{city_name} — Weekly Dengue Cases", fontweight="bold")
        ax.set_ylabel("Cases")
        ax.legend()

    axes[1].set_xlabel("Week number")
    plt.tight_layout()
    return fig


def plot_case_distribution(df):
    """Histogram of total_cases split by city."""
    train  = df[df["type"] == "train"].copy()
    cities = {"sj": "San Juan", "iq": "Iquitos"}
    colors = {"sj": "#4C72B0", "iq": "#DD8452"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (city_code, city_name) in zip(axes, cities.items()):
        subset = train[train["city"] == city_code]["total_cases"]
        ax.hist(subset, bins=30, color=colors[city_code], edgecolor="white", alpha=0.85)
        ax.set_title(city_name, fontweight="bold")
        ax.set_xlabel("Cases per week")
        ax.set_ylabel("Frequency")

    fig.suptitle("Distribution of Weekly Case Counts", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_feature_correlation(df):
    """Heatmap of Pearson correlation between features and total_cases (train only)."""
    train    = df[df["type"] == "train"].copy()
    skip     = {"type", "total_cases"}
    num_cols = [c for c in train.select_dtypes("number").columns if c not in skip]

    corrs = (
        train[num_cols]
        .corrwith(train["total_cases"])
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(5, 8))
    colors = ["#f87171" if v < 0 else "#4C72B0" for v in corrs.values]
    corrs.plot.barh(ax=ax, color=colors, edgecolor="white")
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_title("Feature Correlation with total_cases\n(train set only)", fontweight="bold")
    ax.set_xlabel("Pearson correlation")
    plt.tight_layout()
    return fig


# ── Register plots here ───────────────────────────────────────────────────────
# Each entry is (section_title, plot_function).
# Add a new tuple to include a new plot in the report.

PLOTS = [
    ("Missing Values",            plot_missing_values),
    ("Cases Over Time",           plot_cases_over_time),
    ("Case Count Distribution",   plot_case_distribution),
    ("Feature Correlations",      plot_feature_correlation),
]


# ── HTML assembly ─────────────────────────────────────────────────────────────

def generate_report(df, output_path="report.html"):
    sections = []

    for title, plot_fn in PLOTS:
        fig = plot_fn(df)
        if fig is None:
            continue
        img = _fig_to_base64(fig)
        sections.append(f"""
        <section>
          <h2>{title}</h2>
          <img src="data:image/png;base64,{img}" alt="{title}" />
        </section>""")

    html = textwrap.dedent(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <title>DengAI — EDA Report</title>
          <style>
            body   {{ font-family: system-ui, sans-serif; max-width: 960px;
                      margin: 2rem auto; padding: 0 1.5rem; color: #1e293b; }}
            h1     {{ font-size: 2rem; border-bottom: 3px solid #4C72B0;
                      padding-bottom: 0.4rem; }}
            h2     {{ font-size: 1.3rem; color: #4C72B0; margin-top: 2.5rem; }}
            section{{ margin-bottom: 2rem; }}
            img    {{ max-width: 100%; border: 1px solid #e2e8f0;
                      border-radius: 8px; margin-top: 0.5rem; }}
            footer {{ margin-top: 3rem; font-size: 0.8rem; color: #94a3b8; }}
          </style>
        </head>
        <body>
          <h1>DengAI — Exploratory Data Analysis</h1>
          <p>Preliminary investigation report. Add new plots in <code>src/report.py</code>.</p>
          {"".join(sections)}
          <footer>Generated by src/report.py</footer>
        </body>
        </html>
    """).strip()

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Report saved: {output_path}")
