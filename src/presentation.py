"""
presentation.py
───────────────
Plot functions for the pipeline walkthrough presentation.

Each function returns a matplotlib Figure (or None if data is unavailable).
Use _fig_to_base64() from this module to embed figures into HTML.

Sections
--------
1. plot_dataset_overview        — cases over time + distribution
2. plot_missing_values_raw      — horizontal bar chart of missing counts
3. plot_cleaning_interpolation  — ndvi_ne before/after interpolation (SJ)
4. plot_feature_engineering     — lag features + rolling mean demo
5. plot_tuning_results          — MAE vs min_samples_leaf bar charts
6. plot_feature_importance      — top-20 importances for both cities
7. plot_predictions             — train history + test predictions
"""

import io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ── Colour palette ────────────────────────────────────────────────────────────
C_SJ   = "#4C72B0"
C_IQ   = "#DD8452"
C_MEAN = "#94a3b8"


# ── Utility ───────────────────────────────────────────────────────────────────

def _fig_to_base64(fig):
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ── Section 1 — The Dataset ───────────────────────────────────────────────────

def plot_dataset_overview(df_raw):
    """
    Returns a figure with:
      - Top two panels: cases over time for San Juan and Iquitos.
      - Bottom two panels: histogram of total_cases distribution per city.
    """
    train  = df_raw[df_raw["type"] == "train"].copy()
    cities = [("sj", "San Juan", C_SJ), ("iq", "Iquitos", C_IQ)]

    fig = plt.figure(figsize=(14, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)

    # ── cases over time ───────────────────────────────────────────────────────
    for col_idx, (code, name, color) in enumerate(cities):
        ax  = fig.add_subplot(gs[0, col_idx])
        sub = train[train["city"] == code].reset_index(drop=True)
        x   = np.arange(len(sub))
        y   = sub["total_cases"].values

        ax.fill_between(x, y, alpha=0.18, color=color)
        ax.plot(x, y, color=color, linewidth=0.9)
        mean_val = y.mean()
        ax.axhline(mean_val, color=C_MEAN, linestyle="--", linewidth=1.2,
                   label=f"Mean = {mean_val:.0f}")
        ax.set_title(f"{name} — Weekly Cases", fontweight="bold")
        ax.set_ylabel("Cases")
        ax.legend(fontsize=8)

        # year-boundary ticks
        years     = sub["year"].values
        tick_pos  = [i for i in range(len(years))
                     if i == 0 or years[i] != years[i - 1]]
        tick_labs = [str(years[i]) for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labs, rotation=90, fontsize=7)

    # ── histograms ────────────────────────────────────────────────────────────
    for col_idx, (code, name, color) in enumerate(cities):
        ax     = fig.add_subplot(gs[1, col_idx])
        subset = train[train["city"] == code]["total_cases"]
        ax.hist(subset, bins=30, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"{name} — Case Distribution", fontweight="bold")
        ax.set_xlabel("Cases per week")
        ax.set_ylabel("Frequency")

    fig.suptitle("The Dataset", fontsize=15, fontweight="bold", y=1.01)
    return fig


# ── Section 2 — Missing Values (Raw Data) ────────────────────────────────────

def plot_missing_values_raw(df_raw):
    """
    Horizontal bar chart of missing value COUNT per column (raw data, sorted
    descending). Only columns with at least one missing value are shown.
    """
    counts = (
        df_raw
        .drop(columns=["type", "total_cases", "city", "year",
                        "weekofyear", "week_start_date"], errors="ignore")
        .isnull()
        .sum()
    )
    counts = counts[counts > 0].sort_values(ascending=True)   # ascending → longest bar at top

    if counts.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, max(3, len(counts) * 0.38)))
    counts.plot.barh(ax=ax, color=C_IQ, edgecolor="white")
    ax.set_title("Missing Value Count per Column (raw data)",
                 fontweight="bold")
    ax.set_xlabel("Number of missing values")
    ax.set_ylabel("")
    # annotate each bar with its count
    for bar, val in zip(ax.patches, counts.values):
        ax.text(bar.get_width() + counts.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(int(val)), va="center", fontsize=8)
    plt.tight_layout()
    return fig


# ── Section 3 — Cleaning: Interpolation ──────────────────────────────────────

def plot_cleaning_interpolation(df_raw, df_clean):
    """
    Two stacked panels for ndvi_ne (San Juan only):
      Top    — raw data (NaN gaps break the line); gap region shaded.
      Bottom — after linear interpolation (smooth continuous line).
    X-axis: integer row index within San Juan.
    """
    raw_sj   = df_raw[(df_raw["city"] == "sj") &
                      (df_raw["type"] == "train")].reset_index(drop=True)
    # df_clean has city encoded as 0/1 after clean()
    clean_sj = df_clean[(df_clean["city"] == 0) &
                        (df_clean["type"] == "train")].reset_index(drop=True)

    col = "ndvi_ne"
    if col not in raw_sj.columns or col not in clean_sj.columns:
        return None

    raw_vals   = raw_sj[col].values
    clean_vals = clean_sj[col].values
    x          = np.arange(len(raw_vals))

    # find gap regions (runs of NaN)
    nan_mask = np.isnan(raw_vals)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 6),
                                         sharex=True,
                                         gridspec_kw={"hspace": 0.35})

    # Top: raw
    ax_top.plot(x, raw_vals, color=C_SJ, linewidth=0.9)
    ax_top.set_title("Raw ndvi_ne — San Juan (NaN gaps break the line)",
                     fontweight="bold")
    ax_top.set_ylabel("NDVI NE")

    # shade the contiguous gap regions
    if nan_mask.any():
        in_gap   = False
        gap_start = None
        for i, is_nan in enumerate(nan_mask):
            if is_nan and not in_gap:
                in_gap    = True
                gap_start = i
            elif not is_nan and in_gap:
                in_gap = False
                ax_top.axvspan(gap_start - 0.5, i - 0.5,
                               color="#f87171", alpha=0.25,
                               label="NaN region" if gap_start == (np.where(nan_mask)[0][0]) else "")
        if in_gap:
            ax_top.axvspan(gap_start - 0.5, len(nan_mask) - 0.5,
                           color="#f87171", alpha=0.25)
        ax_top.legend(fontsize=8)

    # Bottom: after interpolation
    ax_bot.plot(x, clean_vals, color=C_SJ, linewidth=0.9)
    ax_bot.set_title("After linear interpolation — continuous signal",
                     fontweight="bold")
    ax_bot.set_ylabel("NDVI NE")
    ax_bot.set_xlabel("Row index (San Juan training weeks)")

    fig.suptitle("Cleaning: Interpolation", fontsize=13, fontweight="bold")
    return fig


# ── Section 4 — Feature Engineering ──────────────────────────────────────────

def plot_feature_engineering(df_feat):
    """
    Two side-by-side panels using the first 104 rows of San Juan training data:
      Left  — specific humidity (raw) vs lag4 and lag8
      Right — station_precip_mm (bars) + station_precip_roll4 (line)
    df_feat must already have had build_features() applied.
    """
    # city is 0 after clean()
    sj = df_feat[(df_feat["city"] == 0) &
                 (df_feat["type"] == "train")].reset_index(drop=True).head(104)
    x  = np.arange(len(sj))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5),
                                     gridspec_kw={"wspace": 0.35})

    # ── left: lag features ────────────────────────────────────────────────────
    hum_col  = "reanalysis_specific_humidity_g_per_kg"
    lag4_col = "specific_humidity_lag4"
    lag8_col = "specific_humidity_lag8"

    if hum_col in sj.columns:
        ax_l.plot(x, sj[hum_col],  color=C_SJ,   linewidth=1.5,
                  label="specific_humidity (raw)")
    if lag4_col in sj.columns:
        ax_l.plot(x, sj[lag4_col], color=C_IQ,   linewidth=1.2,
                  linestyle="--", label="lag4")
    if lag8_col in sj.columns:
        ax_l.plot(x, sj[lag8_col], color="#55a868", linewidth=1.0,
                  linestyle=":",  label="lag8")

    ax_l.set_title("Lag Features — signal shifted back in time",
                   fontweight="bold")
    ax_l.set_xlabel("Week (San Juan training, first 104)")
    ax_l.set_ylabel("Specific Humidity (g/kg)")
    ax_l.legend(fontsize=8)

    # ── right: rolling mean ───────────────────────────────────────────────────
    precip_col  = "station_precip_mm"
    roll4_col   = "station_precip_roll4"

    if precip_col in sj.columns:
        ax_r.bar(x, sj[precip_col], color=C_SJ, alpha=0.35,
                 label="station_precip_mm")
    if roll4_col in sj.columns:
        ax_r.plot(x, sj[roll4_col], color="#DD8452", linewidth=1.8,
                  label="4-week rolling mean")

    ax_r.set_title("Rolling Mean — 4-week smoothed precipitation",
                   fontweight="bold")
    ax_r.set_xlabel("Week (San Juan training, first 104)")
    ax_r.set_ylabel("Precipitation (mm)")
    ax_r.legend(fontsize=8)

    fig.suptitle("Feature Engineering", fontsize=13, fontweight="bold")
    return fig


# ── Section 5 — Hyperparameter Tuning ────────────────────────────────────────

PARAM_DESCRIPTIONS = [
    (
        "n_estimators = 500",
        "Number of trees in the forest. Each tree trains on a random bootstrap "
        "sample of the data. More trees lower prediction variance but yield "
        "diminishing returns past ~300. We use 500 for stability. "
        "This parameter does <em>not</em> affect bias.",
    ),
    (
        "max_features = 'sqrt'",
        "Features randomly considered at each split. With 125 features, "
        "<code>sqrt</code> ≈ 11 features per split. Lower values force more "
        "diversity between trees (less correlated ensemble). The regression "
        "default of <code>1.0</code> (all features) makes trees too similar, "
        "weakening the ensemble.",
    ),
    (
        "min_samples_leaf = 20",
        "Minimum training samples required at a leaf node. "
        "The most important regulariser for this small dataset (~700 rows per city). "
        "The default of <code>1</code> lets trees memorise individual training "
        "weeks, overfitting badly. A value of 20 forces each leaf to represent "
        "at least 20 weeks of data, smoothing noisy outbreak spikes.",
    ),
    (
        "max_depth = None",
        "Maximum depth of each tree. We found this has negligible effect once "
        "<code>min_samples_leaf=20</code> is set — the leaf constraint already "
        "controls tree complexity. Left unrestricted.",
    ),
]


def _param_cards_html():
    """Return an HTML block of parameter description cards."""
    cards = []
    for name, desc in PARAM_DESCRIPTIONS:
        cards.append(
            f"<div class='param-card'>"
            f"<div class='param-name'>{name}</div>"
            f"<div class='param-desc'>{desc}</div>"
            f"</div>"
        )
    return "<div class='param-grid'>" + "".join(cards) + "</div>"


def plot_tuning_results(tuning_csv_path="output/tuning_results.csv"):
    """
    Returns (fig_lines, fig_heatmap, top5_html) or (None, None, error_html).

    fig_lines   : MAE vs min_samples_leaf — one line per max_features, per city.
    fig_heatmap : min_samples_leaf × max_features interaction heatmap, per city.
    top5_html   : HTML tables of the top 5 parameter combinations per city.
    """
    try:
        results = pd.read_csv(tuning_csv_path)
    except FileNotFoundError:
        msg = ("<p style='color:#ef4444;font-weight:bold;'>"
               "Tuning results not found — run "
               "<code>python main.py --tune</code> first.</p>")
        return None, None, msg

    cities      = results["city"].unique()
    city_colors = {"San Juan": C_SJ, "Iquitos": C_IQ}
    mf_styles   = {"sqrt": "-", "0.3": "--", "0.5": "-.", "1.0": ":"}
    mf_colors   = {"sqrt": "#4C72B0", "0.3": "#DD8452",
                   "0.5": "#55a868", "1.0": "#c44e52"}

    # ── Figure 1: line plots ──────────────────────────────────────────────────
    fig_lines, axes = plt.subplots(1, len(cities), figsize=(13, 5),
                                   gridspec_kw={"wspace": 0.35})
    if len(cities) == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        city_df = results[results["city"] == city]
        mf_vals = sorted(city_df["param_max_features"].astype(str).unique())

        for mf in mf_vals:
            subset = (
                city_df[city_df["param_max_features"].astype(str) == mf]
                .groupby("param_min_samples_leaf")["mae"]
                .mean()
                .reset_index()
                .sort_values("param_min_samples_leaf")
            )
            ax.plot(
                subset["param_min_samples_leaf"].astype(str),
                subset["mae"],
                marker="o",
                linestyle=mf_styles.get(mf, "-"),
                color=mf_colors.get(mf, "#888"),
                linewidth=1.8,
                label=f"max_features={mf}",
            )

        ax.set_title(city, fontweight="bold")
        ax.set_xlabel("min_samples_leaf")
        ax.set_ylabel("MAE (5-fold TimeSeriesSplit)")
        ax.legend(fontsize=8)

    fig_lines.suptitle("MAE vs min_samples_leaf  (averaged over max_depth)",
                       fontsize=12, fontweight="bold")

    # ── Figure 2: interaction heatmaps ────────────────────────────────────────
    fig_heat, axes = plt.subplots(1, len(cities), figsize=(13, 4),
                                  gridspec_kw={"wspace": 0.4})
    if len(cities) == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        city_df = results[results["city"] == city]
        pivot   = (
            city_df
            .groupby(["param_min_samples_leaf", "param_max_features"])["mae"]
            .mean()
            .reset_index()
            .pivot(index="param_min_samples_leaf",
                   columns="param_max_features",
                   values="mae")
        )
        sns.heatmap(
            pivot, ax=ax,
            annot=True, fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "MAE"},
            linewidths=0.4,
        )
        ax.set_title(city, fontweight="bold")
        ax.set_xlabel("max_features")
        ax.set_ylabel("min_samples_leaf")

    fig_heat.suptitle(
        "MAE Heatmap — min_samples_leaf × max_features  (averaged over max_depth)",
        fontsize=12, fontweight="bold",
    )

    # ── Top-5 HTML tables ─────────────────────────────────────────────────────
    table_rows = []
    for city in cities:
        top5 = (
            results[results["city"] == city]
            .sort_values("mae")
            .head(5)
            [["param_max_features", "param_min_samples_leaf",
              "param_max_depth", "mae", "std"]]
            .rename(columns={
                "param_max_features":     "max_features",
                "param_min_samples_leaf": "min_samples_leaf",
                "param_max_depth":        "max_depth",
            })
        )
        top5["mae"] = top5["mae"].round(2)
        top5["std"] = top5["std"].round(2)
        table_rows.append(f"<h3 style='margin-top:1.2rem'>{city} — Top 5</h3>")
        table_rows.append(
            top5.to_html(index=False, border=0,
                         classes="tuning-table", justify="left")
        )

    top5_html = "\n".join(table_rows)
    return fig_lines, fig_heat, top5_html


# ── Section 6 — Feature Importance ───────────────────────────────────────────

def plot_feature_importance(df_feat, feat_cols, top_n=20):
    """
    Train the production RF on each city's training data and plot the top-N
    feature importances side-by-side.
    df_feat must already have clean() + build_features() applied.
    """
    from sklearn.ensemble import RandomForestRegressor

    cities = [(0, "San Juan", C_SJ), (1, "Iquitos", C_IQ)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                             gridspec_kw={"wspace": 0.6})

    for ax, (code, name, color) in zip(axes, cities):
        city_train = df_feat[(df_feat["city"] == code) &
                             (df_feat["type"] == "train")]
        X = city_train[feat_cols]
        y = city_train["total_cases"]

        rf = RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X, y)

        importance = pd.Series(rf.feature_importances_, index=feat_cols)
        top = importance.sort_values(ascending=False).head(top_n).sort_values()

        ax.barh(top.index, top.values, color=color, edgecolor="white")
        ax.set_title(f"{name}", fontweight="bold")
        ax.set_xlabel("Mean decrease in impurity")
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(f"Top {top_n} Feature Importances — Production RF Model",
                 fontsize=13, fontweight="bold")
    return fig


# ── Section 7 — Predictions ───────────────────────────────────────────────────

def plot_predictions(df_feat, feat_cols):
    """
    For each city: training total_cases (solid line) followed by test
    predictions (dashed line, lower alpha).  A vertical dashed line marks the
    train/test boundary.
    """
    from src.models import train_model, predict as model_predict

    cities = [(0, "San Juan", C_SJ), (1, "Iquitos", C_IQ)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                             gridspec_kw={"wspace": 0.35})

    for ax, (code, name, color) in zip(axes, cities):
        city_df    = df_feat[df_feat["city"] == code]
        train_df   = city_df[city_df["type"] == "train"]
        test_df    = city_df[city_df["type"] == "test"]

        X_train    = train_df[feat_cols]
        y_train    = train_df["total_cases"]
        X_test     = test_df[feat_cols]

        model      = train_model(X_train, y_train)
        preds      = model_predict(model, X_test)

        n_train    = len(y_train)
        n_test     = len(preds)

        x_train    = np.arange(n_train)
        x_test     = np.arange(n_train, n_train + n_test)

        ax.plot(x_train, y_train.values, color=color, linewidth=0.9,
                label="Training data")
        ax.plot(x_test, preds, color=color, linewidth=1.1,
                linestyle="--", alpha=0.65, label="Test predictions")
        ax.axvline(n_train - 0.5, color="#64748b", linewidth=1.2,
                   linestyle=":", label="Train / Test boundary")

        ax.set_title(f"{name}", fontweight="bold")
        ax.set_ylabel("Cases")
        ax.set_xlabel("Week index")
        ax.legend(fontsize=8)

    fig.suptitle("Predictions — Training History and Test Forecast",
                 fontsize=13, fontweight="bold")
    return fig
