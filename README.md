# DengAI — Predicting Dengue Fever Spread

A machine learning pipeline for the [DrivenData DengAI competition](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/), which asks competitors to predict the number of dengue fever cases reported each week in two cities: **San Juan, Puerto Rico** (1990–2008) and **Iquitos, Peru** (2000–2010). Submissions are evaluated by **Mean Absolute Error (MAE)**.

This project was built as an educational exercise for the Data Science Retreat (Berlin) program, with an emphasis on interpretability, clean modularity, and principled feature engineering.

---

## Abstract

Dengue fever transmission is closely tied to climate conditions — mosquito breeding depends on temperature, humidity, and rainfall, with biological delays of several weeks between environmental conditions and reported cases. We model this as a supervised regression problem, training separate Random Forest models for each city.

Key findings:
- **Lag and rolling features matter.** Climate signals lagged 2–8 weeks, along with 4-week rolling means, consistently outperform raw instantaneous readings. The lag captures the delayed biological lifecycle (egg → larva → adult → bite → incubation → report). The rolling mean captures the effect of *sustained* favourable conditions rather than isolated events.
- **Regularisation is critical.** With only ~700 training rows per city and 125 engineered features, the default Random Forest settings overfit severely. Setting `min_samples_leaf=20` (each leaf must represent at least 20 weeks of data) proved the single most impactful hyperparameter change, reducing cross-validated MAE by ~15% for San Juan.
- **Feature diversity improves the ensemble.** Setting `max_features='sqrt'` forces each tree to see only ~11 of 125 features per split, creating genuinely diverse trees and a stronger ensemble.
- **Interpolation beats mean imputation.** Filling missing climate values with per-city linear interpolation (rather than the global training mean) better respects the temporal structure of the data and produces smoother feature series.

The full pipeline walkthrough — including EDA plots, cleaning steps, feature engineering illustrations, hyperparameter tuning results, and feature importances — is available in `output/presentation.html`.

---

## Project Structure

```
dengu_prediction/
├── main.py              # single entry point (see Usage below)
├── data/                # raw CSVs from DrivenData
├── src/
│   ├── data_loader.py   # load and merge train/test CSVs
│   ├── cleaning.py      # missing value imputation, city encoding
│   ├── features.py      # feature engineering (lags, rolling means)
│   ├── models.py        # Random Forest model definition
│   ├── report.py        # EDA HTML report
│   ├── tuning.py        # GridSearchCV with TimeSeriesSplit
│   └── presentation.py  # presentation plot functions
├── notebooks/           # exploratory notebooks
├── output/              # generated files (predictions, reports)
└── requirements.txt
```

---

## Setup

This project uses **Python 3.11.9** managed with pyenv.

```bash
# create and activate the virtualenv
pyenv virtualenv 3.11.9 dengai
pyenv local dengai

# install dependencies
pip install -r requirements.txt
```

---

## Usage

All functionality is accessed through a single entry point:

### Run the pipeline (default)

Generates `output/predictions.csv` (competition submission) and `output/report.html` (EDA report).

```bash
python main.py
```

### Run hyperparameter tuning

Searches over `max_features`, `min_samples_leaf`, and `max_depth` using 5-fold `TimeSeriesSplit` cross-validation on training data only. Results saved to `output/tuning_results.csv`.

```bash
python main.py --tune
```

> **Note:** Uses 100 trees per fit for speed (~5–10 minutes on a modern laptop). The production model uses 500 trees.

### Generate the presentation

Builds a self-contained `output/presentation.html` with all pipeline stages illustrated: raw data, cleaning, feature engineering, hyperparameter tuning (requires `--tune` to have been run first), feature importances, and final predictions.

```bash
python main.py --present
```

---

## Features Engineered

For each key climate signal (specific humidity, average/min/max temperature, dew point, precipitation), the pipeline creates:

| Type | Description |
|---|---|
| Raw | Original measurement |
| Rolling mean (4w) | 4-week average — captures *sustained* conditions |
| Lag (1–8 weeks) | Signal from N weeks ago — captures biological delay |
| Lag of rolling (1–8 weeks) | "What were sustained conditions N weeks ago?" |

NDVI (vegetation index) is averaged across four spatial quadrants and lagged 1–8 weeks. Total: **125 features**.

---

## Model

Two independent `RandomForestRegressor` models — one per city.

| Hyperparameter | Value | Reason |
|---|---|---|
| `n_estimators` | 500 | Stable variance; diminishing returns past ~300 |
| `max_features` | `'sqrt'` | ~11 features/split; forces tree diversity |
| `min_samples_leaf` | 20 | Key regulariser; prevents overfitting on small dataset |
| `max_depth` | None | Irrelevant once `min_samples_leaf` is set |

Best cross-validated MAE (5-fold TimeSeriesSplit):

| City | MAE |
|---|---|
| San Juan | ~32.6 |
| Iquitos | ~6.6 |
