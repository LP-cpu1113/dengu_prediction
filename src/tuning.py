"""
tuning.py
─────────
Hyperparameter search for the Random Forest model.
Uses TimeSeriesSplit so validation always comes after training — no future leakage.

Import and call run_search() from tune.py; nothing in the main pipeline is touched.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# ── Parameter grid ────────────────────────────────────────────────────────────
# max_features : controls tree diversity (key with 125 features)
# min_samples_leaf : main regularizer for small dataset (~700 rows per city)
# max_depth : secondary regularizer

PARAM_GRID = {
    "max_features":      ["sqrt", 0.3, 0.5, 1.0],
    "min_samples_leaf":  [1, 5, 10, 20],
    "max_depth":         [10, 20, None],
}


def run_search(X, y, n_splits=5, n_estimators=100):
    """
    Run GridSearchCV with temporal cross-validation.

    n_estimators is kept low for speed — the production model uses more trees.
    n_jobs=-1 on GridSearchCV parallelises across parameter combinations;
    the individual RF uses n_jobs=1 to avoid nested parallelism issues.
    """
    cv = TimeSeriesSplit(n_splits=n_splits)

    base_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=1,
    )

    search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="neg_mean_absolute_error",
        refit=False,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_)
    results["mae"]   = -results["mean_test_score"]
    results["std"]   = results["std_test_score"]

    cols = [
        "param_max_features",
        "param_min_samples_leaf",
        "param_max_depth",
        "mae",
        "std",
    ]
    return results[cols].sort_values("mae").reset_index(drop=True)
