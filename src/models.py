"""
models.py
─────────
Model definitions. Each model is a plain function: fit(X, y) → model object.

To try a new model:
  1. Add a train_<name>() function here.
  2. Swap it in for train_model() in main.py — nothing else changes.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def train_random_forest(X, y):
    model = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X, y)
    return model


# ── Plug-in point ─────────────────────────────────────────────────────────────
# Change this one line to switch the model used by the pipeline.

def train_model(X, y):
    return train_random_forest(X, y)


# ── Predict ───────────────────────────────────────────────────────────────────

def predict(model, X):
    raw = model.predict(X)
    return np.clip(np.round(raw), 0, None).astype(int)
