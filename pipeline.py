from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# ── Define features and target ────────────────────────────────────────────────

FEATURES = [
    'weekofyear',
    'ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw',
    'precipitation_amt_mm', 'station_precip_mm',
]
TARGET = 'total_cases'

X = df[FEATURES]
y = df[TARGET]

# ── Train / test split (time-ordered — no shuffling) ─────────────────────────

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ── Fit ───────────────────────────────────────────────────────────────────────

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Predict & evaluate ────────────────────────────────────────────────────────

predictions = np.clip(model.predict(X_test).round(), 0, None).astype(int)

print(f'MAE: {mean_absolute_error(y_test, predictions):.2f}')
