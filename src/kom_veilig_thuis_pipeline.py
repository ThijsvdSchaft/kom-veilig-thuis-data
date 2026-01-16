"""
Kom Veilig Thuis – Data-driven UX pipeline
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Data inladen & opschonen
# -----------------------------
df = pd.read_csv("data/sample_incidents.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

df = (
    df.dropna(subset=["timestamp", "lat", "lon"])
      .drop_duplicates(subset=["timestamp", "lat", "lon", "category", "source"])
)

df["category"] = df["category"].astype(str).str.strip().str.lower()

# -----------------------------
# 2. Recency weighting
# -----------------------------
now = pd.Timestamp.utcnow()
cutoff = now - pd.Timedelta(days=30)
df_recent = df[df["timestamp"] >= cutoff].copy()

half_life_days = 7
age_days = (now - df_recent["timestamp"]).dt.total_seconds() / (3600 * 24)
df_recent["w_recency"] = 0.5 ** (age_days / half_life_days)

# -----------------------------
# 3. Aggregatie naar grid-cellen
# -----------------------------
cell_size = 0.0015

df_recent["cell_x"] = np.floor(df_recent["lat"] / cell_size).astype(int)
df_recent["cell_y"] = np.floor(df_recent["lon"] / cell_size).astype(int)

df_recent["severity"] = pd.to_numeric(
    df_recent.get("severity", 1), errors="coerce"
).fillna(1)

cell_risk = (
    df_recent.assign(risk=lambda d: d["severity"] * d["w_recency"])
    .groupby(["cell_x", "cell_y"], as_index=False)
    .agg(
        risk_sum=("risk", "sum"),
        n_reports=("risk", "size"),
        last_seen=("timestamp", "max")
    )
)

# -----------------------------
# 4. Nachtfactor (feature engineering)
# -----------------------------
local_hour = now.tz_convert("Europe/Amsterdam").hour
night_factor = 1.3 if (local_hour >= 22 or local_hour <= 6) else 1.0
cell_risk["risk_score"] = cell_risk["risk_sum"] * night_factor

# -----------------------------
# 5. Mini ML-validatie
# -----------------------------
features = cell_risk[["risk_sum", "n_reports"]]
labels = (cell_risk["risk_score"] > cell_risk["risk_score"].median()).astype(int)

if len(cell_risk) >= 5:
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print("Validatie-accuracy:", round(accuracy, 2))
