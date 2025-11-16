# src/evaluate.py
import pandas as pd
import joblib
import json
from pathlib import Path

FEATURES_PATH = "data/features/features.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "metrics/metrics.json"

def evaluate_model():
    Path(METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)

    pivot = df.pivot_table(index="user_id", columns="product_id", values="score", fill_value=0)

    # reconstruct approx matrix and compute simple MSE
    transformed = model.transform(pivot)
    reconstructed = model.inverse_transform(transformed)
    mse = ((pivot.values - reconstructed) ** 2).mean()

    metrics = {"mse": float(mse)}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Metrics saved →", METRICS_PATH)
    print("Metrics:", metrics)

if __name__ == "__main__":
    evaluate_model()
