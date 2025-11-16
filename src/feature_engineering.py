# src/feature_engineering.py
import pandas as pd
import joblib
from pathlib import Path

PROCESSED_PATH = "data/processed/data_processed.csv"
FEATURES_PATH = "data/features/features.csv"

def create_features():
    Path(FEATURES_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_PATH)

    # implicit score
    mapping = {"view": 1, "cart": 2, "purchase": 3}
    df["score"] = df["event_type"].map(mapping).fillna(1)

    # aggregated features
    features = df.groupby(["user_id", "product_id"], as_index=False)["score"].sum()
    features.to_csv(FEATURES_PATH, index=False)

    # Save encoders
    user_encoder = features["user_id"].unique().tolist()
    product_encoder = features["product_id"].unique().tolist()

    joblib.dump(user_encoder, "models/user_encoder.pkl")
    joblib.dump(product_encoder, "models/product_encoder.pkl")

    print("Features saved:", FEATURES_PATH)
    print("User/Product encoders saved in models/")

if __name__ == "__main__":
    create_features()
