# src/train.py
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import yaml
import mlflow
import numpy as np

FEATURES_PATH = "data/features/features.csv"
MODEL_PATH = "models/model.pkl"
PARAMS_PATH = "params.yaml"

def train_model():

    # Load params
    params = yaml.safe_load(open(PARAMS_PATH))["train"]

    # Load features
    df = pd.read_csv(FEATURES_PATH)

    # Pivot user-item matrix
    pivot = df.pivot_table(
        index="user_id",
        columns="product_id",
        values="score",
        fill_value=0
    )

    Path("models").mkdir(parents=True, exist_ok=True)

    # Save encoders
    joblib.dump(list(pivot.columns), "models/pivot_columns.pkl")
    joblib.dump(pivot, "models/train_matrix.pkl")

    # Split matrix for evaluation
    X_train, X_test = train_test_split(
        pivot,
        test_size=params["test_size"],
        random_state=params["random_state"]
    )

    # MLflow LOCAL SETUP
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Ecommerce-Recsys")

    best_score = float("inf")
    best_model = None
    best_k = None

    # Candidate components
    k_values = [2, 3, 5, 7, 9]

    # Manual tuning loop
    with mlflow.start_run():

        for k in k_values:

            if k >= X_train.shape[1]:
                continue

            model = TruncatedSVD(n_components=k)

            model.fit(X_train)

            # Reconstruction (approx)
            reconstructed = model.inverse_transform(model.transform(X_test))

            mse = ((X_test.values - reconstructed) ** 2).mean()

            # Log to MLflow
            mlflow.log_metric(f"mse_k_{k}", mse)

            print(f"K={k} → MSE={mse}")

            if mse < best_score:
                best_score = mse
                best_model = model
                best_k = k

        mlflow.log_param("best_k", best_k)
        mlflow.log_metric("best_mse", best_score)

        # Save BEST model
        joblib.dump(best_model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)

        print(f"\nBEST K = {best_k}, BEST MSE = {best_score}")
        print("Model saved → models/model.pkl")

if __name__ == "__main__":
    train_model()
