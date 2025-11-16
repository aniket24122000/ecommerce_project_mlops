# src/preprocess.py
import pandas as pd
from pathlib import Path

RAW_PATH = "data/raw/data.csv"
PROCESSED_PATH = "data/processed/data_processed.csv"

def preprocess_data():
    Path(PROCESSED_PATH).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_PATH)

    # basic cleaning
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    else:
        df["rating"] = 0

    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    else:
        df["quantity"] = 0

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    # normalize event_type
    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].str.lower().fillna("view")
    else:
        df["event_type"] = "view"

    df.to_csv(PROCESSED_PATH, index=False)
    print("Preprocessed →", PROCESSED_PATH)

if __name__ == "__main__":
    preprocess_data()
