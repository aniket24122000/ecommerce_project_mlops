# src/data_ingest.py
import pandas as pd
from pathlib import Path

INPUT_PATH = "ecom_recsys_sample.csv"
OUTPUT_PATH = "data/raw/data.csv"

def ingest_data():
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Data copied →", OUTPUT_PATH)

# auto-run
if __name__ == "__main__":
    ingest_data()
