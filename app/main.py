# app.py
from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="E-Commerce Recommendation System API")

# ---------------------------
# LOAD ARTIFACTS
# ---------------------------
MODEL_PATH = "../models/model.pkl"
PIVOT_COLUMNS_PATH = "../models/pivot_columns.pkl"
USER_ENCODER_PATH = "../models/user_encoder.pkl"
PRODUCT_ENCODER_PATH = "../models/product_encoder.pkl"

# Load model
model = joblib.load(MODEL_PATH)

# Load encoders
pivot_columns = joblib.load(PIVOT_COLUMNS_PATH)       # product list
user_encoder = joblib.load(USER_ENCODER_PATH)         # user list
product_encoder = joblib.load(PRODUCT_ENCODER_PATH)   # product list

# Create product index map
product_index = {p: idx for idx, p in enumerate(pivot_columns)}


# ---------------------------
# HOME ROUTE
# ---------------------------
@app.get("/")
def home():
    return {
        "message": "Recommendation System is running!",
        "endpoints": ["/recommend/{user_id}?top_k=5"]
    }


# ---------------------------
# RECOMMENDATION FUNCTION
# ---------------------------
def recommend_products(user_id: str, top_k: int = 5):
    # Check if user exists
    if user_id not in user_encoder:
        return {"error": f"User '{user_id}' not found in data"}

    # Create empty user vector
    user_vector = np.zeros(len(pivot_columns))

    # Convert user interactions into a vector (cold start handling)
    # (In production you combine recent interactions instead)

    # Transform & reconstruct
    user_latent = model.transform([user_vector])
    reconstructed_scores = model.inverse_transform(user_latent)[0]

    # Sort top K products
    top_indices = np.argsort(reconstructed_scores)[::-1][:top_k]
    top_products = [pivot_columns[i] for i in top_indices]
    top_scores = [float(reconstructed_scores[i]) for i in top_indices]

    # Create list of results
    recommendations = [
        {"product_id": product, "score": score}
        for product, score in zip(top_products, top_scores)
    ]

    return recommendations


# ---------------------------
# FASTAPI ENDPOINT
# ---------------------------
@app.get("/recommend/{user_id}")
def recommend(user_id: str, top_k: int = 5):
    result = recommend_products(user_id, top_k)
    return {
        "user_id": user_id,
        "top_k": top_k,
        "recommendations": result
    }
