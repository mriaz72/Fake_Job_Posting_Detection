"""
ML inference module with robust error handling and debugging.
"""

from src.preprocessing import prepare_features
import joblib
import numpy as np
import os

# Define paths relative to this file or absolute
BASE_DIR = "/content/drive/MyDrive/Fake_Job_Posting_Detection"
MODEL_PATH = os.path.join(BASE_DIR, "models/xgb_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, "models/feature_order.pkl")

# Global variables for model assets
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_order = joblib.load(FEATURE_ORDER_PATH)

def predict_fraud(raw_input: dict, threshold=0.5):
    """
    Predict whether a job posting is fraudulent with feature validation.
    """
    try:
        # 1. Prepare Features
        X = prepare_features(raw_input, feature_order)
        
        # 2. Scale Features
        X_scaled = scaler.transform(X)
        
        # 3. Predict
        prob = model.predict_proba(X_scaled)[0, 1]
        label = int(prob >= threshold)
        
        return {
            "is_fraud": label,
            "confidence": round(float(prob), 3),
            "status": "success"
        }
        
    except KeyError as e:
        # This catch helps us identify exactly which key is missing
        return {
            "status": "error",
            "message": f"Feature Mismatch: {str(e)}",
            "expected_features": feature_order
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Runtime Error: {str(e)}"
        }
