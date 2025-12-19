"""
Unified ML + RAG inference pipeline.
"""

from src.inference import predict_fraud
from src.rag_engine import explain_job
from src.utils import combine_raw_text

def run_pipeline(raw_input: dict):
    """
    Full inference pipeline using raw user input only.

    Args:
        raw_input (dict): Raw job posting fields

    Returns:
        dict: Prediction + optional RAG explanation
    """

    # 1. ML prediction (numeric features are created internally)
    prediction = predict_fraud(raw_input)

    result = {
        "prediction": prediction,
        "explanation": None
    }

    # 2. Trigger RAG only if job is suspicious
    if prediction["is_fraud"] == 1:
        raw_text = combine_raw_text(
            raw_input.get("title", ""),
            raw_input.get("company_profile", ""),
            raw_input.get("description", ""),
            raw_input.get("requirements", "")
        )
        result["explanation"] = explain_job(raw_text)

    return result

