import sys
import os

# Ensure the script can find the src directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference import predict_fraud
from src.rag_engine import explain_job
from src.utils import combine_raw_text

def run_job_analysis_pipeline(raw_data):
    """
    Orchestrates the full analysis:
    1. ML Prediction
    2. RAG Explanation (if needed)
    """
    # Step 1: Execute ML Model Prediction
    # This calls preprocessing and model internally
    ml_result = predict_fraud(raw_data)

    # Step 2: If ML result is successful, get RAG explanation
    explanation = None
    if ml_result["status"] == "success":
        # We use our utility to create a rich search query
        search_query = combine_raw_text(
            raw_data.get("title"),
            raw_data.get("company_profile"),
            raw_data.get("description"),
            raw_data.get("requirements")
        )
        explanation = explain_job(search_query)

    return {
        "prediction": ml_result,
        "explanation": explanation
    }

if __name__ == '__main__':
    # Test block for headless validation
    test_data = {
        "title": "Remote Manager",
        "description": "Easy money, no skills.",
        "requirements": "Internet",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0
    }
    print("Running Pipeline Test...")
    print(run_job_analysis_pipeline(test_data))
