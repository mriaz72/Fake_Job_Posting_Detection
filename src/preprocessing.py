
"""
Inference-time preprocessing module.

This module MUST mirror training-time feature engineering.
All features required by feature_order.pkl are recreated here.
"""
import numpy as np
import re
import textstat
import warnings

# Suppress the Sklearn name warning for a cleaner Streamlit UI
warnings.filterwarnings("ignore", category=UserWarning)
def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_text_features(text: str) -> dict:
    words = text.split()
    word_count = len(words)
    unique_word_count = len(set(words))

    try:
        flesch = textstat.flesch_reading_ease(text)
        dale = textstat.dale_chall_readability_score(text)
    except:
        flesch, dale = 50.0, 10.0

    return {
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "lexical_diversity": unique_word_count / max(word_count, 1),
        "flesch_score": flesch,
        "dale_chall_score": dale,
    }

def prepare_features(raw_input: dict, feature_order: list) -> np.ndarray:
    # 1. Text processing
    combined = f"{raw_input.get('title','')} {raw_input.get('description','')} {raw_input.get('requirements','')}"
    text_feats = extract_text_features(clean_text(combined))

    # 2. Map all features (including the missing categorical ones)
    full_feature_dict = {
        **text_feats,
        "telecommuting": int(raw_input.get("telecommuting", 0)),
        "has_company_logo": int(raw_input.get("has_company_logo", 0)),
        "has_questions": int(raw_input.get("has_questions", 0)),
        "has_company_info": int(bool(str(raw_input.get("company_profile", "")).strip())),
        "has_benefits": int(bool(str(raw_input.get("benefits", "")).strip())),
        "salary_explicit": int(bool(str(raw_input.get("salary_range", "")).strip())),
        "log_company_credibility": raw_input.get("log_company_credibility", 0.0),

        # Mapping the missing categorical features
        # (Using .get(key, 0) to prevent the warnings you saw)
        "location_clean_enc": raw_input.get("location_clean_enc", 0),
        "required_experience_clean_enc": raw_input.get("required_experience_clean_enc", 0),
        "employment_type_clean_enc": raw_input.get("employment_type_clean_enc", 0),
        "industry_clean_enc": raw_input.get("industry_clean_enc", 0),
        "function_clean_enc": raw_input.get("function_clean_enc", 0)
    }

    # 3. Enforce order
    feature_vector = [full_feature_dict.get(f, 0.0) for f in feature_order]
    return np.array([feature_vector], dtype=float)
