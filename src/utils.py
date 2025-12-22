"""
Utility helpers.
"""

def combine_raw_text(title="", company="", description="", requirements=""):
    """
    Combine raw job fields for RAG querying with fallbacks.
    """
    # Clean up whitespace and handle None types
    t = str(title or "Unknown Title")
    c = str(company or "Unknown Company")
    d = str(description or "")
    r = str(requirements or "")

    return f"Title: {t}. Company: {c}. Description: {d}. Requirements: {r}."
