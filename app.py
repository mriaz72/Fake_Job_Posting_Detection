import streamlit as st
import sys
import os

# Ensure the app can find the main logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import run_job_analysis_pipeline

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️")

st.title("🛡️ Job Posting Fraud Detector")
st.markdown("Enter the job details below to run a deep analysis.")

# --- FORM START ---
with st.form("job_analysis_form"):
    title = st.text_input("Job Title")
    company = st.text_input("Company Name")
    description = st.text_area("Description", height=200)
    requirements = st.text_area("Requirements", height=100)

    st.markdown("### Metadata Flags")
    col1, col2 = st.columns(2)
    with col1:
        logo = st.checkbox("Has Company Logo", value=True)
    with col2:
        remote = st.checkbox("Telecommuting")

    # IMPORTANT: The submit button MUST be inside the 'with' block
    submitted = st.form_submit_button("🚀 Run Analysis")
# --- FORM END ---

if submitted:
    if not title or not description:
        st.warning("⚠️ Please provide at least a Title and Description.")
    else:
        raw_data = {
            "title": title,
            "description": description,
            "requirements": requirements,
            "company_profile": company,
            "has_company_logo": int(logo),
            "telecommuting": int(remote),
            "has_questions": 0,
            "log_company_credibility": 1.0
        }

        with st.spinner("🕵️ Analyzing Patterns..."):
            results = run_job_analysis_pipeline(raw_data)
            pred = results["prediction"]

            if pred["status"] == "success":
                if pred["is_fraud"] == 1:
                    st.error(f"🚩 HIGH RISK DETECTED ({pred['confidence']*100:.1f}%)")
                    st.subheader("Analysis Explanation")
                    st.info(results["explanation"])
                else:
                    st.success(f"✅ LOW RISK ({100 - pred['confidence']*100:.1f}% Legitimate)")
            else:
                st.error(f"Error: {pred.get('message')}")
