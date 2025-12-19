# 🛡️ FraudShield AI: Fake Job Posting Detector

FraudShield AI is an end-to-end machine learning application designed to detect fraudulent job postings using a hybrid approach of **XGBoost Classification** and **RAG (Retrieval-Augmented Generation)**.

## 🚀 Features
- **Machine Learning Analysis:** Uses an XGBoost model trained on lexical, metadata, and readability features to predict the probability of fraud.
- **Explainable AI (RAG):** If a job is flagged as suspicious, the system retrieves similar verified legitimate jobs from a FAISS vector store to highlight discrepancies.
- **Interactive UI:** A clean Streamlit interface for easy job analysis.

## 🛠️ Tech Stack
- **Language:** Python
- **ML Framework:** XGBoost, Scikit-Learn
- **Vector Database:** FAISS
- **Readability Metrics:** Textstat
- **Embeddings:** HuggingFace Transformers
- **UI:** Streamlit

## 📂 Project Structure
- `src/`: Core logic including preprocessing, inference, and RAG engine.
- `models/`: Pre-trained models and scalers.
- `main.py`: The central pipeline orchestrator.
- `app.py`: The Streamlit dashboard.

## ⚙️ Setup & Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/mriaz72/Fake_Job_Posting_Detection](https://github.com/mriaz72/Fake_Job_Posting_Detection)
