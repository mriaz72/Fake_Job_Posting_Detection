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


```mermaid
graph TD
    %% --- Style Definitions for a Professional Look ---
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef input fill:#fff8e1,stroke:#ff6f00,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef store fill:#efebe9,stroke:#3e2723,stroke-width:2px,stroke-dasharray: 5 5;
    classDef model fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;

    %% --- Nodes ---
    Start([👤 User Input via Streamlit UI]):::input
    Prep[🔄 Feature Engineering Pipeline]:::process
    
    %% ML Branch
    subgraph "ML Inference Layer (Prediction)"
        Scale[📏 Standard Scaler]:::process
        XGB{{🚀 XGBoost Classifier}}:::model
        ResultScore([📊 Fraud Probability Score]):::output
    end

    %% RAG Branch
    subgraph "RAG Explainability Layer (Evidence)"
        Embed[🧠 Hugging Face Embeddings]:::model
        FAISS[(🗄️ FAISS Vector Index)]:::store
        Retrieval[🔎 Retrieve Similar Legit Jobs]:::process
        Explanation[📃 Generate Contextual Explanation]:::process
    end

    %% Final Output Output
    Dashboard([🖥️ Integrated Dashboard Display]):::output

    %% --- Data Flow Connections ---
    Start -->|1. Raw Text & Metadata| Prep
    
    %% Flow to ML
    Prep -- 2a. Structured Features --> Scale
    Scale -->|Scaled Vector| XGB
    XGB --> ResultScore
    
    %% Flow to RAG
    Prep -- 2b. Job Description Text --> Embed
    Embed -->|Query Vector| Retrieval
    FAISS -.->|Verified Job Embeddings| Retrieval
    Retrieval -->|Top-K Legit Examples| Explanation

    %% Merging Results
    ResultScore -->|3a. Risk Level| Dashboard
    Explanation -->|3b. Reasoning & Evidence| Dashboard
