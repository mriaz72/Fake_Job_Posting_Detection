"""
RAG explanation engine with relative paths.
"""
import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. SET RELATIVE PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAG_STORE_PATH = os.path.join(BASE_DIR, "rag_store", "faiss_index")
CONFIG_PATH = os.path.join(BASE_DIR, "rag_store", "embedding_config.json")

# 2. LOAD CONFIG AND STORE
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

embedding_function = HuggingFaceEmbeddings(
    model_name=config["embedding_model"]
)

# Load FAISS using the relative path
vector_store = FAISS.load_local(
    RAG_STORE_PATH,
    embedding_function,
    allow_dangerous_deserialization=True
)

def explain_job(job_text, top_k=3):
    docs = vector_store.similarity_search(job_text, k=top_k)
    explanations = []
    for i, doc in enumerate(docs):
        snippet = doc.page_content[:160].replace("\n", " ")
        explanations.append(f"{i+1}. {snippet}...")

    return (
        "This job posting appears suspicious when compared to legitimate postings.\n\n"
        "Most similar verified job examples:\n"
        + "\n".join(explanations)
    )
