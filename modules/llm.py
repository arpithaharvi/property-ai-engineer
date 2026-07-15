# ============================================================
# modules/llm.py — LLM and Embeddings setup
# All model loading in one place
# Used by: rag.py, pdf_loader.py, app.py
# ============================================================

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from groq import Groq
from config import (
    GROQ_API_KEY,
    GROQ_LLM_MODEL,
    EMBEDDING_MODEL,
    LLM_TEMPERATURE
)


@st.cache_resource
def load_embeddings():
    # Downloads once (~90MB), then runs locally forever
    # Converts text to 384 numbers for ChromaDB storage
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def load_llm():
    # Groq LLaMA — free, fast, bank-grade at temperature=0.0
    return ChatGroq(
        model=GROQ_LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY
    )


def load_groq_client():
    # Direct Groq client for Vision API calls
    return Groq(api_key=GROQ_API_KEY)