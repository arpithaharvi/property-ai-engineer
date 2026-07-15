# ============================================================
# config.py — Central configuration for Property AI Project
# All settings in one place — change here, applies everywhere
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ── Model Names ──
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")
UPLOADS_PATH = os.path.join(BASE_DIR, "data", "uploads")
REPORTS_PATH = os.path.join(BASE_DIR, "reports")

# ── RAG Settings ──
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K_SINGLE = 5    # chunks fetched for single property
RETRIEVER_K_MERGED = 8    # chunks fetched for merged property

# ── Vision Settings ──
VISION_MAX_TOKENS = 2000
VISION_IMAGE_QUALITY = 95
VISION_PAGE_ZOOM = 2       # 2x zoom for better image quality
DIRECT_TEXT_MIN_CHARS = 100  # below this = scanned page

# ── LLM Settings ──
LLM_TEMPERATURE = 0.0      # deterministic — bank grade consistency