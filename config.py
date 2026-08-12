"""
config.py

Central place for all configuration values used across the project.
Keeping settings here means we never hardcode values inside app.py,
rag.py, or ingest.py. If you want to change the LLM, chunk size,
or any other setting, this is the only file you should need to edit.
"""

import os
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# LLM Settings (Groq)
# ---------------------------------------------------------------------------
# Primary model used for answering questions.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Used automatically if the primary model fails or is unavailable.
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")

# Groq's API is OpenAI-compatible, so we just point the base URL at Groq.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Keep answers focused and deterministic.
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 700

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ---------------------------------------------------------------------------
# Vector Database (ChromaDB)
# ---------------------------------------------------------------------------
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
CHROMA_COLLECTION_NAME = "fastapi_docs"

# ---------------------------------------------------------------------------
# Document Ingestion
# ---------------------------------------------------------------------------
DATA_FOLDER = os.getenv("DATA_FOLDER", "data")

# Chunking strategy
# Chunk size = 500 characters keeps enough context for a coherent answer
# without pulling in unrelated paragraphs.
# Overlap = 100 characters prevents sentences from being cut in half
# right at a chunk boundary.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# Top 3 chunks strikes a balance between giving the LLM enough context
# and avoiding irrelevant/noisy information.
TOP_K = 3

# ---------------------------------------------------------------------------
# API Settings
# ---------------------------------------------------------------------------
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
