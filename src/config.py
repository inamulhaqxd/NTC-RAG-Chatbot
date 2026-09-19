import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

EMBEDDING_MODEL = "nomic-embed-text:latest"
INDEX_NAME = "rag-index"
LLM_MODEL = "openai/gpt-oss-20b"
TOP_K = 8
SCORE_THRESHOLD = 0.50

BASE_DIR = Path(__file__).parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
ALLOWED_ORIGINS = ["http://localhost:8000", "http://127.0.0.1:8000"]
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

_missing = [
    name for name, val in [("GROQ_API_KEY", GROQ_API_KEY), ("PINECONE_API_KEY", PINECONE_API_KEY)]
    if not val
]
if _missing:
    sys.exit(f"Missing required env var(s): {', '.join(_missing)}. Add them to .env")
