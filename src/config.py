import os
import sys

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

EMBEDDING_MODEL = "nomic-embed-text:latest"
INDEX_NAME = "rag-index"
LLM_MODEL = "openai/gpt-oss-20b"
TOP_K = 8
SCORE_THRESHOLD = 0.50

_missing = [
    name for name, val in [("GROQ_API_KEY", GROQ_API_KEY), ("PINECONE_API_KEY", PINECONE_API_KEY)]
    if not val
]
if _missing:
    sys.exit(f"Missing required env var(s): {', '.join(_missing)}. Add them to .env")
