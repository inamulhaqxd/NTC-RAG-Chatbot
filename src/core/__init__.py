"""Core modules for NTC Policy Assistant."""

from core.ingestion import extract_text_hybrid
from core.chunking import build_chunks
from core.embedding import embed_query, embed_documents
from core.vector_store import upsert_vectors, query_vectors, delete_by_source, delete_all
from core.retrieval import search, rerank, build_context
from core.llm import generate, build_prompt

__all__ = [
    "extract_text_hybrid",
    "build_chunks",
    "embed_query",
    "embed_documents",
    "upsert_vectors",
    "query_vectors",
    "delete_by_source",
    "delete_all",
    "search",
    "rerank",
    "build_context",
    "generate",
    "build_prompt",
]
