"""Embedding — Ollama embeddings."""

from langchain_ollama import OllamaEmbeddings

from config import EMBEDDING_MODEL

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


def embed_query(text):
    """Embed a single query."""
    return embeddings.embed_query(text)


def embed_documents(texts):
    """Embed multiple documents."""
    return embeddings.embed_documents(texts)
