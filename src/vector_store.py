"""Vector Store — Pinecone operations."""

import logging
from pinecone import Pinecone

from config import INDEX_NAME, PINECONE_API_KEY

log = logging.getLogger(__name__)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


def upsert_vectors(vectors):
    """Upsert vectors to Pinecone."""
    index.upsert(vectors=vectors)
    log.info("Upserted %d vectors", len(vectors))


def query_vectors(vector, top_k=15, include_metadata=True):
    """Query Pinecone for similar vectors."""
    return index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=include_metadata,
    )


def delete_by_source(source_name):
    """Delete all vectors matching a source name."""
    try:
        index.delete(filter={"source": {"$eq": source_name}})
        log.info("Deleted vectors for source: %s", source_name)
    except Exception as e:
        log.warning("Delete failed for %s: %s", source_name, e)


def delete_all():
    """Delete all vectors from index."""
    index.delete(delete_all=True)
    log.info("Deleted all vectors from index")


def get_index_stats():
    """Get index statistics."""
    return index.describe_index_stats()
