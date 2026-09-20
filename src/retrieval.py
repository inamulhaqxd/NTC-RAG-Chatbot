"""Retrieval — hybrid search (vector + BM25), context building."""

import logging
import re
from rank_bm25 import BM25Okapi

from config import SCORE_THRESHOLD
from embedding import embed_query
from vector_store import query_vectors, index

log = logging.getLogger(__name__)

bm25_corpus = []
bm25_ids = []
bm25_model = None
bm25_dirty = True


def reset_bm25_cache():
    """Invalidate BM25 cache so it rebuilds on next search."""
    global bm25_model, bm25_dirty
    bm25_model = None
    bm25_dirty = True
    log.info("BM25 cache invalidated")


def load_bm25_index():
    """Load all chunks from Pinecone into BM25 index (paginated)."""
    global bm25_corpus, bm25_ids, bm25_model, bm25_dirty

    if bm25_model is not None:
        return

    log.info("Loading BM25 index from Pinecone...")
    bm25_corpus = []
    bm25_ids = []

    batch_size = 1000
    next_token = None

    while True:
        kwargs = {"vector": [0] * 768, "top_k": batch_size, "include_metadata": True}
        if next_token:
            kwargs["page_token"] = next_token
        page = index.query(**kwargs)

        for match in page.get("matches", []):
            text = match["metadata"].get("text", "")
            tokens = tokenize(text)
            if tokens:
                bm25_corpus.append(tokens)
                bm25_ids.append(match["id"])

        next_token = page.get("next_page_token")
        if not next_token or not page.get("matches"):
            break

    bm25_model = BM25Okapi(bm25_corpus)
    bm25_dirty = False
    log.info("BM25 index loaded with %d chunks", len(bm25_ids))


def tokenize(text):
    """Simple tokenization for BM25."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()


def hybrid_search(query, top_k=15):
    """Search using both vector and BM25, combine results."""
    query_vector = embed_query(query)
    
    vector_results = query_vectors(query_vector, top_k=top_k)
    
    vector_matches = {
        m['id']: m for m in vector_results["matches"]
        if m["score"] >= SCORE_THRESHOLD
    }
    
    load_bm25_index()
    query_tokens = tokenize(query)
    bm25_scores = bm25_model.get_scores(query_tokens)
    
    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    
    bm25_matches = {}
    for idx in bm25_top_indices:
        if bm25_scores[idx] > 0:
            chunk_id = bm25_ids[idx]
            if chunk_id not in bm25_matches:
                result = index.query(id=chunk_id, top_k=1, include_metadata=True)
                if result["matches"]:
                    bm25_matches[chunk_id] = result["matches"][0]
    
    combined = {}
    combined.update(vector_matches)
    combined.update(bm25_matches)
    
    results = list(combined.values())
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return results[:top_k]


def build_context(matches):
    """Build context from matches with deduplication."""
    context = ""
    seen_sections = set()

    for match in matches:
        meta = match['metadata']
        text = meta['text']
        source = meta.get('source', 'unknown')
        section = meta.get('section', '')
        page = meta.get('page', 0)

        section_key = f"{source}-{section}"
        if section_key in seen_sections:
            continue
        seen_sections.add(section_key)

        context += f"[Source: {source} | Section: {section} | Page: {page}]\n{text}\n\n"

    return context
