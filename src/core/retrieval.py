"""Retrieval — hybrid search (vector + BM25), reranking, context building."""

import logging
import re
from langchain_pinecone import PineconeRerank
from rank_bm25 import BM25Okapi

from config import SCORE_THRESHOLD, TOP_K
from core.embedding import embed_query, embeddings
from core.vector_store import query_vectors, index

log = logging.getLogger(__name__)

reranker = PineconeRerank(model="bge-reranker-v2-m3", top_n=30)

bm25_corpus = []
bm25_ids = []
bm25_model = None


def load_bm25_index():
    """Load all chunks from Pinecone into BM25 index."""
    global bm25_corpus, bm25_ids, bm25_model
    
    if bm25_model is not None:
        return
    
    log.info("Loading BM25 index from Pinecone...")
    all_vectors = index.query(vector=[0] * 768, top_k=10000, include_metadata=True)
    
    bm25_corpus = []
    bm25_ids = []
    
    for match in all_vectors["matches"]:
        text = match["metadata"].get("text", "")
        tokens = tokenize(text)
        if tokens:
            bm25_corpus.append(tokens)
            bm25_ids.append(match["id"])
    
    bm25_model = BM25Okapi(bm25_corpus)
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
            vector_match = vector_results["matches"]
            full_match = next((m for m in vector_match if m['id'] == chunk_id), None)
            if full_match:
                bm25_matches[chunk_id] = full_match
    
    combined = {}
    combined.update(vector_matches)
    combined.update(bm25_matches)
    
    results = list(combined.values())
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return results[:top_k]


def rerank_matches(query, matches):
    """Rerank matches using Pinecone reranker."""
    if not matches:
        return matches
    
    documents = [
        {"id": m['id'], "text": m['metadata'].get('text', '')}
        for m in matches
    ]
    
    try:
        result = reranker.rerank(query=query, documents=documents)
        
        reranked = []
        for ranked in result:
            for m in matches:
                if m['id'] == ranked['id']:
                    m['rerank_score'] = ranked.get('score', 0)
                    reranked.append(m)
                    break
        
        reranked.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        return reranked
    except Exception as e:
        log.warning("Reranking failed, using original order: %s", e)
        return matches


def build_context(matches, char_limit=6000):
    """Build context from matches with deduplication."""
    context = ""
    sources = []
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
        
        entry = f"[Source: {source} | Section: {section} | Page: {page}]\n{text}\n\n"
        
        if len(context) + len(entry) > char_limit:
            break
        
        context += entry
        sources.append(f"{source} — {section} (p.{page})")
    
    return context, sources
