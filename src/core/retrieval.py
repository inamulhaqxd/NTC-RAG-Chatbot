"""Retrieval — vector search, reranking, context building."""

import logging
from langchain_pinecone import PineconeRerank

from config import SCORE_THRESHOLD, TOP_K
from core.embedding import embed_query
from core.vector_store import query_vectors

log = logging.getLogger(__name__)

reranker = PineconeRerank(model="bge-reranker-v2-m3", top_n=15)


def search(query, top_k=15):
    """Search Pinecone for similar vectors."""
    query_vector = embed_query(query)
    results = query_vectors(query_vector, top_k=top_k)
    
    matches = [
        m for m in results["matches"]
        if m["score"] >= SCORE_THRESHOLD
    ]
    
    if not matches:
        matches = results["matches"][:3]
    
    return matches


def rerank(query, matches):
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
