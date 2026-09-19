"""Retrieval — vector search, reranking, context building."""

import logging
import re
from langchain_pinecone import PineconeRerank

from config import SCORE_THRESHOLD, TOP_K
from core.embedding import embed_query
from core.vector_store import query_vectors, index
from core.llm import generate

log = logging.getLogger(__name__)

reranker = PineconeRerank(model="bge-reranker-v2-m3", top_n=15)


EXPAND_PROMPT = """Generate 3 search queries for this question. Return ONLY the queries, one per line, no numbering.

Question: {question}

Queries:"""


def expand_query(question):
    """Generate multiple queries from one question."""
    prompt = EXPAND_PROMPT.format(question=question)
    response = generate(prompt)
    queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
    return [question] + queries[:3]


def keyword_search(query, top_k=15):
    """Search by keyword in metadata."""
    keywords = re.findall(r'\w+', query.lower())
    
    results = index.query(vector=[0] * 768, top_k=1000, include_metadata=True)
    
    scored = []
    for m in results["matches"]:
        text = m["metadata"].get("text", "").lower()
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            m["keyword_score"] = score
            scored.append(m)
    
    scored.sort(key=lambda x: x.get("keyword_score", 0), reverse=True)
    return scored[:top_k]


def search(query, top_k=15):
    """Search Pinecone for similar vectors."""
    queries = expand_query(query)
    
    all_matches = {}
    for q in queries:
        query_vector = embed_query(q)
        results = query_vectors(query_vector, top_k=top_k)
        
        for m in results["matches"]:
            if m["score"] >= SCORE_THRESHOLD:
                if m['id'] not in all_matches or m['score'] > all_matches[m['id']]['score']:
                    all_matches[m['id']] = m
    
    keyword_results = keyword_search(query, top_k=top_k)
    for m in keyword_results:
        if m['id'] not in all_matches:
            m['score'] = 0.5
            all_matches[m['id']] = m
    
    matches = list(all_matches.values())
    matches.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    if not matches:
        query_vector = embed_query(query)
        results = query_vectors(query_vector, top_k=3)
        matches = results["matches"]
    
    return matches[:top_k]


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
