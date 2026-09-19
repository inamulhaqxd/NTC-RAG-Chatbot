"""RAG query — hybrid search (vector + BM25), rerank, send context to LLM."""

import logging
import re
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeRerank
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from config import (
    GROQ_API_KEY,
    INDEX_NAME,
    LLM_MODEL,
    PINECONE_API_KEY,
    SCORE_THRESHOLD,
    TOP_K,
)

log = logging.getLogger(__name__)

llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL)
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
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


def build_context(matches):
    """Build context from matches with deduplication."""
    context = ""
    sources = []
    char_limit = 6000
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


def hybrid_search(query, top_k=15):
    """Search using both vector and BM25, combine results."""
    query_vector = embeddings.embed_query(query)
    
    vector_results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )
    
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


def answer_question(question):
    """Search Pinecone, rerank, and get answer from LLM."""
    import re
    
    questions = re.split(r'(?<=[.?])\s+', question)
    questions = [q.strip() for q in questions if q.strip()]
    
    if len(questions) > 1:
        answers = []
        for q in questions:
            answer = _answer_single(q)
            answers.append(f"**{q}**\n{answer}")
        return "\n\n".join(answers)
    
    return _answer_single(question)


def _answer_single(question):
    """Answer a single question."""
    matches = hybrid_search(question, top_k=TOP_K)
    
    if not matches:
        matches = index.query(
            vector=embeddings.embed_query(question),
            top_k=3,
            include_metadata=True,
        )["matches"]
    
    matches = rerank_matches(question, matches)
    
    context, sources = build_context(matches)
    
    prompt = f"""You are NTC Helper — a friendly assistant for National Telecommunication Corporation employees.

PERSONALITY:
- Warm, helpful, and professional
- Like a knowledgeable colleague who's always happy to help
- Use simple language, avoid jargon

MULTI-QUESTION RULES:
- If the user asks 2+ different questions, answer EACH question
- If one answer is available but another is not, still answer the one you can
- Never skip a question or combine answers

GENERAL RULES:
1. Answer ONLY from the provided context — never make things up
2. If the answer is in the context, give it confidently
3. If the answer is NOT in the context, say: "I don't have that information. Please contact HR/your department for this."
4. Never guess or assume — it's okay to say "I don't know"
5. Use markdown formatting (bullet points, numbered lists, bold for key terms)
6. For multi-part questions, address each part clearly
7. If the user asks in Urdu/English, respond in the same language
8. Never mention: context, documents, embeddings, vectors, chunks, or system internals
9. Be concise but complete — don't leave out important details
10. If greeted, respond warmly and ask how you can help

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or 'exit'): ")
        if question.lower() == "exit":
            break
        print("\n" + answer_question(question))
