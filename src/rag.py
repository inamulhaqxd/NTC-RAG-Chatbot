"""RAG query — search Pinecone, send context to LLM, return answer."""

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from pinecone import Pinecone

from config import (
    GROQ_API_KEY,
    INDEX_NAME,
    LLM_MODEL,
    PINECONE_API_KEY,
    SCORE_THRESHOLD,
    TOP_K,
)

llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL)
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
index = Pinecone(api_key=PINECONE_API_KEY).Index(INDEX_NAME)


def get_parent_context(child_match, all_matches):
    """Get parent section context for a child chunk."""
    parent_id = child_match['metadata'].get('parent_id', '')
    if not parent_id:
        return ""
    
    for m in all_matches:
        if m['id'] == parent_id:
            return m['metadata'].get('text', '')
    return ""


def build_context(matches):
    """Build context from matches with parent-child awareness."""
    context = ""
    sources = []
    char_limit = 4000
    seen_sections = set()
    
    for match in matches:
        meta = match['metadata']
        text = meta['text']
        source = meta.get('source', 'unknown')
        section = meta.get('section', '')
        page = meta.get('page', 0)
        level = meta.get('level', 'chunk')
        chunk_type = meta.get('chunk_type', 'child')
        
        section_key = f"{source}-{section}"
        if section_key in seen_sections and level == 'chunk':
            continue
        seen_sections.add(section_key)
        
        if chunk_type == 'parent':
            entry = f"[Source: {source} | Section: {section} | Page: {page}]\n{text}\n\n"
        else:
            entry = f"[Source: {source} | Section: {section} | Page: {page}]\n{text}\n\n"
        
        if len(context) + len(entry) > char_limit:
            break
        
        context += entry
        sources.append(f"{source} — {section} (p.{page})")
    
    return context, sources


def answer_question(question):
    """Search Pinecone and get answer from LLM."""
    query_vector = embeddings.embed_query(question)
    
    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
    )
    
    filtered_matches = [
        m for m in results["matches"]
        if m["score"] >= SCORE_THRESHOLD
    ]
    
    if not filtered_matches:
        filtered_matches = results["matches"][:3]
    
    context, sources = build_context(filtered_matches)
    
    prompt = f"""You are NTC Helper — a friendly assistant for National Telecommunication Corporation employees.

PERSONALITY:
- Warm, helpful, and professional
- Like a knowledgeable colleague who's always happy to help
- Use simple language, avoid jargon

RULES:
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
