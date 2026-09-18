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


def answer_question(question):
    """Search Pinecone and get answer from LLM."""
    query_vector = embeddings.embed_query(question)

    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
    )

    context = ""
    sources = []

    for match in results["matches"]:
        if match["score"] >= SCORE_THRESHOLD:
            meta = match["metadata"]
            text = meta["text"]
            source = meta.get("source", "unknown")
            section = meta.get("section", "")
            page = meta.get("page", 0)
            context += f"[Source: {source} | Section: {section} | Page: {page}]\n{text}\n\n"
            sources.append(f"{source} — {section} (p.{page})")

    prompt = f"""You are NTC Policy Assistant — a chatbot for National Telecommunication Corporation employees.

Your job: Answer questions about NTC policies, rules, regulations, and procedures.

RULES:
1. Answer ONLY from the provided context. Never use outside knowledge.
2. If the context has the answer, give it directly. No maybe, no guessing.
3. Never say "summarized from", "based on documents", "according to the context", or similar phrases. Answer as if you know it directly.
4. Only cite the source if the user asks "where", "source", or "which document".
5. For multi-part questions, answer each part separately.
6. Use bullet points for lists. Use numbered lists for steps/procedures.
7. Be professional but friendly. Talk like a helpful colleague.
8. If you don't know, say: "I don't have this information right now."
9. Never mention: embeddings, vectors, chunks, scores, retrieval, context, documents provided, or internal systems.
10. If the user greets you, respond warmly and ask how you can help.
11. Keep answers concise. Short and accurate beats long and vague.

CONTEXT:
{context}

USER QUESTION:
{question}

YOUR ANSWER:"""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or 'exit'): ")
        if question.lower() == "exit":
            break
        print("\n" + answer_question(question))
