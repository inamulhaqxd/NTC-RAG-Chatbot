"""FastAPI backend for NTC Policy Assistant."""

import logging
import os
import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import ALLOWED_ORIGINS, DOCUMENTS_DIR, MAX_UPLOAD_BYTES, TOP_K
from core.ingestion import extract_text_hybrid
from core.chunking import build_chunks
from core.embedding import embed_documents
from core.vector_store import upsert_vectors, delete_by_source
from core.retrieval import search, rerank, build_context
from core.llm import generate, build_prompt

app = FastAPI(title="NTC Policy Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


class UploadResponse(BaseModel):
    message: str
    chunks: int


def answer_question(question):
    """Answer a question using RAG pipeline."""
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
    matches = search(question, top_k=TOP_K)
    matches = rerank(question, matches)
    
    context, sources = build_context(matches)
    prompt = build_prompt(context, question)
    return generate(prompt)


@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Question cannot be empty")
    return AnswerResponse(answer=answer_question(req.question))


@app.post("/upload", response_model=UploadResponse)
def upload_pdf(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only PDF files are allowed")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=DOCUMENTS_DIR, suffix=".pdf", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            total_bytes = 0
            while data := file.file.read(1024 * 1024):
                total_bytes += len(data)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="PDF is too large")
                temp_file.write(data)

        delete_by_source(filename)
        
        log.info("Extracting: %s", filename)
        text = extract_text_hybrid(temp_path)
        
        if not text.strip():
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No text extracted from PDF")
        
        chunks = build_chunks(text)
        if not chunks:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No chunks generated from PDF")
        
        chunk_texts = [c['text'] for c in chunks]
        chunk_ids = [f"{temp_path.stem}-{c['id']}" for c in chunks]
        chunk_metadatas = [{
            "text": c['text'],
            "source": filename,
            "chunk_id": c['id'],
            "section": c.get('heading', ''),
            "page": c.get('page', 0),
        } for c in chunks]
        
        vectors_raw = embed_documents(chunk_texts)
        vectors = [{
            "id": cid,
            "values": vec,
            "metadata": meta,
        } for vec, cid, meta in zip(vectors_raw, chunk_ids, chunk_metadatas)]
        
        upsert_vectors(vectors)
        
        os.replace(temp_path, DOCUMENTS_DIR / filename)
        temp_path = None
        
        return UploadResponse(message=f"Uploaded and indexed {filename}", chunks=len(vectors))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="The PDF could not be processed")
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
        file.file.close()


@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


log = logging.getLogger(__name__)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
