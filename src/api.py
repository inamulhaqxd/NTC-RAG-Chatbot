"""FastAPI backend for NTC Policy Assistant."""

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import ALLOWED_ORIGINS, DOCUMENTS_DIR, MAX_UPLOAD_BYTES, TOP_K
from extract import extract_text_hybrid
from chunking import build_chunks, build_metadata
from embedding import embed_documents
from vector_store import upsert_vectors, delete_by_source
from retrieval import hybrid_search, build_context, reset_bm25_cache
from llm import generate, build_prompt

log = logging.getLogger(__name__)

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


GREETINGS = {"hi", "hello", "hey", "who are you", "who are u", "what are you", "how are you", "assalam", "salam", "good morning", "good evening"}


def is_greeting(question):
    """Check if question is a greeting or general intro."""
    return question.lower().strip().rstrip("?!.") in GREETINGS


@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Question cannot be empty")

    if is_greeting(req.question):
        return AnswerResponse(answer=generate(f"You are NTC Helper — a friendly assistant for NTC employees. Respond warmly and briefly to this greeting.\n\nUser: {req.question}\n\nAnswer:"))

    matches = hybrid_search(req.question, top_k=TOP_K)
    if not matches:
        return AnswerResponse(answer="No relevant information found in the documents.")

    context = build_context(matches)
    prompt = build_prompt(context, req.question)
    return AnswerResponse(answer=generate(prompt))


@app.post("/upload", response_model=UploadResponse)
def upload_pdf(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only PDF files are allowed")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=DOCUMENTS_DIR, suffix=".pdf", delete=False) as tmp:
            temp_path = Path(tmp.name)
            total = 0
            while data := file.file.read(1024 * 1024):
                total += len(data)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="PDF is too large")
                tmp.write(data)

        delete_by_source(filename)
        text = extract_text_hybrid(temp_path)

        if not text.strip():
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No text extracted from PDF")

        chunks = build_chunks(text)
        if not chunks:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No chunks generated from PDF")

        vectors = [
            {"id": f"{temp_path.stem}-{c['id']}", "values": vec, "metadata": build_metadata(c, filename)}
            for c, vec in zip(chunks, embed_documents([c['text'] for c in chunks]))
        ]
        upsert_vectors(vectors)
        reset_bm25_cache()

        temp_path.rename(DOCUMENTS_DIR / filename)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
