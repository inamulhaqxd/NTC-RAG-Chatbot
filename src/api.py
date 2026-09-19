"""FastAPI backend for NTC Policy Assistant."""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import ALLOWED_ORIGINS, DOCUMENTS_DIR, MAX_UPLOAD_BYTES
from rag import answer_question

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

        from ingest import process_pdf
        chunks = process_pdf(temp_path, source_name=filename)
        os.replace(temp_path, DOCUMENTS_DIR / filename)
        temp_path = None
        return UploadResponse(message=f"Uploaded and indexed {filename}", chunks=chunks)
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
