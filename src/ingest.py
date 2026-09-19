"""PDF ingestion into Pinecone.

Usage:
    python ingest.py                  # ingest all PDFs in data/documents/
    python ingest.py --file path.pdf  # ingest one PDF
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pymupdf4llm
import pymupdf
from langchain_ollama import OllamaEmbeddings
from pinecone import Pinecone

from chunking import build_chunks
from config import EMBEDDING_MODEL, INDEX_NAME, PINECONE_API_KEY
from delete_old_vector import delete_by_name

log = logging.getLogger(__name__)
DOCUMENTS_DIR = Path(__file__).parent.parent / "data" / "documents"

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
index = Pinecone(api_key=PINECONE_API_KEY).Index(INDEX_NAME)


def clean_text(text):
    """Remove image tags and extra whitespace."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_scanned(pdf_path):
    """Check if PDF is scanned (image-based)."""
    doc = pymupdf.open(str(pdf_path))
    total = sum(len(p.get_text()) for p in doc)
    doc.close()
    return total < 100


def extract_text(pdf_path):
    """Extract text using PyMuPDF4LLM (fast, good for text PDFs)."""
    return pymupdf4llm.to_markdown(str(pdf_path))


def extract_text_ocr(pdf_path):
    """Extract text using Docling OCR (for scanned PDFs)."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import OcrAutoOptions, OcrMode, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=OcrAutoOptions(mode=OcrMode.PDF_AWARE_LAYOUT_REGIONS),
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    return converter.convert(str(pdf_path)).document.export_to_markdown()


def extract_text_hybrid(pdf_path):
    """Hybrid extraction: PyMuPDF4LLM first, fallback to Docling for scanned."""
    try:
        text = extract_text(pdf_path)
        word_count = len(text.split())
        
        if word_count < 50:
            log.info("Low text count (%d words), trying OCR: %s", word_count, pdf_path.name)
            try:
                ocr_text = extract_text_ocr(pdf_path)
                if len(ocr_text.split()) > word_count:
                    return clean_text(ocr_text)
            except Exception as e:
                log.warning("OCR failed: %s", e)
        
        return clean_text(text)
    except Exception as e:
        log.warning("PyMuPDF failed (%s), trying OCR: %s", e, pdf_path.name)
        return clean_text(extract_text_ocr(pdf_path))


def process_pdf(pdf_path):
    """Process a single PDF: extract, chunk, embed, and upsert to Pinecone."""
    delete_by_name(pdf_path.stem, index)
    
    log.info("Extracting: %s", pdf_path.name)
    text = extract_text_hybrid(pdf_path)
    
    if not text.strip():
        log.warning("No text extracted from %s", pdf_path.name)
        return 0
    
    chunks = build_chunks(text)
    if not chunks:
        log.warning("No chunks for %s — skipping", pdf_path.name)
        return 0
    
    vectors = []
    chunk_texts = []
    chunk_ids = []
    chunk_metadatas = []
    
    for chunk in chunks:
        chunk_texts.append(chunk['text'])
        chunk_ids.append(f"{pdf_path.stem}-{chunk['id']}")
        chunk_metadatas.append({
            "text": chunk['text'],
            "source": pdf_path.name,
            "chunk_id": chunk['id'],
            "section": chunk.get('heading', ''),
            "page": chunk.get('page', 0),
            "level": chunk.get('level', 'chunk'),
            "parent_id": chunk.get('parent_id', '') or '',
            "child_count": len(chunk.get('children', [])),
        })
    
    batch_size = 50
    for i in range(0, len(chunk_texts), batch_size):
        batch_texts = chunk_texts[i:i + batch_size]
        batch_ids = chunk_ids[i:i + batch_size]
        batch_metadatas = chunk_metadatas[i:i + batch_size]
        
        vectors_raw = embeddings.embed_documents(batch_texts)
        
        for vec, cid, meta in zip(vectors_raw, batch_ids, batch_metadatas):
            vectors.append({
                "id": cid,
                "values": vec,
                "metadata": meta,
            })
    
    index.upsert(vectors=vectors)
    log.info("Upserted %d chunks for %s", len(vectors), pdf_path.name)
    return len(vectors)


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Pinecone")
    parser.add_argument("--file", type=Path, help="Path to a single PDF")
    args = parser.parse_args()

    if args.file:
        if not args.file.exists():
            sys.exit(f"File not found: {args.file}")
        pdf_files = [args.file]
    else:
        pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
        if not pdf_files:
            sys.exit(f"No PDFs found in {DOCUMENTS_DIR}")

    log.info("Found %d PDF(s)", len(pdf_files))

    total = 0
    for pdf_path in pdf_files:
        try:
            total += process_pdf(pdf_path)
        except Exception:
            log.exception("Failed: %s", pdf_path.name)

    log.info("Done — %d total chunks", total)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
