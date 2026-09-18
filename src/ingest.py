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
DOCUMENTS_DIR = Path("data/documents")

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
index = Pinecone(api_key=PINECONE_API_KEY).Index(INDEX_NAME)


def clean_text(text):
    """Remove image tags and extra whitespace."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_scanned(pdf_path):
    doc = pymupdf.open(str(pdf_path))
    total = sum(len(p.get_text()) for p in doc)
    doc.close()
    return total < 100


def extract_text(pdf_path):
    return pymupdf4llm.to_markdown(str(pdf_path))


def extract_text_ocr(pdf_path):
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


def process_pdf(pdf_path):
    delete_by_name(pdf_path.stem, index)

    if is_scanned(pdf_path):
        log.info("Scanned — using OCR: %s", pdf_path.name)
        text = clean_text(extract_text_ocr(pdf_path))
    else:
        log.info("Text — using pymupdf4llm: %s", pdf_path.name)
        text = clean_text(extract_text(pdf_path))

    chunks = build_chunks(text)
    if not chunks:
        log.warning("No chunks for %s — skipping", pdf_path.name)
        return 0

    chunk_texts = [t for _, t in chunks]
    vectors_raw = embeddings.embed_documents(chunk_texts)

    vectors = []
    for i, ((heading, chunk), vec) in enumerate(zip(chunks, vectors_raw)):
        vectors.append({
            "id": f"{pdf_path.stem}-chunk-{i}",
            "values": vec,
            "metadata": {
                "text": chunk,
                "source": pdf_path.name,
                "chunk_id": i,
                "section": heading,
            },
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
