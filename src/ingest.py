"""PDF ingestion into Pinecone.

Usage:
    python ingest.py                  # ingest all PDFs in data/documents/
    python ingest.py --file path.pdf  # ingest one PDF
"""

import argparse
import logging
import sys
from pathlib import Path

from extract import extract_text_hybrid
from chunking import build_chunks, build_metadata
from embedding import embed_documents
from vector_store import upsert_vectors, delete_by_source

log = logging.getLogger(__name__)
DOCUMENTS_DIR = Path(__file__).parent.parent / "data" / "documents"


def process_pdf(pdf_path, source_name=None):
    """Process a single PDF: extract, chunk, embed, and upsert to Pinecone."""
    source = source_name or pdf_path.name
    delete_by_source(source)
    
    log.info("Extracting: %s", pdf_path.name)
    text = extract_text_hybrid(pdf_path)
    
    if not text.strip():
        log.warning("No text extracted from %s", pdf_path.name)
        return 0
    
    chunks = build_chunks(text)
    if not chunks:
        log.warning("No chunks for %s — skipping", pdf_path.name)
        return 0
    
    chunk_texts = [c['text'] for c in chunks]
    chunk_ids = [f"{pdf_path.stem}-{c['id']}" for c in chunks]
    chunk_metadatas = [build_metadata(c, source) for c in chunks]

    batch_size = 50
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch_texts = chunk_texts[i:i + batch_size]
        batch_ids = chunk_ids[i:i + batch_size]
        batch_metas = chunk_metadatas[i:i + batch_size]
        batch_vecs = embed_documents(batch_texts)
        for vec, cid, meta in zip(batch_vecs, batch_ids, batch_metas):
            vectors.append({"id": cid, "values": vec, "metadata": meta})
    
    upsert_vectors(vectors)
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
