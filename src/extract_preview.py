"""Preview chunks from a PDF before ingesting."""

import pymupdf
from pathlib import Path
from chunking import build_chunks

PDF_PATH = Path("data/documents/ISD-CC-Policy.pdf")


def main():
    doc = pymupdf.open(str(PDF_PATH))
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    print(f"PDF: {PDF_PATH.name}")
    print(f"Characters: {len(text)}")
    print("=" * 60)

    chunks = build_chunks(text)
    print(f"Chunks: {len(chunks)}")
    print("=" * 60)

    for i, (heading, chunk) in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Section: {heading or '(none)'}")
        print(f"Length: {len(chunk)} chars")
        print(f"Text: {chunk[:200]}...")


if __name__ == "__main__":
    main()
