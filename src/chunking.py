"""Chunking: split text into pieces."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_chunks(text, chunk_size=3000, chunk_overlap=300, min_chars=50):
    """Split text into chunks. Returns list of (heading, chunk) tuples."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for piece in splitter.split_text(text):
        if len(piece.strip()) < min_chars:
            continue
        heading = piece.split("\n")[0].strip()[:100]
        chunks.append((heading, piece))

    return chunks
