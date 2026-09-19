"""Chunking — recursive text splitting."""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_chunks(text, chunk_size=1500, chunk_overlap=150, min_chars=50):
    """Build chunks using recursive character splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    
    chunks = []
    chunk_counter = 0
    
    for piece in splitter.split_text(text):
        cleaned = piece.strip()
        if len(cleaned) < min_chars:
            continue
        
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        heading = sentences[0][:100] if sentences else cleaned[:100]
        
        page_match = re.search(r'(?:Page|page|PAGE)\s*(\d+)', cleaned)
        page = int(page_match.group(1)) if page_match else 0
        
        chunk_id = f"chunk-{chunk_counter}"
        chunk_counter += 1
        
        chunks.append({
            'id': chunk_id,
            'text': cleaned,
            'heading': heading,
            'page': page,
        })
    
    return chunks
