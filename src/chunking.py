"""Chunking — heading-aware recursive splitting."""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_page_number(text):
    """Extract page number from text if present."""
    match = re.search(r'(?:Page|page|PAGE)\s*(\d+)', text)
    return int(match.group(1)) if match else 0


def split_by_headings(text):
    """Split text by markdown/section headings."""
    heading_pattern = re.compile(
        r'^(#{1,6}\s+.+|'
        r'\d+\.\s+.+|'
        r'[A-Z][A-Z\s]{3,}:?\s*$|'
        r'CHAPTER[-\s]+.+|'
        r'SECTION[-\s]+.+|'
        r'Rule\s+\d+.*|'
        r'Regulation\s+\d+.*)$',
        re.MULTILINE
    )

    splits = []
    last_end = 0

    for match in heading_pattern.finditer(text):
        start = match.start()
        if start > last_end:
            chunk = text[last_end:start].strip()
            if chunk:
                heading = chunk.split('\n')[0].strip()[:100]
                page = extract_page_number(chunk)
                splits.append({
                    'text': chunk,
                    'heading': heading,
                    'page': page,
                })
        last_end = start

    if last_end < len(text):
        chunk = text[last_end:].strip()
        if chunk:
            heading = chunk.split('\n')[0].strip()[:100]
            page = extract_page_number(chunk)
            splits.append({
                'text': chunk,
                'heading': heading,
                'page': page,
            })

    return splits if splits else [{
        'text': text,
        'heading': text.split('\n')[0].strip()[:100],
        'page': 0,
    }]


def build_chunks(text, chunk_size=1500, chunk_overlap=150, min_chars=50):
    """Build chunks: heading-based sections split recursively."""
    sections = split_by_headings(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )

    all_chunks = []
    chunk_counter = 0

    for section in sections:
        pieces = splitter.split_text(section['text'])

        if not pieces and len(section['text']) >= min_chars:
            pieces = [section['text']]

        for piece in pieces:
            cleaned = piece.strip()
            if len(cleaned) < min_chars:
                continue

            sentences = re.split(r'(?<=[.!?])\s+', cleaned)
            heading = sentences[0][:100] if sentences else cleaned[:100]
            page = extract_page_number(cleaned)

            all_chunks.append({
                'id': f"chunk-{chunk_counter}",
                'text': cleaned,
                'heading': heading,
                'page': page,
                'section_heading': section['heading'],
            })
            chunk_counter += 1

    return all_chunks


def build_metadata(chunk, source):
    """Build Pinecone metadata from a chunk."""
    return {
        "text": chunk['text'],
        "source": source,
        "chunk_id": chunk['id'],
        "section": chunk.get('section_heading', ''),
        "heading": chunk.get('heading', ''),
        "page": chunk.get('page', 0),
    }
