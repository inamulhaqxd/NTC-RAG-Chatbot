"""Chunking — heading-aware, recursive, parent-child with metadata."""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_page_number(text):
    """Extract page number from text if present."""
    match = re.search(r'(?:Page|page|PAGE)\s*(\d+)', text)
    return int(match.group(1)) if match else 0


def split_by_headings(text):
    """Primary split: split text by markdown/section headings."""
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
                    'level': 'section'
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
                'level': 'section'
            })
    
    return splits if splits else [{
        'text': text,
        'heading': text.split('\n')[0].strip()[:100],
        'page': 0,
        'level': 'section'
    }]


def split_by_recursive(text, chunk_size=1500, chunk_overlap=150):
    """Secondary split: recursive character splitting within sections."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    
    chunks = []
    for piece in splitter.split_text(text):
        cleaned = piece.strip()
        if len(cleaned) < 50:
            continue
        
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        if sentences:
            heading = sentences[0][:100]
        else:
            heading = cleaned[:100]
        
        page = extract_page_number(cleaned)
        chunks.append({
            'text': cleaned,
            'heading': heading,
            'page': page,
            'level': 'chunk'
        })
    
    return chunks


def build_chunks(text, chunk_size=1500, chunk_overlap=150, min_chars=50):
    """Build chunks with heading-aware primary + recursive secondary splitting."""
    sections = split_by_headings(text)
    
    all_chunks = []
    chunk_counter = 0
    
    for section in sections:
        section_id = f"section-{chunk_counter}"
        chunk_counter += 1
        
        section_chunks = split_by_recursive(
            section['text'], 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        if not section_chunks and len(section['text']) >= min_chars:
            page = extract_page_number(section['text'])
            section_chunks = [{
                'text': section['text'],
                'heading': section['heading'],
                'page': page,
                'level': 'chunk'
            }]
        
        for i, chunk in enumerate(section_chunks):
            if len(chunk['text'].strip()) < min_chars:
                continue
            
            child_id = f"chunk-{chunk_counter}"
            chunk_counter += 1
            
            all_chunks.append({
                'id': child_id,
                'text': chunk['text'],
                'heading': chunk.get('heading', section['heading']),
                'page': chunk.get('page', section['page']),
                'parent_id': section_id,
                'children': [],
                'level': 'chunk',
                'metadata': {
                    'section': section['heading'],
                    'page': chunk.get('page', section['page']),
                    'chunk_type': 'child'
                }
            })
        
        child_ids = [c['id'] for c in all_chunks if c['parent_id'] == section_id]
        
        all_chunks.append({
            'id': section_id,
            'text': section['text'],
            'heading': section['heading'],
            'page': section['page'],
            'parent_id': None,
            'children': child_ids,
            'level': 'section',
            'metadata': {
                'section': section['heading'],
                'page': section['page'],
                'chunk_type': 'parent',
                'child_count': len(child_ids)
            }
        })
    
    return all_chunks
