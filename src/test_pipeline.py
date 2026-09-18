"""Test script: verify extraction and chunking quality."""

import random
from pathlib import Path

from chunking import build_chunks
from pinecone import Pinecone
from config import INDEX_NAME, PINECONE_API_KEY

DOCUMENTS_DIR = Path("data/documents")
EXTRACTED_DIR = Path("data/extracted")


def check_extracted_files():
    """1. Check extracted files are not empty."""
    print("=" * 60)
    print("1. EXTRACTED FILES CHECK")
    print("=" * 60)

    files = list(EXTRACTED_DIR.glob("*.md"))
    ok = True
    for f in files:
        size = f.stat().st_size
        status = "OK" if size >= 500 else "WARNING (too small)"
        if size < 500:
            ok = False
        print(f"  {f.name}: {size} bytes — {status}")

    return ok


def check_image_tags():
    """2. Check for leftover <!-- image --> tags."""
    print("\n" + "=" * 60)
    print("2. IMAGE TAG CHECK")
    print("=" * 60)

    clean = True
    for f in EXTRACTED_DIR.glob("*.md"):
        content = f.read_text(encoding="utf-8")
        count = content.count("<!-- image -->")
        if count > 0:
            print(f"  {f.name}: FOUND {count} image tags")
            clean = False
        else:
            print(f"  {f.name}: clean")

    return clean


def check_chunks_from_extracted():
    """3. Check chunk sizes from extracted files."""
    print("\n" + "=" * 60)
    print("3. CHUNK SIZE CHECK (from extracted .md files)")
    print("=" * 60)

    all_chunks = []
    for f in sorted(EXTRACTED_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        chunks = build_chunks(text)
        all_chunks.extend(chunks)
        print(f"  {f.name}: {len(chunks)} chunks")

    if not all_chunks:
        print("  No chunks found!")
        return False

    sizes = [len(chunk) for _, chunk in all_chunks]
    print(f"\n  Total chunks: {len(all_chunks)}")
    print(f"  Smallest: {min(sizes)} chars")
    print(f"  Largest: {max(sizes)} chars")
    print(f"  Average: {sum(sizes) // len(sizes)} chars")

    big = [s for s in sizes if s > 1500]
    if big:
        print(f"\n  WARNING: {len(big)} chunks over 1500 chars!")
        return False

    print("  All chunks within 1500 char limit")
    return True


def check_metadata():
    """4. Check first 5 chunks have metadata."""
    print("\n" + "=" * 60)
    print("4. METADATA CHECK (first 5 chunks from Pinecone)")
    print("=" * 60)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    results = index.query(
        vector=[0] * 768,
        top_k=5,
        include_metadata=True,
    )

    ok = True
    for i, match in enumerate(results.get("matches", [])):
        meta = match.get("metadata", {})
        source = meta.get("source", "")
        page = meta.get("page", "")

        has_source = bool(source)
        has_page = bool(page)

        if not has_source or not has_page:
            ok = False

        status = "OK" if (has_source and has_page) else "MISSING"
        print(f"  Chunk {i+1}: source={source}, page={page} — {status}")

    return ok


def show_random_chunks():
    """5. Print 5 random chunks with metadata."""
    print("\n" + "=" * 60)
    print("5. RANDOM CHUNKS PREVIEW")
    print("=" * 60)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    results = index.query(
        vector=[0] * 768,
        top_k=50,
        include_metadata=True,
    )

    matches = results.get("matches", [])
    if not matches:
        print("  No chunks in index!")
        return

    samples = random.sample(matches, min(5, len(matches)))
    for i, match in enumerate(samples):
        meta = match.get("metadata", {})
        text = meta.get("text", "")[:200]
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        section = meta.get("section", "")

        print(f"\n  --- Chunk {i+1} ---")
        print(f"  Source: {source} | Page: {page}")
        print(f"  Section: {section}")
        print(f"  Text: {text}...")


def main():
    print("RAG PIPELINE TEST\n")

    results = {}
    results["extracted_files"] = check_extracted_files()
    results["image_tags"] = check_image_tags()
    results["chunk_sizes"] = check_chunks_from_extracted()
    results["metadata"] = check_metadata()

    show_random_chunks()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test}: {status}")

    all_pass = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
