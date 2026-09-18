"""Delete old vectors from Pinecone.

Usage:
    python delete.py --name Civil-Servants   # delete chunks matching a name
    python delete.py --all                   # delete everything in the index
"""

import argparse
import logging
import sys

from pinecone import Pinecone
from config import INDEX_NAME, PINECONE_API_KEY

log = logging.getLogger(__name__)


def delete_by_name(name, index):
    """Delete all chunks where ID starts with the given name."""
    try:
        results = index.query(
            vector=[0] * 768,
            top_k=10_000,
            include_metadata=False,
        )
        ids = [
            m["id"] for m in results.get("matches", [])
            if m["id"].startswith(name)
        ]
        if ids:
            index.delete(ids=ids)
            log.info("Deleted %d chunks matching '%s'", len(ids), name)
        else:
            log.info("No chunks found matching '%s'", name)
    except Exception as e:
        log.error("Failed to delete: %s", e)


def delete_all(index):
    """Delete everything in the index."""
    try:
        index.delete(delete_all=True)
        log.info("Deleted all vectors from index")
    except Exception as e:
        log.error("Failed to delete all: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Delete vectors from Pinecone")
    parser.add_argument("--name", help="Delete chunks matching this source name")
    parser.add_argument("--all", action="store_true", help="Delete everything")
    args = parser.parse_args()

    if not args.name and not args.all:
        sys.exit("Use --name <pdf_name> or --all")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    if args.all:
        delete_all(index)
    else:
        delete_by_name(args.name, index)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
