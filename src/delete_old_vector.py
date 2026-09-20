"""Delete vectors from Pinecone index.

Usage:
    python delete_old_vector.py --all           # delete all vectors
    python delete_old_vector.py --name NAME     # delete by source name
"""

import argparse
import logging

from vector_store import delete_all, delete_by_source, get_index_stats

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Delete vectors from Pinecone")
    parser.add_argument("--all", action="store_true", help="Delete all vectors")
    parser.add_argument("--name", type=str, help="Delete vectors by source name")
    args = parser.parse_args()

    if args.all:
        delete_all()
        log.info("Deleted all vectors from index")
    elif args.name:
        delete_by_source(args.name)
        log.info("Deleted vectors for source: %s", args.name)
    else:
        stats = get_index_stats()
        log.info("Index stats: %s", stats)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
