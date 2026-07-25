"""Retrieve procurement context from MongoDB Atlas Vector Search."""

from __future__ import annotations

import argparse
from typing import Any

from pymongo import MongoClient

from code.config import Settings
from code.ingest import get_embeddings


def retrieve(query: str, limit: int, settings: Settings) -> list[dict[str, Any]]:
    """Embed *query* and return the most relevant stored document chunks."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    # The query must use the same embedding model as the stored chunk vectors.
    query_vector = get_embeddings(settings).embed_query(query)
    pipeline = [
        {
            # Atlas finds chunk vectors that are semantically closest to the question.
            "$vectorSearch": {
                "index": settings.vector_index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": limit * 20,
                "limit": limit,
            }
        },
        {
            # Return answer context, source metadata, and the similarity score only.
            "$project": {
                "_id": 0,
                "text": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    with MongoClient(settings.mongodb_uri) as client:
        collection = client[settings.mongodb_database][settings.mongodb_collection]
        return list(collection.aggregate(pipeline))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question to search for in procurement documents.")
    parser.add_argument("--limit", type=int, default=4, help="Maximum matching chunks to return.")
    args = parser.parse_args()

    results = retrieve(args.query, args.limit, Settings.from_environment())
    for number, result in enumerate(results, start=1):
        source = result["metadata"].get("source", "Unknown source")
        print(f"[{number}] score={result['score']:.4f} source={source}\n{result['text']}\n")


if __name__ == "__main__":
    main()
