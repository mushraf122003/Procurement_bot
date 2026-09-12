"""Load local procurement documents, embed chunks, and store them in Atlas."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

# pyrefly: ignore [missing-import]
from scr_code.config import Settings


LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls"}


def load_documents(data_dir: Path) -> list[Document]:
    """Read every supported file below *data_dir* using LangChain loaders."""
    # The local data directory is the source of procurement files for this run.
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    documents: list[Document] = []
    for file_path in sorted(data_dir.rglob("*")):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        # Select the LangChain loader that understands the current file type.
        loader = PyPDFLoader(str(file_path)) if file_path.suffix.lower() == ".pdf" else UnstructuredExcelLoader(str(file_path))
        loaded_documents = loader.load()
        for document in loaded_documents:
            # Keep origin information so users can trace a retrieved answer to its file.
            document.metadata["source"] = str(file_path.resolve())
            document.metadata["file_name"] = file_path.name
            document.metadata["file_type"] = file_path.suffix.lower()
        documents.extend(loaded_documents)
        LOGGER.info("Loaded %s document page(s)/sheet(s) from %s", len(loaded_documents), file_path.name)

    if not documents:
        raise ValueError(f"No PDF, XLS, or XLSX files found in {data_dir}")
    return documents


def split_documents(documents: Iterable[Document], settings: Settings) -> list[Document]:
    # Overlap prevents important sentences from being split across chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(list(documents))
    for chunk_number, chunk in enumerate(chunks):
        # The chunk position supports deterministic IDs during repeat ingestion.
        chunk.metadata["chunk_number"] = chunk_number
    return chunks


def get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    # Normalization matches the cosine-similarity index configured in Atlas.
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ensure_vector_index(collection: Collection, settings: Settings, dimensions: int) -> None:
    """Create the Atlas vector index if it is missing (Atlas-only operation)."""
    existing_names = {index["name"] for index in collection.list_search_indexes()}
    if settings.vector_index_name in existing_names:
        return

    # Atlas indexes this vector field for nearest-neighbor semantic search.
    model = SearchIndexModel(
        name=settings.vector_index_name,
        type="vectorSearch",
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": dimensions,
                    "similarity": "cosine",
                }
            ]
        },
    )
    collection.create_search_index(model=model)
    LOGGER.info("Created Atlas vector search index '%s'.", settings.vector_index_name)


def upsert_chunks(collection: Collection, chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> int:
    # Batch embedding is faster than making one model request per chunk.
    vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    operations: list[UpdateOne] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        # This hash makes repeated ingestion update existing chunks rather than duplicate them.
        identity = f"{chunk.metadata['source']}:{chunk.metadata['chunk_number']}:{chunk.page_content}"
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        operations.append(
            UpdateOne(
                {"_id": chunk_id},
                {
                    "$set": {
                        "text": chunk.page_content,
                        "embedding": vector,
                        "metadata": dict(chunk.metadata),
                    }
                },
                upsert=True,
            )
        )

    result = collection.bulk_write(operations, ordered=False)
    return result.upserted_count + result.modified_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-index", action="store_true", help="Do not create the Atlas vector index.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = Settings.from_environment()
    # Perform the processing stages in order: load, split, embed/store, then index.
    documents = load_documents(settings.data_dir)
    chunks = split_documents(documents, settings)
    embeddings = get_embeddings(settings)

    with MongoClient(settings.mongodb_uri) as client:
        collection = client[settings.mongodb_database][settings.mongodb_collection]
        changed_count = upsert_chunks(collection, chunks, embeddings)
        if not args.skip_index:
            # Read dimensions from the model so the index always matches its vectors.
            dimensions = len(embeddings.embed_query("vector dimension check"))
            ensure_vector_index(collection, settings, dimensions)

    LOGGER.info("Ingestion complete: %s chunks processed.", changed_count)


if __name__ == "__main__":
    main()
