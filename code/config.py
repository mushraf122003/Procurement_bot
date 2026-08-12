"""Environment-backed settings shared by the pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# Resolve relative data paths from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str
    mongodb_database: str
    mongodb_collection: str
    vector_index_name: str
    embedding_model: str
    data_dir: Path
    chunk_size: int
    chunk_overlap: int
    mysql_uri: str
    mysql_purchase_order_tables: str
    mysql_invoice_tables: str

    @classmethod
    def from_environment(cls) -> "Settings":
        # Keep Atlas credentials outside source control and fail before doing any work.
        mongodb_uri = os.environ.get("MONGODB_URI", "")
        if not mongodb_uri:
            raise ValueError("MONGODB_URI must be set before running this pipeline.")

        return cls(
            mongodb_uri=mongodb_uri,
            mongodb_database=os.environ.get("MONGODB_DATABASE", "procurement_bot"),
            mongodb_collection=os.environ.get("MONGODB_COLLECTION", "documents"),
            vector_index_name=os.environ.get("MONGODB_VECTOR_INDEX", "vector_index"),
            embedding_model=os.environ.get(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            data_dir=Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data")),
            chunk_size=int(os.environ.get("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "150")),
            mysql_uri=os.environ.get("MYSQL_URI", ""),
            mysql_purchase_order_tables=os.environ.get("MYSQL_PURCHASE_ORDER_TABLES", ""),
            mysql_invoice_tables=os.environ.get("MYSQL_INVOICE_TABLES", ""),
        )
