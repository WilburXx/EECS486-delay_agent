"""FAISS-backed local vector store for retrieved passages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FAISSVectorStore:
    """Persist and query chunk embeddings using a local FAISS index."""

    def __init__(self, index: faiss.IndexFlatIP | None = None) -> None:
        """Initialize the vector store with an optional prebuilt index."""
        self._index = index
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> list[dict[str, Any]]:
        """Return the indexed records in insertion order."""
        return self._records

    def build(self, embeddings: list[list[float]], records: list[dict[str, Any]]) -> None:
        """Create a new FAISS index from embeddings and associated records."""
        if not embeddings:
            self._index = None
            self._records = []
            return

        matrix = self._normalize(np.asarray(embeddings, dtype="float32"))
        dimension = int(matrix.shape[1])
        index = faiss.IndexFlatIP(dimension)
        index.add(matrix)

        self._index = index
        self._records = records

    def save(self, directory: Path) -> None:
        """Persist the FAISS index and record metadata to disk."""
        if self._index is None:
            raise ValueError("Cannot save an empty vector store.")

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "chunks.index"))
        (directory / "chunks.records.json").write_text(
            json.dumps(self._records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, directory: Path) -> None:
        """Load the FAISS index and record metadata from disk."""
        index_path = directory / "chunks.index"
        records_path = directory / "chunks.records.json"

        if not index_path.exists() or not records_path.exists():
            raise FileNotFoundError("Vector store files were not found.")

        self._index = faiss.read_index(str(index_path))
        self._records = json.loads(records_path.read_text(encoding="utf-8"))

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top matching records for a query embedding."""
        if self._index is None or not self._records:
            return []

        query_matrix = self._normalize(np.asarray([query_embedding], dtype="float32"))
        distances, indices = self._index.search(query_matrix, top_k)

        matches: list[dict[str, Any]] = []
        for score, index_value in zip(distances[0], indices[0], strict=False):
            if index_value < 0:
                continue

            record = dict(self._records[int(index_value)])
            record["relevance_score"] = float(score)
            matches.append(record)

        return matches

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for cosine-like similarity search."""
        faiss.normalize_L2(matrix)
        return matrix
