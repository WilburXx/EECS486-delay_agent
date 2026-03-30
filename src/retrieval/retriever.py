"""Chunk indexing and retrieval utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.config import AppConfig
from core.schemas import Chunk, DocumentMetadata, RetrievedPassage, SourceType
from retrieval.embedder import OpenAIEmbedder
from retrieval.vector_store import FAISSVectorStore


class PassageRetriever:
    """Index processed chunks and retrieve the most relevant passages."""

    def __init__(
        self,
        embedder: OpenAIEmbedder | None = None,
        vector_store: FAISSVectorStore | None = None,
        config: AppConfig | None = None,
    ) -> None:
        """Initialize retrieval dependencies."""
        self._config = config or AppConfig.from_env()
        self._embedder = embedder or OpenAIEmbedder(config=self._config)
        self._vector_store = vector_store or FAISSVectorStore()

    def index_chunks(
        self,
        chunks_path: Path | None = None,
        persist_directory: Path | None = None,
    ) -> int:
        """Index chunks from a JSONL file into the FAISS vector store."""
        source_path = chunks_path or self._config.data_dir / "processed" / "chunks.jsonl"
        records = self._load_chunk_records(source_path)
        embeddings = self._embedder.embed_texts([record["chunk"]["text"] for record in records])
        self._vector_store.build(embeddings=embeddings, records=records)

        if persist_directory is not None:
            self._vector_store.save(persist_directory)

        return len(records)

    def load_index(self, persist_directory: Path | None = None) -> None:
        """Load a previously persisted FAISS index."""
        directory = persist_directory or self._config.data_dir / "processed" / "vector_store"
        self._vector_store.load(directory)

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_types: list[SourceType] | None = None,
        search_k: int | None = None,
    ) -> list[RetrievedPassage]:
        """Search indexed chunks and return typed retrieved passages."""
        query_embedding = self._embedder.embed_text(query)
        matches = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=search_k or max(top_k, 20),
        )
        if source_types:
            allowed = set(source_types)
            matches = [
                match
                for match in matches
                if DocumentMetadata.model_validate(match["metadata"]).source_type in allowed
            ]
        matches = matches[:top_k]
        return [self._to_retrieved_passage(match, query) for match in matches]

    def _load_chunk_records(self, path: Path) -> list[dict[str, Any]]:
        """Load chunk records from a JSONL file."""
        if not path.exists():
            raise FileNotFoundError(f"Chunk file not found: {path}")

        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        return records

    def _to_retrieved_passage(self, record: dict[str, Any], query: str) -> RetrievedPassage:
        """Convert a vector-store record into a RetrievedPassage model."""
        chunk = Chunk.model_validate(record["chunk"])
        metadata = DocumentMetadata.model_validate(record["metadata"])
        page_numbers = record.get("page_numbers", [])
        citation = self._build_citation(
            document_id=metadata.document_id,
            chunk_id=chunk.chunk_id,
            page_numbers=page_numbers,
        )
        return RetrievedPassage(
            chunk=chunk,
            metadata=metadata,
            relevance_score=float(record["relevance_score"]),
            query=query,
            citation=citation,
            rationale=record.get("rationale"),
        )

    def _build_citation(
        self,
        document_id: str,
        chunk_id: str,
        page_numbers: list[int],
    ) -> str:
        """Format a stable citation string for a retrieved chunk."""
        if page_numbers:
            page_start = min(page_numbers)
            page_end = max(page_numbers)
        else:
            page_start = 0
            page_end = 0

        return f"[{document_id}:{page_start}-{page_end}:{chunk_id}]"
