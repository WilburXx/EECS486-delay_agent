"""Ingestion pipeline for PDFs and cleaned text policy documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.config import AppConfig
from core.schemas import DocumentMetadata, SourceType
from ingestion.chunker import TextChunker
from ingestion.cleaner import TextCleaner
from ingestion.html_loader import HTMLLoader
from ingestion.pdf_loader import PDFLoader


class IngestPipeline:
    """Load supported document formats, chunk them, and persist JSONL output."""

    def __init__(
        self,
        config: AppConfig | None = None,
        pdf_loader: PDFLoader | None = None,
        html_loader: HTMLLoader | None = None,
        cleaner: TextCleaner | None = None,
        chunker: TextChunker | None = None,
    ) -> None:
        """Initialize ingestion pipeline dependencies."""
        self._config = config or AppConfig.from_env()
        self._pdf_loader = pdf_loader or PDFLoader()
        self._html_loader = html_loader or HTMLLoader()
        self._cleaner = cleaner or TextCleaner()
        self._chunker = chunker or TextChunker()

    def ingest_pdf(
        self,
        pdf_path: Path,
        source_type: SourceType,
        provider_name: str,
        product_name: str | None = None,
        source_url: str | None = None,
        published_at: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Ingest a PDF document into chunked JSONL output."""
        document = self._pdf_loader.load(pdf_path)
        cleaned_pages = self._cleaner.clean_pages(document.pages)
        metadata = self._build_metadata(
            document_id=document.document_id,
            title=document.title,
            source_type=source_type,
            provider_name=provider_name,
            product_name=product_name,
            source_url=source_url,
            published_at=published_at,
            tags=tags,
        )
        chunk_payloads = self._chunker.chunk_pages(cleaned_pages, metadata)
        self._write_chunks(chunk_payloads)
        return chunk_payloads

    def ingest_text_file(
        self,
        text_path: Path,
        document_id: str,
        title: str,
        source_type: SourceType,
        provider_name: str,
        product_name: str | None = None,
        source_url: str | None = None,
        published_at: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Ingest a cleaned text file derived from an official HTML page."""
        pages = self._html_loader.load_clean_text(text_path)
        cleaned_pages = self._cleaner.clean_pages(pages)
        metadata = self._build_metadata(
            document_id=document_id,
            title=title,
            source_type=source_type,
            provider_name=provider_name,
            product_name=product_name,
            source_url=source_url,
            published_at=published_at,
            tags=tags,
        )
        chunk_payloads = self._chunker.chunk_pages(cleaned_pages, metadata)
        self._write_chunks(chunk_payloads)
        return chunk_payloads

    def ingest_document(self, manifest_entry: dict[str, Any]) -> list[dict[str, Any]]:
        """Ingest a document based on a dataset manifest entry."""
        format_name = str(manifest_entry["format"]).lower()
        local_path = Path(manifest_entry["local_path"])
        source_type = SourceType(str(manifest_entry["source_type"]))
        provider_name = str(manifest_entry["provider"])
        product_name = manifest_entry.get("product_name")
        source_url = manifest_entry.get("source_url")
        title = str(manifest_entry["title"])
        doc_id = str(manifest_entry["doc_id"])

        if format_name == "pdf":
            return self.ingest_pdf(
                pdf_path=local_path,
                source_type=source_type,
                provider_name=provider_name,
                product_name=product_name,
                source_url=source_url,
                tags=[format_name],
            )

        if format_name in {"txt", "html_text"}:
            return self.ingest_text_file(
                text_path=local_path,
                document_id=doc_id,
                title=title,
                source_type=source_type,
                provider_name=provider_name,
                product_name=product_name,
                source_url=source_url,
                tags=[format_name],
            )

        raise ValueError(f"Unsupported document format: {format_name}")

    def _build_metadata(
        self,
        document_id: str,
        title: str,
        source_type: SourceType,
        provider_name: str,
        product_name: str | None,
        source_url: str | None,
        published_at: str | None,
        tags: list[str] | None,
    ) -> DocumentMetadata:
        """Build a normalized metadata object for chunking."""
        return DocumentMetadata(
            document_id=document_id,
            source_type=source_type,
            title=title,
            provider_name=provider_name,
            product_name=product_name,
            source_url=source_url,
            published_at=published_at,
            tags=tags or [],
        )

    def _write_chunks(self, chunk_payloads: list[dict[str, Any]]) -> None:
        """Append serialized chunk payloads to the processed JSONL output."""
        output_path = self._config.data_dir / "processed" / "chunks.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("a", encoding="utf-8") as file_handle:
            for payload in chunk_payloads:
                file_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
