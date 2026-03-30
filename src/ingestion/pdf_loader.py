"""PDF loading utilities backed by PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import fitz
from pydantic import BaseModel, Field


class LoadedPage(BaseModel):
    """A single page extracted from a PDF document."""

    page_number: int = Field(ge=1)
    text: str
    section_title: str | None = None


class PDFDocument(BaseModel):
    """Structured result of PDF extraction."""

    document_id: str
    file_path: Path
    title: str
    pages: list[LoadedPage] = Field(default_factory=list)


class PDFLoader:
    """Extract text from PDF documents while preserving page numbers."""

    def load(self, path: Path) -> PDFDocument:
        """Read a PDF file into a page-aware document model."""
        resolved_path = path.expanduser().resolve()
        with fitz.open(resolved_path) as document:
            pages = [
                LoadedPage(page_number=index + 1, text=page.get_text("text").strip())
                for index, page in enumerate(document)
            ]

            metadata = document.metadata or {}
            title = metadata.get("title") or resolved_path.stem

        return PDFDocument(
            document_id=resolved_path.stem,
            file_path=resolved_path,
            title=title,
            pages=pages,
        )
