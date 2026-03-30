"""Chunking utilities for page-aware document passages."""

from __future__ import annotations

import re
from typing import Any

from core.schemas import Chunk, DocumentMetadata
from ingestion.pdf_loader import LoadedPage


class TextChunker:
    """Create overlapping chunks without aggressively splitting clauses."""

    def __init__(self, target_chars: int = 1200, overlap_chars: int = 250) -> None:
        """Initialize chunk sizing parameters."""
        if overlap_chars >= target_chars:
            raise ValueError("overlap_chars must be smaller than target_chars")

        self._target_chars = target_chars
        self._overlap_chars = overlap_chars

    def chunk_pages(
        self,
        pages: list[LoadedPage],
        metadata: DocumentMetadata,
    ) -> list[dict[str, Any]]:
        """Convert cleaned pages into overlapping chunk payloads with metadata."""
        segments = self._build_segments(pages)
        if not segments:
            return []

        chunk_payloads: list[dict[str, Any]] = []
        window: list[dict[str, Any]] = []
        current_size = 0
        chunk_index = 1

        for segment in segments:
            segment_size = len(segment["text"])
            if window and current_size + segment_size > self._target_chars:
                chunk_payloads.append(self._build_chunk_payload(window, metadata, chunk_index))
                chunk_index += 1
                window = self._build_overlap_window(window)
                current_size = sum(len(item["text"]) for item in window)

            window.append(segment)
            current_size += segment_size

        if window:
            chunk_payloads.append(self._build_chunk_payload(window, metadata, chunk_index))

        return chunk_payloads

    def _build_segments(self, pages: list[LoadedPage]) -> list[dict[str, Any]]:
        """Split pages into paragraph-oriented segments."""
        segments: list[dict[str, Any]] = []
        for page in pages:
            blocks = [block.strip() for block in re.split(r"\n{2,}", page.text) if block.strip()]
            for block in blocks:
                segments.extend(self._split_block(block, page.page_number, page.section_title))

        return segments

    def _split_block(
        self,
        block: str,
        page_number: int,
        section_title: str | None,
    ) -> list[dict[str, Any]]:
        """Split oversized blocks at sentence-like boundaries only when needed."""
        if len(block) <= self._target_chars:
            return [{"text": block, "page_number": page_number, "section_title": section_title}]

        sentences = re.split(r"(?<=[.!?;:])\s+(?=[A-Z0-9(])", block)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if len(sentences) <= 1:
            return self._split_by_length(block, page_number, section_title)

        segments: list[dict[str, Any]] = []
        buffer: list[str] = []
        buffer_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)
            if buffer and buffer_size + sentence_size > self._target_chars:
                segments.append(
                    {
                        "text": " ".join(buffer),
                        "page_number": page_number,
                        "section_title": section_title,
                    }
                )
                buffer = [sentence]
                buffer_size = sentence_size
            else:
                buffer.append(sentence)
                buffer_size += sentence_size + 1

        if buffer:
            segments.append(
                {
                    "text": " ".join(buffer),
                    "page_number": page_number,
                    "section_title": section_title,
                }
            )

        return segments

    def _split_by_length(
        self,
        text: str,
        page_number: int,
        section_title: str | None,
    ) -> list[dict[str, Any]]:
        """Fallback split for long text without clear sentence boundaries."""
        pieces: list[dict[str, Any]] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self._target_chars)
            if end < len(text):
                split_at = text.rfind(" ", start, end)
                end = split_at if split_at > start else end

            pieces.append(
                {
                    "text": text[start:end].strip(),
                    "page_number": page_number,
                    "section_title": section_title,
                }
            )
            start = end

        return [piece for piece in pieces if piece["text"]]

    def _build_overlap_window(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Retain trailing segments to create overlap with the next chunk."""
        overlap: list[dict[str, Any]] = []
        retained_size = 0

        for segment in reversed(segments):
            overlap.insert(0, segment)
            retained_size += len(segment["text"])
            if retained_size >= self._overlap_chars:
                break

        return overlap

    def _build_chunk_payload(
        self,
        segments: list[dict[str, Any]],
        metadata: DocumentMetadata,
        chunk_index: int,
    ) -> dict[str, Any]:
        """Construct a serialized chunk payload with metadata."""
        text = "\n\n".join(segment["text"] for segment in segments).strip()
        page_numbers = sorted({int(segment["page_number"]) for segment in segments})
        section_titles = [
            title
            for title in {segment.get("section_title") for segment in segments}
            if isinstance(title, str) and title
        ]
        chunk = Chunk(
            chunk_id=f"{metadata.document_id}-chunk-{chunk_index:04d}",
            document_id=metadata.document_id,
            text=text,
            position=chunk_index - 1,
            page_number=page_numbers[0],
            section_title=section_titles[0] if len(section_titles) == 1 else None,
            token_count=self._estimate_token_count(text),
        )
        return {
            "chunk": chunk.model_dump(mode="json"),
            "metadata": metadata.model_dump(mode="json"),
            "page_numbers": page_numbers,
            "section_titles": sorted(section_titles),
        }

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count using a simple word-based heuristic."""
        return max(1, round(len(text.split()) * 1.3))
