"""Text cleaning helpers for extracted PDF content."""

from __future__ import annotations

import re
from collections import Counter

from ingestion.pdf_loader import LoadedPage


class TextCleaner:
    """Clean page text and remove repeated headers or footers when detectable."""

    _whitespace_pattern = re.compile(r"[ \t]+")
    _newline_pattern = re.compile(r"\n{3,}")

    def clean_pages(self, pages: list[LoadedPage]) -> list[LoadedPage]:
        """Normalize whitespace and strip common repeated boundary lines."""
        header_candidates = Counter(
            self._normalize_boundary_line(lines[0])
            for lines in self._page_lines(pages)
            if lines and self._normalize_boundary_line(lines[0])
        )
        footer_candidates = Counter(
            self._normalize_boundary_line(lines[-1])
            for lines in self._page_lines(pages)
            if lines and self._normalize_boundary_line(lines[-1])
        )

        repeated_headers = self._select_repeated_lines(header_candidates, total_pages=len(pages))
        repeated_footers = self._select_repeated_lines(footer_candidates, total_pages=len(pages))

        cleaned_pages: list[LoadedPage] = []
        for page in pages:
            lines = self._strip_boundaries(page.text.splitlines(), repeated_headers, repeated_footers)
            cleaned_text = self._normalize_text("\n".join(lines))
            cleaned_pages.append(LoadedPage(page_number=page.page_number, text=cleaned_text))

        return cleaned_pages

    def _page_lines(self, pages: list[LoadedPage]) -> list[list[str]]:
        """Return non-empty line lists for each page."""
        return [[line.strip() for line in page.text.splitlines() if line.strip()] for page in pages]

    def _select_repeated_lines(self, counts: Counter[str], total_pages: int) -> set[str]:
        """Keep short repeated boundary lines as likely headers or footers."""
        threshold = max(2, total_pages // 2)
        return {
            line
            for line, count in counts.items()
            if count >= threshold and len(line) <= 120
        }

    def _strip_boundaries(
        self,
        lines: list[str],
        repeated_headers: set[str],
        repeated_footers: set[str],
    ) -> list[str]:
        """Remove matching header and footer lines from a page."""
        normalized_lines = [self._normalize_boundary_line(line) for line in lines]
        start_index = 0
        end_index = len(lines)

        while start_index < end_index and normalized_lines[start_index] in repeated_headers:
            start_index += 1

        while end_index > start_index and normalized_lines[end_index - 1] in repeated_footers:
            end_index -= 1

        return [line.strip() for line in lines[start_index:end_index] if line.strip()]

    def _normalize_boundary_line(self, line: str) -> str:
        """Normalize a line for repeated-header/footer comparison."""
        collapsed = self._whitespace_pattern.sub(" ", line.strip())
        collapsed = re.sub(r"\bpage\s+\d+\b", "page", collapsed, flags=re.IGNORECASE)
        collapsed = re.sub(r"\b\d+\s+of\s+\d+\b", "n of n", collapsed, flags=re.IGNORECASE)
        return collapsed.lower()

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph breaks."""
        normalized_lines = [self._whitespace_pattern.sub(" ", line).strip() for line in text.splitlines()]
        compact_text = "\n".join(line for line in normalized_lines if line)
        return self._newline_pattern.sub("\n\n", compact_text).strip()
