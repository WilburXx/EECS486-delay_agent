"""HTML fetching and cleaned text loading utilities for official policy pages."""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field

from ingestion.pdf_loader import LoadedPage


class HTMLSection(BaseModel):
    """A readable section extracted from an official HTML policy page."""

    section_number: int = Field(ge=1)
    title: str
    text: str


class HTMLDocument(BaseModel):
    """Structured HTML extraction result."""

    document_id: str
    title: str
    source_url: str
    raw_html: str
    sections: list[HTMLSection] = Field(default_factory=list)


class HTMLLoader:
    """Fetch official HTML pages and extract readable main content."""

    _block_tags = ("h1", "h2", "h3", "h4", "p", "li")
    _boilerplate_selectors = (
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "form",
        "noscript",
        "svg",
        ".breadcrumb",
        ".breadcrumbs",
        ".cookie-banner",
        ".legal-links",
        ".social-share",
    )

    def fetch(self, url: str, timeout: float = 30.0) -> str:
        """Fetch raw HTML from a URL."""
        response = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        response.raise_for_status()
        return response.text

    def parse(self, html: str, source_url: str, document_id: str, title: str | None = None) -> HTMLDocument:
        """Parse HTML into readable sections."""
        soup = BeautifulSoup(html, "html.parser")
        next_data_text = self._extract_next_data_text(soup)
        for selector in self._boilerplate_selectors:
            for tag in soup.select(selector):
                tag.decompose()

        root = self._select_main_content(soup)
        extracted_title = title or (soup.title.string.strip() if soup.title and soup.title.string else document_id)
        sections = self._extract_sections(root)
        if not sections:
            sections = self._extract_nextjs_sections(next_data_text)

        return HTMLDocument(
            document_id=document_id,
            title=extracted_title,
            source_url=source_url,
            raw_html=html,
            sections=sections,
        )

    def to_clean_text(self, document: HTMLDocument) -> str:
        """Render extracted HTML sections as a clean text file."""
        lines: list[str] = [f"# {document.title}", f"URL: {document.source_url}", ""]
        for section in document.sections:
            lines.append(f"## {section.section_number}. {section.title}")
            lines.append(section.text)
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def load_clean_text(self, path: Path) -> list[LoadedPage]:
        """Load a cleaned text file into section-aware pages for ingestion."""
        content = path.read_text(encoding="utf-8")
        pattern = re.compile(r"^##\s+(\d+)\.\s+(.*?)\n", re.MULTILINE)
        matches = list(pattern.finditer(content))
        pages: list[LoadedPage] = []

        if not matches:
            normalized = content.strip()
            if normalized:
                pages.append(LoadedPage(page_number=1, section_title=path.stem, text=normalized))
            return pages

        for index, match in enumerate(matches):
            section_number = int(match.group(1))
            section_title = match.group(2).strip()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            text = content[start:end].strip()
            if text:
                pages.append(
                    LoadedPage(
                        page_number=section_number,
                        section_title=section_title,
                        text=text,
                    )
                )

        return pages

    def _select_main_content(self, soup: BeautifulSoup) -> Tag:
        """Pick the main content container from a policy page."""
        selectors = [
            "main",
            "article",
            "[role='main']",
            ".main-content",
            ".content",
            "#main",
            "#content",
        ]
        for selector in selectors:
            node = soup.select_one(selector)
            if isinstance(node, Tag):
                return node
        return soup.body if isinstance(soup.body, Tag) else soup

    def _extract_sections(self, root: Tag) -> list[HTMLSection]:
        """Split readable HTML into titled sections."""
        sections: list[HTMLSection] = []
        current_title = "Overview"
        current_lines: list[str] = []
        section_number = 1

        for element in root.find_all(self._block_tags):
            text = element.get_text(" ", strip=True)
            if not text:
                continue

            if element.name in {"h1", "h2", "h3", "h4"}:
                if current_lines:
                    sections.append(
                        HTMLSection(
                            section_number=section_number,
                            title=current_title,
                            text="\n".join(current_lines).strip(),
                        )
                    )
                    section_number += 1
                    current_lines = []
                current_title = text
            elif element.name == "li":
                current_lines.append(f"- {text}")
            else:
                current_lines.append(text)

        if current_lines:
            sections.append(
                HTMLSection(
                    section_number=section_number,
                    title=current_title,
                    text="\n".join(current_lines).strip(),
                )
            )

        return sections

    def _extract_next_data_text(self, soup: BeautifulSoup) -> str | None:
        """Extract raw Next.js payload text before scripts are removed."""
        next_data = soup.find("script", id="__NEXT_DATA__")
        if not next_data:
            return None
        return next_data.text or next_data.string

    def _extract_nextjs_sections(self, next_data_text: str | None) -> list[HTMLSection]:
        """Extract readable sections from embedded Next.js page data when DOM text is sparse."""
        if not next_data_text:
            return []

        try:
            payload = json.loads(next_data_text)
        except json.JSONDecodeError:
            return []

        sections: list[HTMLSection] = []
        self._walk_nextjs_payload(payload, sections)

        deduped: list[HTMLSection] = []
        seen: set[tuple[str, str]] = set()
        for index, section in enumerate(sections, start=1):
            key = (section.title, section.text)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                HTMLSection(
                    section_number=index,
                    title=section.title,
                    text=section.text,
                )
            )

        return deduped

    def _walk_nextjs_payload(self, node: object, sections: list[HTMLSection]) -> None:
        """Recursively walk a Next.js payload and collect text-heavy sections."""
        if isinstance(node, dict):
            section = self._section_from_mapping(node)
            if section is not None:
                sections.append(section)
            for value in node.values():
                self._walk_nextjs_payload(value, sections)
            return

        if isinstance(node, list):
            for item in node:
                self._walk_nextjs_payload(item, sections)

    def _section_from_mapping(self, mapping: dict[object, object]) -> HTMLSection | None:
        """Build a section from a structured mapping when it has a heading and body text."""
        json_rte = mapping.get("json_rte")
        if isinstance(json_rte, dict):
            title = self._first_string(mapping, ("title", "heading", "headline", "name", "label"))
            body_text = self._render_json_rte(json_rte)
            if title and len(body_text) >= 40:
                return HTMLSection(section_number=1, title=title, text=body_text)

        title = self._first_string(
            mapping,
            ("title", "heading", "headline", "name", "label", "eyebrow"),
        )
        body_parts = self._collect_strings(
            mapping,
            ("text", "copy", "description", "body", "content", "details"),
        )
        body_text = "\n".join(part for part in body_parts if part).strip()

        if not title or len(body_text) < 40:
            return None

        return HTMLSection(section_number=1, title=title, text=body_text)

    def _first_string(self, mapping: dict[object, object], keys: tuple[str, ...]) -> str | None:
        """Return the first non-empty string found for a set of keys."""
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, str):
                cleaned = self._normalize_embedded_text(value)
                if cleaned:
                    return cleaned
        return None

    def _collect_strings(self, mapping: dict[object, object], keys: tuple[str, ...]) -> list[str]:
        """Collect normalized string values from selected keys."""
        parts: list[str] = []
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, str):
                cleaned = self._normalize_embedded_text(value)
                if cleaned:
                    parts.append(cleaned)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        cleaned = self._normalize_embedded_text(item)
                        if cleaned:
                            parts.append(cleaned)
        return parts

    def _normalize_embedded_text(self, value: str) -> str:
        """Normalize text recovered from embedded JSON content."""
        if value.startswith(("http://", "https://")) and " " not in value:
            return value.strip()
        cleaned = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _render_json_rte(self, node: dict[str, object]) -> str:
        """Render a Contentstack-style JSON rich text document into plain text."""
        lines: list[str] = []
        self._walk_json_rte(node, lines)
        rendered = "\n".join(line for line in lines if line.strip())
        rendered = re.sub(r"\n{3,}", "\n\n", rendered)
        return rendered.strip()

    def _walk_json_rte(self, node: object, lines: list[str]) -> None:
        """Walk a JSON rich text structure and collect readable text lines."""
        if isinstance(node, dict):
            node_type = node.get("type")
            text_value = node.get("text")
            if isinstance(text_value, str):
                cleaned = self._normalize_embedded_text(text_value)
                if cleaned:
                    lines.append(cleaned)
                if node.get("break"):
                    lines.append("")
                return

            if node_type in {"h1", "h2", "h3", "h4", "p", "fragment", "li"} and lines:
                lines.append("")

            children = node.get("children")
            if isinstance(children, list):
                for child in children:
                    self._walk_json_rte(child, lines)
            return

        if isinstance(node, list):
            for item in node:
                self._walk_json_rte(item, lines)
