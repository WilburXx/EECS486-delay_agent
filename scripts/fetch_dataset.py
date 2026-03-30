"""Fetch and prepare the official DelayAgent starter corpus."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingestion.html_loader import HTMLLoader  # noqa: E402


@dataclass(frozen=True)
class DatasetEntry:
    """Dataset manifest entry for an official source document."""

    doc_id: str
    title: str
    source_type: str
    provider: str
    product_name: str
    format: str
    source_url: str
    local_path: str


PDF_ENTRIES: list[DatasetEntry] = [
    DatasetEntry(
        doc_id="chase_sapphire_preferred_guide_to_benefits",
        title="Chase Sapphire Preferred Guide to Benefits",
        source_type="credit_card",
        provider="Chase",
        product_name="Sapphire Preferred",
        format="pdf",
        source_url="https://static.chasecdn.com/content/services/structured-document/document.en.pdf/card/benefits-center/product-benefits-guide-pdf/BGC11387_v2.pdf",
        local_path="data/raw/credit_cards/BGC11387_v2.pdf",
    ),
    DatasetEntry(
        doc_id="amex_trip_delay_insurance",
        title="American Express Trip Delay Insurance",
        source_type="credit_card",
        provider="American Express",
        product_name="Trip Delay Insurance",
        format="pdf",
        source_url="https://www.americanexpress.com/content/dam/amex/us/credit-cards/features-benefits/policies/trip-delay/trip-delay-insurance-500-6hours.pdf",
        local_path="data/raw/credit_cards/trip-delay-insurance-500-6hours.pdf",
    ),
    DatasetEntry(
        doc_id="mastercard_baggage_delay",
        title="Mastercard Baggage Delay",
        source_type="credit_card",
        provider="Mastercard",
        product_name="Baggage Delay",
        format="pdf",
        source_url="https://www.mastercard.us/content/dam/mccom/en-us/documents/Cardholder%20Benefits/GTB-Baggage_Delay_Std_MC_Cov_012315.pdf",
        local_path="data/raw/credit_cards/GTB-Baggage_Delay_Std_MC_Cov_012315.pdf",
    ),
    DatasetEntry(
        doc_id="mastercard_lost_or_damaged_luggage",
        title="Mastercard Lost or Damaged Luggage",
        source_type="credit_card",
        provider="Mastercard",
        product_name="Lost or Damaged Luggage",
        format="pdf",
        source_url="https://www.mastercard.us/content/dam/mccom/en-us/documents/Cardholder%20Benefits/GTB-Lost_or_Damaged_Luggage_Std_MC_Cov_012315.pdf",
        local_path="data/raw/credit_cards/GTB-Lost_or_Damaged_Luggage_Std_MC_Cov_012315.pdf",
    ),
    DatasetEntry(
        doc_id="mastercard_trip_cancellation",
        title="Mastercard Trip Cancellation",
        source_type="credit_card",
        provider="Mastercard",
        product_name="Trip Cancellation",
        format="pdf",
        source_url="https://www.mastercard.us/content/dam/mccom/en-us/documents/Cardholder%20Benefits/GTB-Trip_Cancellation_MC_Base_form_012315.pdf",
        local_path="data/raw/credit_cards/GTB-Trip_Cancellation_MC_Base_form_012315.pdf",
    ),
]


HTML_ENTRIES: list[DatasetEntry] = [
    DatasetEntry(
        doc_id="delta_us_contract_of_carriage",
        title="Delta U.S. Contract of Carriage",
        source_type="airline",
        provider="Delta Air Lines",
        product_name="U.S. Contract of Carriage",
        format="html_text",
        source_url="https://www.delta.com/us/en/legal/contract-of-carriage-dgr",
        local_path="data/raw/html_text/delta_us_contract_of_carriage.txt",
    ),
    DatasetEntry(
        doc_id="delta_international_contract_of_carriage",
        title="Delta International Contract of Carriage",
        source_type="airline",
        provider="Delta Air Lines",
        product_name="International Contract of Carriage",
        format="html_text",
        source_url="https://www.delta.com/us/en/legal/contract-of-carriage-igr",
        local_path="data/raw/html_text/delta_international_contract_of_carriage.txt",
    ),
    DatasetEntry(
        doc_id="delta_delayed_lost_damaged_baggage",
        title="Delta Delayed, Lost, or Damaged Baggage",
        source_type="airline",
        provider="Delta Air Lines",
        product_name="Delayed, Lost, or Damaged Baggage",
        format="html_text",
        source_url="https://www.delta.com/us/en/baggage/delayed-lost-damaged-baggage",
        local_path="data/raw/html_text/delta_delayed_lost_damaged_baggage.txt",
    ),
    DatasetEntry(
        doc_id="delta_delayed_or_canceled_flight",
        title="Delta Delayed or Canceled Flight",
        source_type="airline",
        provider="Delta Air Lines",
        product_name="Delayed or Canceled Flight",
        format="html_text",
        source_url="https://www.delta.com/us/en/change-cancel/delayed-or-canceled-flight",
        local_path="data/raw/html_text/delta_delayed_or_canceled_flight.txt",
    ),
    DatasetEntry(
        doc_id="delta_baggage_overview",
        title="Delta Baggage Overview",
        source_type="airline",
        provider="Delta Air Lines",
        product_name="Baggage Overview and Fees",
        format="html_text",
        source_url="https://www.delta.com/us/en/baggage/overview",
        local_path="data/raw/html_text/delta_baggage_overview.txt",
    ),
    DatasetEntry(
        doc_id="delta_bag_guarantee",
        title="Delta 20-Minute Baggage Guarantee",
        source_type="airline",
        provider="Delta Air Lines",
        product_name="20-Minute Baggage Guarantee",
        format="html_text",
        source_url="https://www.delta.com/bag-guarantee",
        local_path="data/raw/html_text/delta_bag_guarantee.txt",
    ),
    DatasetEntry(
        doc_id="united_contract_of_carriage",
        title="United Contract of Carriage",
        source_type="airline",
        provider="United Airlines",
        product_name="Contract of Carriage",
        format="html_text",
        source_url="https://www.united.com/en/us/fly/contract-of-carriage.html",
        local_path="data/raw/html_text/united_contract_of_carriage.txt",
    ),
    DatasetEntry(
        doc_id="american_conditions_of_carriage",
        title="American Airlines Conditions of Carriage",
        source_type="airline",
        provider="American Airlines",
        product_name="Conditions of Carriage",
        format="html_text",
        source_url="https://www.aa.com/i18n/customer-service/support/conditions-of-carriage.jsp",
        local_path="data/raw/html_text/american_conditions_of_carriage.txt",
    ),
    DatasetEntry(
        doc_id="american_delayed_or_damaged_baggage",
        title="American Airlines Delayed or Damaged Baggage",
        source_type="airline",
        provider="American Airlines",
        product_name="Delayed or Damaged Baggage",
        format="html_text",
        source_url="https://www.aa.com/i18n/travel-info/baggage/delayed-or-damaged-baggage.jsp",
        local_path="data/raw/html_text/american_delayed_or_damaged_baggage.txt",
    ),
    DatasetEntry(
        doc_id="alaska_contract_of_carriage",
        title="Alaska Airlines Contract of Carriage",
        source_type="airline",
        provider="Alaska Airlines",
        product_name="Contract of Carriage",
        format="html_text",
        source_url="https://www.alaskaair.com/content/legal/contract-of-carriage/view-all",
        local_path="data/raw/html_text/alaska_contract_of_carriage.txt",
    ),
]


DATASET_ENTRIES: list[DatasetEntry] = PDF_ENTRIES + HTML_ENTRIES


class DatasetFetcher:
    """Fetch and materialize the official DelayAgent starter corpus."""

    def __init__(self, project_root: Path, force: bool = False) -> None:
        """Initialize fetch paths and helpers."""
        self._project_root = project_root
        self._data_dir = project_root / "data"
        self._raw_dir = self._data_dir / "raw"
        self._manifest_path = self._data_dir / "dataset_manifest.json"
        self._html_loader = HTMLLoader()
        self._force = force

    def run(self) -> dict[str, Any]:
        """Fetch all dataset entries and update the manifest."""
        self._ensure_directories()
        self._update_manifest(DATASET_ENTRIES)

        summary = {
            "fetched": 0,
            "skipped": 0,
            "failed": 0,
            "items": [],
        }

        for entry in DATASET_ENTRIES:
            try:
                status = self._fetch_entry(entry)
            except Exception as error:  # pragma: no cover - operational reporting
                status = {"doc_id": entry.doc_id, "status": "failed", "detail": str(error)}
                summary["failed"] += 1
            else:
                summary[status["status"]] += 1

            summary["items"].append(status)

        return summary

    def _ensure_directories(self) -> None:
        """Create required dataset directories."""
        for path in (
            self._raw_dir / "credit_cards",
            self._raw_dir / "html",
            self._raw_dir / "html_text",
            self._data_dir / "processed",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _fetch_entry(self, entry: DatasetEntry) -> dict[str, str]:
        """Fetch a single dataset entry."""
        if entry.format == "pdf":
            return self._download_pdf(entry)
        if entry.format == "html_text":
            return self._download_html_bundle(entry)
        raise ValueError(f"Unsupported dataset format: {entry.format}")

    def _download_pdf(self, entry: DatasetEntry) -> dict[str, str]:
        """Download a PDF source to the raw corpus directory."""
        destination = self._project_root / entry.local_path
        if destination.exists() and not self._force:
            return {"doc_id": entry.doc_id, "status": "skipped", "detail": str(destination)}

        response = requests.get(
            entry.source_url,
            timeout=60.0,
            headers=self._browser_headers(accept="application/pdf,application/octet-stream;q=0.9,*/*;q=0.8"),
        )
        response.raise_for_status()
        destination.write_bytes(response.content)
        return {"doc_id": entry.doc_id, "status": "fetched", "detail": str(destination)}

    def _download_html_bundle(self, entry: DatasetEntry) -> dict[str, str]:
        """Fetch an HTML page and save both raw and cleaned text forms."""
        raw_destination, text_destination = self._html_output_paths(entry)
        if raw_destination.exists() and text_destination.exists() and not self._force:
            return {"doc_id": entry.doc_id, "status": "skipped", "detail": str(text_destination)}

        raw_html = self._fetch_html(entry.source_url)
        self._validate_html_response(raw_html, entry.source_url)
        cleaned_text = self._clean_html_to_text(
            html=raw_html,
            source_url=entry.source_url,
            document_id=entry.doc_id,
            title=entry.title,
        )

        raw_destination.write_text(raw_html, encoding="utf-8")
        text_destination.write_text(cleaned_text, encoding="utf-8")
        return {
            "doc_id": entry.doc_id,
            "status": "fetched",
            "detail": f"{raw_destination} | {text_destination}",
        }

    def _fetch_html(self, url: str) -> str:
        """Fetch raw HTML using browser-like headers."""
        return self._html_loader.fetch(url, timeout=60.0)

    def _clean_html_to_text(
        self,
        html: str,
        source_url: str,
        document_id: str,
        title: str,
    ) -> str:
        """Convert raw HTML into readable structured text."""
        document = self._html_loader.parse(
            html=html,
            source_url=source_url,
            document_id=document_id,
            title=title,
        )
        return self._html_loader.to_clean_text(document)

    def _html_output_paths(self, entry: DatasetEntry) -> tuple[Path, Path]:
        """Return raw HTML and cleaned text destinations for an HTML entry."""
        filename = self._generate_html_filename(entry)
        raw_destination = self._raw_dir / "html" / f"{filename}.html"
        text_destination = self._raw_dir / "html_text" / f"{filename}.txt"
        return raw_destination, text_destination

    def _generate_html_filename(self, entry: DatasetEntry) -> str:
        """Generate a stable filename for an HTML dataset entry."""
        return entry.doc_id

    def _update_manifest(self, entries: list[DatasetEntry]) -> None:
        """Write the dataset manifest using the current entry set."""
        manifest = [asdict(entry) for entry in entries]
        self._manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _browser_headers(self, accept: str) -> dict[str, str]:
        """Return a browser-like header set for remote fetching."""
        return {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _validate_html_response(self, raw_html: str, source_url: str) -> None:
        """Reject obvious block pages or empty HTML responses."""
        normalized = raw_html.strip().lower()
        if not normalized:
            raise ValueError(f"Received empty HTML response for {source_url}")
        if "access denied" in normalized and "you don't have permission" in normalized:
            raise ValueError(f"Blocked by source site while fetching {source_url}")
        if "<title>enable javascript" in normalized and len(normalized) < 600:
            raise ValueError(f"Received incomplete JavaScript gate page for {source_url}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fetch the DelayAgent starter dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch files even if they already exist locally.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the dataset fetcher and print a summary."""
    args = parse_args()
    fetcher = DatasetFetcher(project_root=PROJECT_ROOT, force=args.force)
    summary = fetcher.run()

    print("DelayAgent dataset fetch summary")
    print(f"Fetched: {summary['fetched']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Failed: {summary['failed']}")
    for item in summary["items"]:
        print(f"- {item['doc_id']}: {item['status']} -> {item['detail']}")


if __name__ == "__main__":
    main()
