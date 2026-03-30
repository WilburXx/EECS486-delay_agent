"""Process the local DelayAgent dataset into chunks and, optionally, a FAISS index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import APIConnectionError, APIError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.service import DocumentIngestionService  # noqa: E402
from core.config import AppConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process the local DelayAgent dataset.")
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Only build processed chunks and skip FAISS embedding/index generation.",
    )
    return parser.parse_args()


def main() -> None:
    """Ingest all manifest-backed local documents and print a summary."""
    args = parse_args()
    config = AppConfig.from_env()
    service = DocumentIngestionService(config=config)

    if args.skip_index:
        result = service.ingest_all_documents(build_index=False)
        _print_summary(result, chunk_count=_count_chunks(config))
        return

    try:
        result = service.ingest_all_documents(build_index=True)
    except (APIConnectionError, APIError) as error:
        result = service.ingest_all_documents(build_index=False)
        result["index_error"] = str(error)

    _print_summary(result, chunk_count=_count_chunks(config))


def _count_chunks(config: AppConfig) -> int:
    """Count processed chunk records written to the JSONL output."""
    chunks_path = config.data_dir / "processed" / "chunks.jsonl"
    if not chunks_path.exists():
        return 0
    return sum(1 for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip())


def _print_summary(result: dict[str, object], chunk_count: int) -> None:
    """Print a user-friendly processing summary."""
    print("DelayAgent dataset processing summary")
    print(f"Documents ingested: {result['documents_ingested']}")
    print(f"Chunks written: {chunk_count}")
    print(f"Chunks indexed: {result['chunks_indexed']}")
    print(f"Vector store path: {result['vector_store_path']}")
    if "index_error" in result:
        print(f"Indexing skipped or failed: {result['index_error']}")
    for item in result["documents"]:
        print(f"- {item['document_id']} ({item['source_type']}): {item['path']}")


if __name__ == "__main__":
    main()
