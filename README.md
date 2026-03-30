# DelayAgent

DelayAgent is a modular Python 3.11 project for analyzing travel delay and baggage delay claims.

## Features

- Structured project layout under `src/`
- Typed domain models with docstrings
- Modular ingestion, retrieval, extraction, and reasoning layers
- Official starter corpus fetch script and manifest
- Mixed-source ingestion for PDFs and airline HTML-derived text
- Environment-based configuration
- Starter test suite with `pytest`

## Project Structure

```text
DelayAgent/
├── data/
│   ├── dataset_manifest.json
│   ├── processed/
│   └── raw/
├── scripts/
├── src/
│   ├── app/
│   ├── core/
│   ├── extraction/
│   ├── ingestion/
│   ├── prompts/
│   ├── reasoning/
│   ├── retrieval/
│   ├── utils/
│   └── main.py
├── tests/
├── .env.example
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.11

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Dataset Setup

Fetch the official starter corpus:

```bash
python -m scripts.fetch_dataset
```

Use `--force` to refetch documents that already exist:

```bash
python -m scripts.fetch_dataset --force
```

This creates:

- `data/dataset_manifest.json`
- PDF files in `data/raw/credit_cards/`
- raw airline HTML in `data/raw/html/`
- cleaned airline text in `data/raw/html_text/`

Airline policy and support pages are stored in both formats so you can keep the original markup for inspection while ingesting the cleaned `.txt` files through the existing pipeline.

## Ingest Mixed Documents

The ingestion pipeline supports:

- PDF files from `data/raw/credit_cards/`
- cleaned `.txt` files derived from official airline HTML pages in `data/raw/html_text/`

To ingest the mixed starter corpus and build the FAISS index, run the API and call `POST /ingest`, or use the app service from Python after fetching the dataset.

For a local command-line processing flow:

```bash
PYTHONPATH=src python -m scripts.process_dataset
```

The processing step reads `data/dataset_manifest.json`, ingests all available local files, writes chunk records to `data/processed/chunks.jsonl`, and rebuilds the FAISS index under `data/processed/vector_store/`.

## Run

CLI smoke test:

```bash
PYTHONPATH=src python -m main
```

FastAPI:

```bash
PYTHONPATH=src uvicorn app.api:app --reload
```

Streamlit UI:

```bash
PYTHONPATH=src streamlit run src/app/ui.py
```

## Test

```bash
PYTHONPATH=src pytest
```

## Notes

Suggested flow:

1. `python -m scripts.fetch_dataset`
2. Start the API with `PYTHONPATH=src uvicorn app.api:app --reload`
3. Call `POST /ingest` to ingest the manifest-backed PDF and HTML-derived text corpus
4. Call `POST /analyze_claim` or use the Streamlit UI
