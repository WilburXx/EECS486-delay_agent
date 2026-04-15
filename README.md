# DelayAgent

DelayAgent is a modular Python 3.11 project that retrieves airline and credit card delay policies, determines claim eligibility based on policy rules, and extracts relevant policy fields from natural-language queries.

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
- On Linux, make sure the `venv` module is available for your Python 3.11 install.

## Setup

Linux and macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` and set your OpenAI API key:

```
OPENAI_API_KEY=your_key_here
```

If `python3.11` is not on your `PATH` but Python 3.11 is your default `python3`, this also works:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
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

All run commands below work on Linux and macOS from the project root after activating the virtual environment.

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

## Experiments

To run the scenario-based experiment suite:

```bash
PYTHONPATH=src python -m scripts.run_experiments
```

This reads scenarios from `experiments/scenarios.json` and writes a structured results file to:

```text
experiments/results/latest.json
```

You can use this output to summarize retrieval behavior, eligibility labels, lane selection, and extracted policy fields in your report.

## Notes

Suggested flow:

1. `python -m scripts.fetch_dataset`
2. `PYTHONPATH=src python -m scripts.process_dataset`
3. Start the API with `PYTHONPATH=src uvicorn app.api:app --reload` or open the Streamlit UI
4. Call `POST /analyze_claim` or use the Streamlit UI

## Linux Compatibility

DelayAgent is designed to run cross-platform and uses `pathlib` for filesystem paths, so there are no hard-coded macOS-only runtime paths in the application code.

If you are running on Linux, the usual workflow is:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
PYTHONPATH=src python -m scripts.process_dataset
PYTHONPATH=src streamlit run src/app/ui.py
```
