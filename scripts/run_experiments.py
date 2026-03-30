"""Run scenario-based experiments for DelayAgent and save structured results."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.service import ClaimAnalysisService  # noqa: E402
from core.config import AppConfig  # noqa: E402


@dataclass(frozen=True)
class ExperimentScenario:
    """Scenario definition for batch evaluation."""

    id: str
    query: str
    top_k: int = 5
    expected_label: str | None = None
    expected_primary_lane: str | None = None
    notes: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DelayAgent experiment scenarios.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "scenarios.json",
        help="Path to the JSON scenario file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "latest.json",
        help="Path to the JSON results file to write.",
    )
    return parser.parse_args()


def load_scenarios(path: Path) -> list[ExperimentScenario]:
    """Load scenarios from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ExperimentScenario(**item) for item in payload]


def infer_primary_lane(summary: str) -> str:
    """Infer the primary lane from the natural-language summary."""
    normalized = summary.lower()
    if "airline policy" in normalized:
        return "airline"
    if "card or insurance benefit" in normalized or "benefit policy" in normalized:
        return "benefit"
    return "general"


def run_scenario(service: ClaimAnalysisService, scenario: ExperimentScenario) -> dict[str, Any]:
    """Run a single experiment scenario and return a structured result record."""
    response = service.analyze_claim(user_query=scenario.query, top_k=scenario.top_k)
    primary_lane = infer_primary_lane(response.summary)
    retrieved_doc_ids = [passage.metadata.document_id for passage in response.retrieved_passages]

    label_match = (
        None
        if scenario.expected_label is None
        else response.eligibility.label.value == scenario.expected_label
    )
    lane_match = (
        None
        if scenario.expected_primary_lane is None or scenario.expected_primary_lane == "airline_or_benefit"
        else primary_lane == scenario.expected_primary_lane
    )

    return {
        "scenario": asdict(scenario),
        "result": {
            "summary": response.summary,
            "eligibility_label": response.eligibility.label.value,
            "confidence": response.eligibility.confidence,
            "primary_lane": primary_lane,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieved_citations": [passage.citation for passage in response.retrieved_passages],
            "supporting_chunk_ids": response.eligibility.supporting_chunk_ids,
            "delay_threshold_hours": response.extracted_requirements.minimum_delay_threshold_hours.value,
            "reimbursement_cap": response.extracted_requirements.reimbursement_cap.value,
            "filing_deadline": response.extracted_requirements.filing_deadline.value,
            "covered_expense_categories": response.extracted_requirements.covered_expense_categories.value,
            "required_documentation": response.extracted_requirements.required_documentation.value,
            "follow_up_questions": response.claim_plan.follow_up_questions,
        },
        "checks": {
            "label_match": label_match,
            "primary_lane_match": lane_match,
        },
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute simple aggregate experiment statistics."""
    successful_records = [record for record in records if "checks" in record]
    label_checks = [
        record["checks"]["label_match"]
        for record in successful_records
        if record["checks"]["label_match"] is not None
    ]
    lane_checks = [
        record["checks"]["primary_lane_match"]
        for record in successful_records
        if record["checks"]["primary_lane_match"] is not None
    ]
    return {
        "total_scenarios": len(records),
        "successful_scenarios": len(successful_records),
        "label_match_rate": (sum(label_checks) / len(label_checks)) if label_checks else None,
        "primary_lane_match_rate": (sum(lane_checks) / len(lane_checks)) if lane_checks else None,
    }


def main() -> None:
    """Run all experiment scenarios and write the results to disk."""
    args = parse_args()
    scenarios = load_scenarios(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config = AppConfig.from_env()
    service = ClaimAnalysisService(config=config)

    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        try:
            record = run_scenario(service, scenario)
        except Exception as error:
            record = {
                "scenario": asdict(scenario),
                "error": str(error),
            }
        records.append(record)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario_file": str(args.input),
        "summary": build_summary(records),
        "records": records,
    }

    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("DelayAgent experiment run complete")
    print(f"Scenarios: {payload['summary']['total_scenarios']}")
    print(f"Results file: {args.output}")
    if payload["summary"]["label_match_rate"] is not None:
        print(f"Label match rate: {payload['summary']['label_match_rate']:.2%}")
    if payload["summary"]["primary_lane_match_rate"] is not None:
        print(f"Primary lane match rate: {payload['summary']['primary_lane_match_rate']:.2%}")


if __name__ == "__main__":
    main()
