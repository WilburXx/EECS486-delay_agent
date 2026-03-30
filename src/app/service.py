"""Top-level application services for ingestion and claim analysis."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from core.config import AppConfig
from core.models import DelayAnalysisResult, TravelClaim
from core.schemas import AnalysisResponse, ClaimPlan, RetrievedPassage, SourceType
from extraction.extractor import PolicyRequirementExtractor
from extraction.parser import ClaimFactExtractor
from ingestion.html_loader import HTMLLoader
from ingestion.ingest_pipeline import IngestPipeline
from reasoning.claim_analyzer import ClaimAnalyzer
from reasoning.eligibility import EligibilityEvaluator
from reasoning.uncertainty import UncertaintyAssessment, UncertaintyDetector
from retrieval.policy_repository import PolicyRepository
from retrieval.retriever import PassageRetriever


@dataclass
class LaneAnalysis:
    """Internal analysis result for a single coverage lane."""

    lane_name: str
    source_types: list[SourceType]
    retrieved_passages: list[RetrievedPassage]
    extracted_requirements: object
    eligibility: object
    uncertainty: UncertaintyAssessment


class DelayAgentService:
    """Coordinate ingestion, extraction, retrieval, and reasoning."""

    def __init__(
        self,
        extractor: ClaimFactExtractor | None = None,
        policy_repository: PolicyRepository | None = None,
        analyzer: ClaimAnalyzer | None = None,
    ) -> None:
        """Initialize service dependencies."""
        self._extractor = extractor or ClaimFactExtractor()
        self._policy_repository = policy_repository or PolicyRepository()
        self._analyzer = analyzer or ClaimAnalyzer()

    def analyze_claim(self, claim: TravelClaim) -> DelayAnalysisResult:
        """Run the end-to-end claim analysis pipeline."""
        facts = self._extractor.extract(claim)
        rule = self._policy_repository.get_rule(claim.claim_type)
        return self._analyzer.analyze(claim=claim, facts=facts, rule=rule)


class DocumentIngestionService:
    """Ingest manifest-backed raw documents and rebuild the retrieval index."""

    def __init__(
        self,
        config: AppConfig | None = None,
        ingest_pipeline: IngestPipeline | None = None,
        retriever: PassageRetriever | None = None,
        html_loader: HTMLLoader | None = None,
    ) -> None:
        """Initialize ingestion dependencies."""
        self._config = config or AppConfig.from_env()
        self._ingest_pipeline = ingest_pipeline or IngestPipeline(config=self._config)
        self._retriever = retriever or PassageRetriever(config=self._config)
        self._html_loader = html_loader or HTMLLoader()

    def ingest_all_pdfs(self) -> dict[str, object]:
        """Backward-compatible entry point for ingesting all supported raw documents."""
        return self.ingest_all_documents()

    def ingest_all_documents(self, build_index: bool = True) -> dict[str, object]:
        """Ingest every manifest entry and optionally rebuild the FAISS index."""
        processed_dir = self._config.data_dir / "processed"
        chunks_path = processed_dir / "chunks.jsonl"
        vector_store_dir = processed_dir / "vector_store"
        manifest_path = self._config.data_dir / "dataset_manifest.json"

        processed_dir.mkdir(parents=True, exist_ok=True)
        chunks_path.write_text("", encoding="utf-8")
        ingested_documents: list[dict[str, str]] = []

        for entry in self._load_manifest_entries(manifest_path):
            local_path = Path(entry["local_path"])
            if not local_path.exists():
                continue

            self._ensure_html_text_exists(entry, local_path)
            self._ingest_pipeline.ingest_document(entry)
            ingested_documents.append(
                {
                    "document_id": str(entry["doc_id"]),
                    "source_type": str(entry["source_type"]),
                    "path": str(local_path),
                }
            )

        indexed_chunks = 0
        index_error: str | None = None
        if ingested_documents and build_index:
            indexed_chunks = self._retriever.index_chunks(
                chunks_path=chunks_path,
                persist_directory=vector_store_dir,
            )

        result: dict[str, object] = {
            "documents_ingested": len(ingested_documents),
            "chunks_indexed": indexed_chunks,
            "vector_store_path": str(vector_store_dir),
            "documents": ingested_documents,
        }
        if index_error is not None:
            result["index_error"] = index_error
        return result

    def _load_manifest_entries(self, manifest_path: Path) -> list[dict[str, object]]:
        """Load dataset manifest entries using project-relative paths."""
        if not manifest_path.exists():
            return []

        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        project_root = manifest_path.parent.parent
        resolved_entries: list[dict[str, object]] = []
        for entry in entries:
            resolved_entry = dict(entry)
            resolved_path = (project_root / str(entry["local_path"])).resolve()
            if not resolved_path.exists():
                fallback = self._find_fallback_path(
                    filename=Path(str(entry["local_path"])).name,
                    data_root=project_root / "data" / "raw",
                )
                if fallback is not None:
                    resolved_path = fallback.resolve()
            resolved_entry["local_path"] = str(resolved_path)
            resolved_entries.append(resolved_entry)
        return resolved_entries

    def _find_fallback_path(self, filename: str, data_root: Path) -> Path | None:
        """Find a dataset file anywhere under the raw data directory by filename."""
        matches = sorted(data_root.rglob(filename))
        return matches[0] if matches else None

    def _ensure_html_text_exists(self, entry: dict[str, object], local_path: Path) -> None:
        """Generate a cleaned text file from raw HTML when the manifest expects html_text."""
        if str(entry.get("format")) != "html_text":
            return
        if local_path.suffix.lower() == ".txt" and local_path.exists():
            return

        document_id = str(entry["doc_id"])
        raw_html_path = self._config.data_dir / "raw" / "html" / f"{document_id}.html"
        text_path = self._config.data_dir / "raw" / "html_text" / f"{document_id}.txt"
        if not raw_html_path.exists() or text_path.exists():
            return

        html = raw_html_path.read_text(encoding="utf-8")
        document = self._html_loader.parse(
            html=html,
            source_url=str(entry.get("source_url", "")),
            document_id=document_id,
            title=str(entry["title"]),
        )
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(self._html_loader.to_clean_text(document), encoding="utf-8")
        entry["local_path"] = str(text_path.resolve())


class ClaimAnalysisService:
    """Run the end-to-end claim analysis workflow for a natural-language query."""

    def __init__(
        self,
        config: AppConfig | None = None,
        retriever: PassageRetriever | None = None,
        extractor: PolicyRequirementExtractor | None = None,
        eligibility_evaluator: EligibilityEvaluator | None = None,
        uncertainty_detector: UncertaintyDetector | None = None,
    ) -> None:
        """Initialize analysis dependencies."""
        self._config = config or AppConfig.from_env()
        self._retriever = retriever or PassageRetriever(config=self._config)
        self._extractor = extractor or PolicyRequirementExtractor(config=self._config)
        self._eligibility_evaluator = eligibility_evaluator or EligibilityEvaluator()
        self._uncertainty_detector = uncertainty_detector or UncertaintyDetector()

    def analyze_claim(self, user_query: str, top_k: int = 5) -> AnalysisResponse:
        """Analyze a natural-language disruption query and return a structured response."""
        self._ensure_retriever_ready()
        lane_analyses = self._analyze_relevant_lanes(user_query=user_query, top_k=top_k)
        if not lane_analyses:
            lane_analyses = [
                self._analyze_lane(
                    lane_name="general",
                    user_query=user_query,
                    top_k=top_k,
                    source_types=None,
                )
            ]

        primary_lane, secondary_lanes = self._choose_primary_lane(user_query, lane_analyses)
        combined_passages = self._merge_passages(
            primary_passages=primary_lane.retrieved_passages,
            secondary_passages=[lane.retrieved_passages for lane in secondary_lanes],
            top_k=top_k,
        )
        combined_source_types = self._collect_source_types(combined_passages)
        claim_plan = self._build_claim_plan(
            uncertainty=primary_lane.uncertainty,
            eligibility=primary_lane.eligibility,
            retrieved_passages=combined_passages,
        )
        summary = self._build_summary(
            primary_lane=primary_lane,
            secondary_lanes=secondary_lanes,
        )

        return AnalysisResponse(
            claim_id=self._build_claim_id(user_query),
            source_types_considered=combined_source_types,
            retrieved_passages=combined_passages,
            extracted_requirements=primary_lane.extracted_requirements,
            eligibility=primary_lane.eligibility,
            claim_plan=claim_plan,
            summary=summary,
        )

    def _analyze_lane(
        self,
        lane_name: str,
        user_query: str,
        top_k: int,
        source_types: list[SourceType] | None,
    ) -> LaneAnalysis:
        """Run analysis for a single lane using a constrained or unconstrained source set."""
        retrieved_passages = self._retrieve_passages(
            user_query=user_query,
            top_k=top_k,
            source_types=source_types,
        )
        extracted_requirements = self._extractor.extract(user_query=user_query, passages=retrieved_passages)
        eligibility = self._eligibility_evaluator.evaluate(
            user_query=user_query,
            extracted_requirements=extracted_requirements,
            retrieved_passages=retrieved_passages,
        )
        uncertainty = self._uncertainty_detector.assess(
            user_query=user_query,
            retrieved_passages=retrieved_passages,
        )
        return LaneAnalysis(
            lane_name=lane_name,
            source_types=source_types or [],
            retrieved_passages=retrieved_passages,
            extracted_requirements=extracted_requirements,
            eligibility=eligibility,
            uncertainty=uncertainty,
        )

    def _analyze_relevant_lanes(self, user_query: str, top_k: int) -> list[LaneAnalysis]:
        """Analyze the lanes implied by the query."""
        lane_specs = self._infer_lane_specs(user_query)
        analyses: list[LaneAnalysis] = []
        for lane_name, source_types in lane_specs:
            analyses.append(
                self._analyze_lane(
                    lane_name=lane_name,
                    user_query=user_query,
                    top_k=top_k,
                    source_types=source_types,
                )
            )
        return analyses

    def _retrieve_passages(
        self,
        user_query: str,
        top_k: int,
        source_types: list[SourceType] | None,
    ) -> list[RetrievedPassage]:
        """Retrieve passages for a given source set, with light fallback when needed."""
        if source_types:
            passages = self._retriever.search(
                query=user_query,
                top_k=top_k,
                source_types=source_types,
                search_k=max(top_k * 12, 60),
            )
            if passages:
                return passages
        return self._retriever.search(query=user_query, top_k=top_k, search_k=max(top_k * 6, 30))

    def _infer_lane_specs(self, user_query: str) -> list[tuple[str, list[SourceType]]]:
        """Infer which analysis lanes should run for the query."""
        normalized_query = user_query.lower()
        has_card_signal = any(
            signal in normalized_query
            for signal in (
                "chase",
                "sapphire",
                "amex",
                "american express",
                "mastercard",
                "visa",
                "credit card",
                "card benefit",
                "benefit guide",
                "insurance",
            )
        )
        has_airline_signal = any(
            signal in normalized_query
            for signal in (
                "delta",
                "united",
                "american airlines",
                "alaska",
                "flight",
                "bag",
                "baggage",
                "carrier",
                "airline",
                "connection",
                "missed my connection",
            )
        )

        if has_airline_signal and has_card_signal:
            return [
                ("airline", [SourceType.AIRLINE]),
                ("benefit", [SourceType.CREDIT_CARD, SourceType.INSURANCE]),
            ]
        if has_airline_signal:
            return [("airline", [SourceType.AIRLINE])]
        if has_card_signal:
            return [("benefit", [SourceType.CREDIT_CARD, SourceType.INSURANCE])]
        return []

    def _choose_primary_lane(
        self,
        user_query: str,
        lane_analyses: list[LaneAnalysis],
    ) -> tuple[LaneAnalysis, list[LaneAnalysis]]:
        """Choose the primary lane and order any secondary lanes."""
        if len(lane_analyses) == 1:
            return lane_analyses[0], []

        ranked = sorted(
            lane_analyses,
            key=lambda lane: self._lane_priority(user_query, lane),
            reverse=True,
        )
        return ranked[0], ranked[1:]

    def _lane_priority(self, user_query: str, lane: LaneAnalysis) -> tuple[int, float, int]:
        """Rank a lane based on decision strength, confidence, and specificity."""
        label = lane.eligibility.label.value
        label_rank = {
            "Eligible": 3,
            "Unclear": 2,
            "Not Eligible": 1,
        }.get(label, 0)
        specificity_rank = 1 if lane.lane_name == "benefit" and self._mentions_benefit(user_query) else 0
        return (label_rank, lane.eligibility.confidence, specificity_rank)

    def _mentions_benefit(self, user_query: str) -> bool:
        """Return whether the query explicitly mentions card or insurance benefits."""
        normalized_query = user_query.lower()
        return any(
            signal in normalized_query
            for signal in (
                "chase",
                "sapphire",
                "amex",
                "american express",
                "mastercard",
                "visa",
                "credit card",
                "insurance",
                "benefit",
            )
        )

    def _merge_passages(
        self,
        primary_passages: list[RetrievedPassage],
        secondary_passages: list[list[RetrievedPassage]],
        top_k: int,
    ) -> list[RetrievedPassage]:
        """Merge primary and secondary passages while preserving the primary lane first."""
        merged: list[RetrievedPassage] = []
        seen_chunk_ids: set[str] = set()
        for passage_group in [primary_passages, *secondary_passages]:
            for passage in passage_group:
                if passage.chunk.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(passage.chunk.chunk_id)
                merged.append(passage)
                if len(merged) >= top_k:
                    return merged
        return merged

    def _ensure_retriever_ready(self) -> None:
        """Load the persisted vector index for searching."""
        self._retriever.load_index(self._config.data_dir / "processed" / "vector_store")

    def _build_claim_plan(
        self,
        uncertainty: UncertaintyAssessment,
        eligibility: object,
        retrieved_passages: list[RetrievedPassage],
    ) -> ClaimPlan:
        """Create a simple action plan from uncertainty signals."""
        recommended_actions = [
            "Review the cited policy passages before filing the claim.",
        ]
        if uncertainty.ambiguous_policy_signals and eligibility.label.value == "Unclear":
            recommended_actions.append("Resolve ambiguous policy wording with the issuer, insurer, or airline.")
        if uncertainty.missing_fields:
            recommended_actions.append("Collect the missing claim facts before relying on the analysis.")

        required_documents: list[str] = []
        filing_steps = [
            "Confirm the applicable policy or benefits guide.",
            "Match the disruption facts to the cited policy thresholds and exclusions.",
            "Submit the claim with supporting evidence and keep copies of all records.",
        ]
        if not retrieved_passages:
            filing_steps = ["Ingest and index policy documents before analyzing the claim."]

        return ClaimPlan(
            recommended_actions=recommended_actions,
            required_documents=required_documents,
            follow_up_questions=uncertainty.follow_up_questions,
            filing_steps=filing_steps,
            priority="high" if uncertainty.missing_fields else "normal",
        )

    def _build_summary(
        self,
        primary_lane: LaneAnalysis,
        secondary_lanes: list[LaneAnalysis],
    ) -> str:
        """Generate a concise analysis summary."""
        if primary_lane.lane_name == "airline" and primary_lane.eligibility.label.value == "Eligible":
            summary = "Eligibility assessment: Potentially eligible via airline policy."
        else:
            summary = f"Eligibility assessment: {primary_lane.eligibility.label.value} via {primary_lane.lane_name} policy."
        if primary_lane.uncertainty.missing_fields:
            summary += " Critical claim details are still missing."
        elif (
            primary_lane.uncertainty.ambiguous_policy_signals
            and primary_lane.eligibility.label.value == "Unclear"
        ):
            summary += " Some policy language is ambiguous and may require manual review."
        if secondary_lanes:
            secondary_text = ", ".join(
                f"{lane.lane_name}: {lane.eligibility.label.value}" for lane in secondary_lanes
            )
            summary += f" Secondary lane results: {secondary_text}."
        return summary

    def _collect_source_types(self, passages: list[RetrievedPassage]) -> list[SourceType]:
        """Collect unique source types from retrieved passages."""
        ordered: list[SourceType] = []
        seen: set[SourceType] = set()
        for passage in passages:
            source_type = passage.metadata.source_type
            if source_type in seen:
                continue
            seen.add(source_type)
            ordered.append(source_type)
        return ordered

    def _build_claim_id(self, user_query: str) -> str:
        """Generate a stable lightweight identifier from the query text."""
        normalized = re.sub(r"[^a-z0-9]+", "-", user_query.lower()).strip("-")
        suffix = normalized[:24] or "claim"
        return f"query-{suffix}"
