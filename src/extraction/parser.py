"""Parsing and validation helpers for policy extraction outputs."""

from __future__ import annotations

import json
import re

from core.schemas import (
    ExtractedRequirements,
    FloatRequirementField,
    RetrievedPassage,
    StringListRequirementField,
    StringRequirementField,
)


class PolicyExtractionParser:
    """Parse and validate structured extraction output from an LLM."""

    def parse(
        self,
        raw_output: str,
        passages: list[RetrievedPassage],
    ) -> ExtractedRequirements:
        """Parse model output into validated requirements tied to known chunk IDs."""
        payload = self._load_json(raw_output)
        payload = self._normalize_payload(payload)
        extracted = ExtractedRequirements.model_validate(payload)
        valid_chunk_ids = {passage.chunk.chunk_id for passage in passages}
        return self._sanitize_requirements(extracted, valid_chunk_ids)

    def empty(self) -> ExtractedRequirements:
        """Return an empty requirements object for unsupported cases."""
        return ExtractedRequirements()

    def _load_json(self, raw_output: str) -> dict[str, object]:
        """Load JSON from a raw model response, including fenced blocks."""
        text = raw_output.strip()
        fenced_match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
        candidate = fenced_match.group(1) if fenced_match else text

        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError as error:
            raise ValueError("LLM extraction output was not valid JSON.") from error

        if not isinstance(loaded, dict):
            raise ValueError("LLM extraction output must be a JSON object.")

        return loaded

    def _normalize_payload(self, payload: dict[str, object]) -> dict[str, object]:
        """Coerce common scalar mismatches from model output into schema-friendly values."""
        normalized = dict(payload)
        for field_name in ("reimbursement_currency", "filing_deadline"):
            field_value = normalized.get(field_name)
            if isinstance(field_value, dict):
                value = field_value.get("value")
                if value is not None and not isinstance(value, str):
                    field_value = dict(field_value)
                    field_value["value"] = str(value)
                    normalized[field_name] = field_value
        return normalized

    def _sanitize_requirements(
        self,
        extracted: ExtractedRequirements,
        valid_chunk_ids: set[str],
    ) -> ExtractedRequirements:
        """Drop unsupported values and invalid evidence references."""
        requirements = extracted.model_copy(deep=True)

        requirements.minimum_delay_threshold_hours = self._sanitize_float_field(
            requirements.minimum_delay_threshold_hours,
            valid_chunk_ids,
        )
        requirements.covered_expense_categories = self._sanitize_list_field(
            requirements.covered_expense_categories,
            valid_chunk_ids,
        )
        requirements.reimbursement_cap = self._sanitize_float_field(
            requirements.reimbursement_cap,
            valid_chunk_ids,
        )
        requirements.reimbursement_currency = self._sanitize_string_field(
            requirements.reimbursement_currency,
            valid_chunk_ids,
        )
        requirements.required_documentation = self._sanitize_list_field(
            requirements.required_documentation,
            valid_chunk_ids,
        )
        requirements.filing_deadline = self._sanitize_string_field(
            requirements.filing_deadline,
            valid_chunk_ids,
        )
        requirements.exclusions = self._sanitize_list_field(
            requirements.exclusions,
            valid_chunk_ids,
        )

        return requirements

    def _sanitize_float_field(
        self,
        field: FloatRequirementField,
        valid_chunk_ids: set[str],
    ) -> FloatRequirementField:
        """Retain numeric values only when supported by valid evidence."""
        filtered_ids = self._filter_chunk_ids(field.evidence_chunk_ids, valid_chunk_ids)
        if field.value is None or not filtered_ids:
            return FloatRequirementField()
        return FloatRequirementField(value=field.value, evidence_chunk_ids=filtered_ids)

    def _sanitize_string_field(
        self,
        field: StringRequirementField,
        valid_chunk_ids: set[str],
    ) -> StringRequirementField:
        """Retain string values only when supported by valid evidence."""
        filtered_ids = self._filter_chunk_ids(field.evidence_chunk_ids, valid_chunk_ids)
        if not field.value or not filtered_ids:
            return StringRequirementField()
        return StringRequirementField(value=field.value.strip(), evidence_chunk_ids=filtered_ids)

    def _sanitize_list_field(
        self,
        field: StringListRequirementField,
        valid_chunk_ids: set[str],
    ) -> StringListRequirementField:
        """Retain list values only when supported by valid evidence."""
        filtered_ids = self._filter_chunk_ids(field.evidence_chunk_ids, valid_chunk_ids)
        cleaned_values = [value.strip() for value in field.value if value and value.strip()]
        if not cleaned_values or not filtered_ids:
            return StringListRequirementField()
        return StringListRequirementField(value=cleaned_values, evidence_chunk_ids=filtered_ids)

    def _filter_chunk_ids(self, chunk_ids: list[str], valid_chunk_ids: set[str]) -> list[str]:
        """Keep only evidence chunk IDs present in the retrieved passages."""
        return [chunk_id for chunk_id in chunk_ids if chunk_id in valid_chunk_ids]


class ClaimFactExtractor:
    """Backward-compatible deterministic fact extractor for claim records."""

    def extract(self, claim: object) -> object:
        """Preserve the previous parser module contract for existing imports."""
        try:
            from core.models import ExtractedClaimFacts, TravelClaim
        except ImportError as error:
            raise RuntimeError("Claim fact extraction models are unavailable.") from error

        if not isinstance(claim, TravelClaim):
            raise TypeError("claim must be an instance of TravelClaim")

        receipts = any("receipt" in item.lower() for item in claim.evidence)
        return ExtractedClaimFacts(
            delay_hours=claim.delay_hours,
            baggage_delay_hours=claim.baggage_delay_hours,
            evidence_count=len(claim.evidence),
            has_receipts=receipts,
        )
