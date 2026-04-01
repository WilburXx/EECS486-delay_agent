"""Eligibility reasoning for retrieved policy requirements."""

from __future__ import annotations

import re

from core.schemas import EligibilityLabel, EligibilityResult, ExtractedRequirements, RetrievedPassage


class EligibilityEvaluator:
    """Classify eligibility conservatively from extracted rules and user-provided facts."""

    _delay_patterns = [
        re.compile(r"\bdelay(?:ed)?(?:\s+(?:for|of))?\s+(\d+(?:\.\d+)?)\s*hours?\b", re.IGNORECASE),
        re.compile(r"\b(\d+(?:\.\d+)?)\s*hours?\s+(?:delay|delayed)\b", re.IGNORECASE),
        re.compile(r"\b(?:more than|over|at least)\s+(\d+(?:\.\d+)?)\s*hours?\b", re.IGNORECASE),
        re.compile(r"\b(\d+(?:\.\d+)?)\s*hr[s]?\b", re.IGNORECASE),
        re.compile(r"\ba full day\b|\bfull day\b", re.IGNORECASE),
        re.compile(r"\bovernight\b", re.IGNORECASE),
    ]

    def evaluate(
        self,
        user_query: str,
        extracted_requirements: ExtractedRequirements,
        retrieved_passages: list[RetrievedPassage],
    ) -> EligibilityResult:
        """Determine whether the user appears eligible under explicitly extracted rules."""
        delay_hours = self._extract_delay_hours(user_query)
        rationale: list[str] = []
        qualifying_facts: list[str] = []
        missing_information: list[str] = []
        supporting_chunk_ids: list[str] = []

        threshold_field = extracted_requirements.minimum_delay_threshold_hours
        exclusions_field = extracted_requirements.exclusions
        filing_deadline_field = extracted_requirements.filing_deadline
        documentation_field = extracted_requirements.required_documentation

        if threshold_field.value is not None:
            supporting_chunk_ids.extend(threshold_field.evidence_chunk_ids)
            rationale.append(
                f"Retrieved policy threshold is {threshold_field.value:.1f} hours."
            )
        else:
            missing_information.append("minimum delay threshold")

        if delay_hours is not None:
            qualifying_facts.append(f"User query states a delay of {delay_hours:.1f} hours.")
        else:
            missing_information.append("claimed delay duration")

        if exclusions_field.value:
            supporting_chunk_ids.extend(exclusions_field.evidence_chunk_ids)
            rationale.append(
                "Policy exclusions were extracted and should be reviewed for disqualifying conditions."
            )

        if filing_deadline_field.value:
            supporting_chunk_ids.extend(filing_deadline_field.evidence_chunk_ids)
            rationale.append(f"Filing deadline reference found: {filing_deadline_field.value}.")

        if documentation_field.value:
            supporting_chunk_ids.extend(documentation_field.evidence_chunk_ids)
            rationale.append(
                f"Required documentation identified: {', '.join(documentation_field.value)}."
            )
        else:
            missing_information.append("required documentation")

        supporting_chunk_ids = self._dedupe_preserve_order(supporting_chunk_ids)

        if delay_hours is not None and threshold_field.value is not None:
            if delay_hours >= threshold_field.value:
                rationale.append(
                    f"Claimed delay meets or exceeds the explicit threshold ({delay_hours:.1f} >= {threshold_field.value:.1f})."
                )
                qualifying_facts.append("Explicit delay threshold comparison supports eligibility.")
                label = EligibilityLabel.ELIGIBLE
                confidence = 0.8 if supporting_chunk_ids else 0.7
            else:
                rationale.append(
                    f"Claimed delay is below the explicit threshold ({delay_hours:.1f} < {threshold_field.value:.1f})."
                )
                qualifying_facts.append("Explicit delay threshold comparison weighs against eligibility.")
                label = EligibilityLabel.NOT_ELIGIBLE
                confidence = 0.85 if supporting_chunk_ids else 0.75
        else:
            rationale.append(
                "Eligibility cannot be determined confidently because explicit policy requirements or user facts are missing."
            )
            label = EligibilityLabel.UNCLEAR
            confidence = 0.45

        if missing_information and label is not EligibilityLabel.NOT_ELIGIBLE:
            rationale.append(
                "Required information is missing, so the conservative classification is Unclear."
            )
            label = EligibilityLabel.UNCLEAR
            confidence = min(confidence, 0.5)

        if not retrieved_passages:
            rationale.append("No retrieved policy passages were provided.")
            label = EligibilityLabel.UNCLEAR
            confidence = 0.2

        return EligibilityResult(
            label=label,
            confidence=confidence,
            rationale=rationale,
            qualifying_facts=qualifying_facts,
            missing_information=self._dedupe_preserve_order(missing_information),
            supporting_chunk_ids=supporting_chunk_ids,
        )

    def _extract_delay_hours(self, user_query: str) -> float | None:
        """Extract an explicitly stated delay duration from the user query."""
        for pattern in self._delay_patterns:
            match = pattern.search(user_query)
            if match:
                if match.lastindex:
                    return float(match.group(1))
                phrase = match.group(0).lower()
                if "full day" in phrase:
                    return 24.0
                if "overnight" in phrase:
                    return 8.0
        return None

    def _dedupe_preserve_order(self, values: list[str]) -> list[str]:
        """Deduplicate a list while preserving first-seen order."""
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered
