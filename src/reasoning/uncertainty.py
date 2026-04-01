"""Uncertainty detection for missing facts and ambiguous policy language."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from core.schemas import RetrievedPassage


class MissingField(BaseModel):
    """A critical user fact that is missing from the current claim context."""

    field_name: str
    reason: str
    follow_up_question: str


class AmbiguousPolicySignal(BaseModel):
    """An ambiguous policy phrase found in retrieved material."""

    phrase: str
    chunk_id: str
    citation: str
    explanation: str


class UncertaintyAssessment(BaseModel):
    """Structured uncertainty output for downstream clarification workflows."""

    missing_fields: list[MissingField] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    ambiguous_policy_signals: list[AmbiguousPolicySignal] = Field(default_factory=list)


class UncertaintyDetector:
    """Detect missing user facts and ambiguity in retrieved policy passages."""

    _airline_patterns = [
        re.compile(r"\bdelta\b", re.IGNORECASE),
        re.compile(r"\bunited\b", re.IGNORECASE),
        re.compile(r"\bamerican airlines\b|\bamerican air(?:lines)?\b|\baa\b", re.IGNORECASE),
        re.compile(r"\balaska\b|\balaska airlines\b", re.IGNORECASE),
        re.compile(r"\bsouthwest\b", re.IGNORECASE),
        re.compile(r"\bjetblue\b", re.IGNORECASE),
        re.compile(r"\b(?:airline|carrier)\s+(?:was|is)\s+([A-Z][A-Za-z0-9&.\- ]+)", re.IGNORECASE),
        re.compile(r"\b(?:on|with)\s+([A-Z][A-Za-z0-9&.\- ]+(?:air|airlines|airways))\b", re.IGNORECASE),
    ]
    _delay_patterns = [
        re.compile(r"\bdelay(?:ed)?(?:\s+(?:for|of))?\s+\d+(?:\.\d+)?\s*hours?\b", re.IGNORECASE),
        re.compile(r"\b\d+(?:\.\d+)?\s*hours?\s+(?:delay|delayed)\b", re.IGNORECASE),
        re.compile(r"\b(?:more than|over|at least)\s+\d+(?:\.\d+)?\s*hours?\b", re.IGNORECASE),
        re.compile(r"\b\d+(?:\.\d+)?\s*hr[s]?\b", re.IGNORECASE),
        re.compile(r"\ba full day\b|\bfull day\b", re.IGNORECASE),
        re.compile(r"\bovernight\b", re.IGNORECASE),
    ]
    _disruption_patterns = [
        re.compile(r"\bflight delay\b", re.IGNORECASE),
        re.compile(r"\bflight was delayed\b", re.IGNORECASE),
        re.compile(r"\bflight delayed\b", re.IGNORECASE),
        re.compile(r"\bflight\b.*\bdelayed\b", re.IGNORECASE),
        re.compile(r"\bbaggage delay\b", re.IGNORECASE),
        re.compile(r"\blost baggage\b", re.IGNORECASE),
        re.compile(r"\bdelayed baggage\b", re.IGNORECASE),
        re.compile(r"\bchecked bag\b", re.IGNORECASE),
        re.compile(r"\bcancel(?:led|ed|ation)\b", re.IGNORECASE),
        re.compile(r"\bmissed connection\b", re.IGNORECASE),
    ]
    _payment_patterns = [
        re.compile(r"\bpaid with\b", re.IGNORECASE),
        re.compile(r"\bcredit card\b", re.IGNORECASE),
        re.compile(r"\bvisa\b|\bmastercard\b|\bamex\b|\bamerican express\b", re.IGNORECASE),
        re.compile(r"\bchase sapphire preferred\b", re.IGNORECASE),
        re.compile(r"\bchase sapphire reserve\b", re.IGNORECASE),
        re.compile(r"\bchase\b", re.IGNORECASE),
    ]
    _policy_patterns = [
        re.compile(r"\bpolicy\b", re.IGNORECASE),
        re.compile(r"\bcertificate of insurance\b", re.IGNORECASE),
        re.compile(r"\bbenefit guide\b", re.IGNORECASE),
        re.compile(r"\btrip delay benefit\b", re.IGNORECASE),
        re.compile(r"\bguide to benefits\b", re.IGNORECASE),
        re.compile(r"\bchase sapphire preferred\b", re.IGNORECASE),
        re.compile(r"\bamerican express\b", re.IGNORECASE),
        re.compile(r"\bmastercard\b", re.IGNORECASE),
    ]
    _receipt_patterns = [
        re.compile(r"\breceipt(?:s)?\b", re.IGNORECASE),
        re.compile(r"\binvoice(?:s)?\b", re.IGNORECASE),
        re.compile(r"\bproof of purchase\b", re.IGNORECASE),
    ]
    _ambiguous_phrases = [
        re.compile(r"\bmay be covered\b", re.IGNORECASE),
        re.compile(r"\bsubject to\b", re.IGNORECASE),
        re.compile(r"\bat our discretion\b", re.IGNORECASE),
        re.compile(r"\bwhere applicable\b", re.IGNORECASE),
        re.compile(r"\breasonable(?: and necessary)?\b", re.IGNORECASE),
        re.compile(r"\bup to\b", re.IGNORECASE),
        re.compile(r"\bunless otherwise\b", re.IGNORECASE),
        re.compile(r"\bif approved\b", re.IGNORECASE),
    ]

    def assess(
        self,
        user_query: str,
        retrieved_passages: list[RetrievedPassage],
    ) -> UncertaintyAssessment:
        """Return missing critical fields and policy ambiguities without guessing."""
        missing_fields = self._detect_missing_fields(user_query, retrieved_passages)
        ambiguous_signals = self._detect_ambiguous_policy_language(retrieved_passages)
        follow_up_questions = [item.follow_up_question for item in missing_fields]

        if ambiguous_signals and self._needs_policy_disambiguation(user_query, retrieved_passages):
            follow_up_questions.append(
                "Which exact policy or benefits guide should we rely on when the wording is ambiguous?"
            )

        return UncertaintyAssessment(
            missing_fields=missing_fields,
            follow_up_questions=self._dedupe_preserve_order(follow_up_questions),
            ambiguous_policy_signals=ambiguous_signals,
        )

    def _detect_missing_fields(self, user_query: str) -> list[MissingField]:
        """Identify critical missing claim facts from the user query."""
        missing: list[MissingField] = []

        if not self._matches_any(user_query, self._airline_patterns):
            missing.append(
                MissingField(
                    field_name="airline",
                    reason="The airline or carrier is not explicitly identified in the user query.",
                    follow_up_question="Which airline or carrier operated the disrupted trip?",
                )
            )

        if not self._matches_any(user_query, self._delay_patterns):
            missing.append(
                MissingField(
                    field_name="delay_duration",
                    reason="The delay duration is not stated explicitly enough for threshold comparison.",
                    follow_up_question="How many hours was the flight or baggage delayed?",
                )
            )

        if not self._matches_any(user_query, self._disruption_patterns):
            missing.append(
                MissingField(
                    field_name="disruption_type",
                    reason="The disruption type is not clearly stated.",
                    follow_up_question="Was this a flight delay, baggage delay, cancellation, or another disruption?",
                )
            )

        if not self._matches_any(user_query, self._payment_patterns):
            missing.append(
                MissingField(
                    field_name="payment_method",
                    reason="The payment method is not specified, which matters for credit-card benefits.",
                    follow_up_question="How did you pay for the trip, and which card or method was used?",
                )
            )

        if not self._matches_any(user_query, self._policy_patterns):
            missing.append(
                MissingField(
                    field_name="policy_identity",
                    reason="The relevant policy, benefits guide, or insurer is not clearly identified.",
                    follow_up_question="Which policy, insurance plan, or card benefits guide should be checked?",
                )
            )

        if not self._matches_any(user_query, self._receipt_patterns):
            missing.append(
                MissingField(
                    field_name="receipts",
                    reason="The query does not say whether receipts or proof of purchase are available.",
                    follow_up_question="Do you have receipts or other proof of purchase for the claimed expenses?",
                )
            )

        return missing

    def _detect_missing_fields(
        self,
        user_query: str,
        retrieved_passages: list[RetrievedPassage],
    ) -> list[MissingField]:
        """Identify critical missing claim facts from the user query and retrieved context."""
        missing: list[MissingField] = []

        if not self._matches_any(user_query, self._airline_patterns):
            missing.append(
                MissingField(
                    field_name="airline",
                    reason="The airline or carrier is not explicitly identified in the user query.",
                    follow_up_question="Which airline or carrier operated the disrupted trip?",
                )
            )

        if not self._matches_any(user_query, self._delay_patterns):
            missing.append(
                MissingField(
                    field_name="delay_duration",
                    reason="The delay duration is not stated explicitly enough for threshold comparison.",
                    follow_up_question="How many hours was the flight or baggage delayed?",
                )
            )

        if not self._matches_any(user_query, self._disruption_patterns):
            missing.append(
                MissingField(
                    field_name="disruption_type",
                    reason="The disruption type is not clearly stated.",
                    follow_up_question="Was this a flight delay, baggage delay, cancellation, or another disruption?",
                )
            )

        if not self._matches_any(user_query, self._payment_patterns):
            missing.append(
                MissingField(
                    field_name="payment_method",
                    reason="The payment method is not specified, which matters for credit-card benefits.",
                    follow_up_question="How did you pay for the trip, and which card or method was used?",
                )
            )

        if not self._has_policy_identity(user_query, retrieved_passages):
            missing.append(
                MissingField(
                    field_name="policy_identity",
                    reason="The relevant policy, benefits guide, or insurer is not clearly identified.",
                    follow_up_question="Which policy, insurance plan, or card benefits guide should be checked?",
                )
            )

        if not self._matches_any(user_query, self._receipt_patterns):
            missing.append(
                MissingField(
                    field_name="receipts",
                    reason="The query does not say whether receipts or proof of purchase are available.",
                    follow_up_question="Do you have receipts or other proof of purchase for the claimed expenses?",
                )
            )

        return missing

    def _has_policy_identity(
        self,
        user_query: str,
        retrieved_passages: list[RetrievedPassage],
    ) -> bool:
        """Determine whether the applicable policy identity is already reasonably clear."""
        if self._matches_any(user_query, self._policy_patterns):
            return True

        if not retrieved_passages:
            return False

        document_ids = {passage.metadata.document_id for passage in retrieved_passages}
        provider_names = {passage.metadata.provider_name.lower() for passage in retrieved_passages}
        titles = {passage.metadata.title.lower() for passage in retrieved_passages}
        normalized_query = user_query.lower()

        if len(document_ids) == 1:
            return True

        return any(provider in normalized_query for provider in provider_names) or any(
            title in normalized_query for title in titles
        )

    def _needs_policy_disambiguation(
        self,
        user_query: str,
        retrieved_passages: list[RetrievedPassage],
    ) -> bool:
        """Return whether ambiguity should trigger a policy-identity follow-up."""
        return not self._has_policy_identity(user_query, retrieved_passages)

    def _detect_ambiguous_policy_language(
        self,
        retrieved_passages: list[RetrievedPassage],
    ) -> list[AmbiguousPolicySignal]:
        """Find ambiguous policy phrases that may require clarification."""
        signals: list[AmbiguousPolicySignal] = []

        for passage in retrieved_passages:
            text = passage.chunk.text
            for pattern in self._ambiguous_phrases:
                for match in pattern.finditer(text):
                    signals.append(
                        AmbiguousPolicySignal(
                            phrase=match.group(0),
                            chunk_id=passage.chunk.chunk_id,
                            citation=passage.citation,
                            explanation=(
                                "This wording is conditional or discretionary and may need manual review "
                                "before relying on it for a final eligibility decision."
                            ),
                        )
                    )

        return self._dedupe_ambiguous_signals(signals)

    def _matches_any(self, text: str, patterns: list[re.Pattern[str]]) -> bool:
        """Return whether any pattern matches the given text."""
        return any(pattern.search(text) for pattern in patterns)

    def _dedupe_preserve_order(self, values: list[str]) -> list[str]:
        """Deduplicate strings while preserving order."""
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _dedupe_ambiguous_signals(
        self,
        signals: list[AmbiguousPolicySignal],
    ) -> list[AmbiguousPolicySignal]:
        """Deduplicate ambiguous signals by phrase and chunk."""
        seen: set[tuple[str, str]] = set()
        ordered: list[AmbiguousPolicySignal] = []
        for signal in signals:
            key = (signal.phrase.lower(), signal.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(signal)
        return ordered
