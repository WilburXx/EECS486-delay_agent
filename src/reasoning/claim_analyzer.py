"""Core reasoning engine for travel and baggage delay claims."""

from __future__ import annotations

from core.models import (
    ClaimStatus,
    ClaimType,
    DelayAnalysisResult,
    ExtractedClaimFacts,
    PolicyRule,
    TravelClaim,
)


class ClaimAnalyzer:
    """Apply policy rules to extracted claim facts."""

    def analyze(
        self,
        claim: TravelClaim,
        facts: ExtractedClaimFacts,
        rule: PolicyRule,
    ) -> DelayAnalysisResult:
        """Analyze a claim against the supplied rule."""
        measured_delay = (
            facts.delay_hours
            if claim.claim_type == ClaimType.TRAVEL_DELAY
            else facts.baggage_delay_hours
        )

        rationale: list[str] = [
            f"Measured delay: {measured_delay:.1f} hours.",
            f"Evidence items submitted: {facts.evidence_count}.",
            f"Policy minimum delay: {rule.minimum_delay_hours:.1f} hours.",
        ]

        if measured_delay < rule.minimum_delay_hours:
            rationale.append("Delay threshold not met.")
            return DelayAnalysisResult(
                claim_id=claim.claim_id,
                status=ClaimStatus.DENIED,
                eligible_amount=0.0,
                rationale=rationale,
                applied_rule=rule,
            )

        if facts.evidence_count < rule.required_evidence_count:
            rationale.append("More supporting documentation is required.")
            return DelayAnalysisResult(
                claim_id=claim.claim_id,
                status=ClaimStatus.REVIEW,
                eligible_amount=0.0,
                rationale=rationale,
                applied_rule=rule,
            )

        eligible_amount = min(claim.claimed_amount, rule.maximum_payout)
        rationale.append(f"Eligible amount capped at {eligible_amount:.2f} {claim.currency}.")
        return DelayAnalysisResult(
            claim_id=claim.claim_id,
            status=ClaimStatus.APPROVED,
            eligible_amount=eligible_amount,
            rationale=rationale,
            applied_rule=rule,
        )
