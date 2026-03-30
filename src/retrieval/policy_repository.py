"""Policy retrieval logic for claim adjudication."""

from __future__ import annotations

from core.models import ClaimType, PolicyRule


class PolicyRepository:
    """Provide policy rules for supported claim types."""

    def __init__(self) -> None:
        """Initialize an in-memory rule catalog."""
        self._rules: dict[ClaimType, PolicyRule] = {
            ClaimType.TRAVEL_DELAY: PolicyRule(
                claim_type=ClaimType.TRAVEL_DELAY,
                minimum_delay_hours=3.0,
                maximum_payout=500.0,
                required_evidence_count=1,
            ),
            ClaimType.BAGGAGE_DELAY: PolicyRule(
                claim_type=ClaimType.BAGGAGE_DELAY,
                minimum_delay_hours=6.0,
                maximum_payout=300.0,
                required_evidence_count=2,
            ),
        }

    def get_rule(self, claim_type: ClaimType) -> PolicyRule:
        """Return the applicable rule for a claim type."""
        return self._rules[claim_type]
