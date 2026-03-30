"""Tests for the DelayAgent service layer."""

from __future__ import annotations

from app.service import DelayAgentService
from core.models import ClaimStatus, ClaimType, TravelClaim


def test_travel_delay_claim_is_approved_when_threshold_and_evidence_are_met() -> None:
    """Travel delay claims should approve when policy requirements are met."""
    service = DelayAgentService()
    claim = TravelClaim(
        claim_id="CLM-2001",
        claim_type=ClaimType.TRAVEL_DELAY,
        traveler_name="Jamie Doe",
        carrier="SkyBridge",
        delay_hours=4.0,
        claimed_amount=250.0,
        evidence=["receipt_meal.pdf"],
    )

    result = service.analyze_claim(claim)

    assert result.status is ClaimStatus.APPROVED
    assert result.eligible_amount == 250.0


def test_baggage_delay_claim_is_marked_for_review_when_evidence_is_missing() -> None:
    """Baggage delay claims should require review without enough documentation."""
    service = DelayAgentService()
    claim = TravelClaim(
        claim_id="CLM-2002",
        claim_type=ClaimType.BAGGAGE_DELAY,
        traveler_name="Taylor Doe",
        carrier="SkyBridge",
        baggage_delay_hours=10.0,
        claimed_amount=180.0,
        evidence=["bag_tag.jpg"],
    )

    result = service.analyze_claim(claim)

    assert result.status is ClaimStatus.REVIEW
    assert result.eligible_amount == 0.0
