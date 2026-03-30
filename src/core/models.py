"""Domain models used across the DelayAgent pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    """Supported claim categories."""

    TRAVEL_DELAY = "travel_delay"
    BAGGAGE_DELAY = "baggage_delay"


class ClaimStatus(str, Enum):
    """High-level claim outcome."""

    APPROVED = "approved"
    REVIEW = "review"
    DENIED = "denied"


class TravelClaim(BaseModel):
    """Canonical claim record used by the service layer."""

    claim_id: str
    claim_type: ClaimType
    traveler_name: str
    carrier: str
    delay_hours: float = Field(default=0.0, ge=0)
    baggage_delay_hours: float = Field(default=0.0, ge=0)
    claimed_amount: float = Field(default=0.0, ge=0)
    currency: str = Field(default="USD")
    narrative: str = Field(default="")
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedClaimFacts(BaseModel):
    """Normalized facts extracted from a raw claim."""

    delay_hours: float = Field(default=0.0, ge=0)
    baggage_delay_hours: float = Field(default=0.0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    has_receipts: bool = Field(default=False)


class PolicyRule(BaseModel):
    """A simplified policy rule used during eligibility evaluation."""

    claim_type: ClaimType
    minimum_delay_hours: float = Field(default=0.0, ge=0)
    maximum_payout: float = Field(default=0.0, ge=0)
    required_evidence_count: int = Field(default=0, ge=0)


class DelayAnalysisResult(BaseModel):
    """Result of the claim analysis workflow."""

    claim_id: str
    status: ClaimStatus
    eligible_amount: float = Field(default=0.0, ge=0)
    rationale: list[str] = Field(default_factory=list)
    applied_rule: PolicyRule | None = Field(default=None)
