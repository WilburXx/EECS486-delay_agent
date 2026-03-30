"""Executable entry point for DelayAgent."""

from __future__ import annotations

import json

from app.service import DelayAgentService
from core.config import AppConfig
from core.models import ClaimType, TravelClaim
from utils.logging import configure_logging


def build_sample_claim() -> TravelClaim:
    """Create a sample claim for local smoke testing."""
    return TravelClaim(
        claim_id="CLM-1001",
        claim_type=ClaimType.TRAVEL_DELAY,
        traveler_name="Alex Traveler",
        carrier="Example Air",
        delay_hours=5.5,
        claimed_amount=420.0,
        narrative="Flight delay caused overnight accommodation and meal expenses.",
        evidence=["boarding_pass.pdf", "receipt_hotel.pdf"],
    )


def main() -> None:
    """Run the application with a sample claim payload."""
    config = AppConfig.from_env()
    configure_logging(config.log_level)

    service = DelayAgentService()
    result = service.analyze_claim(build_sample_claim())
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
