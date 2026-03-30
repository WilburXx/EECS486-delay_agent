"""Helpers for loading raw claim data into domain models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.models import TravelClaim


class ClaimLoader:
    """Load travel claims from supported input formats."""

    def load_from_dict(self, payload: dict[str, Any]) -> TravelClaim:
        """Convert a dictionary payload into a validated claim model."""
        return TravelClaim.model_validate(payload)

    def load_from_json(self, path: Path) -> TravelClaim:
        """Load a claim from a JSON file."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self.load_from_dict(payload)
