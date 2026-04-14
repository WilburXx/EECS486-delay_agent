"""LLM-backed policy requirement extraction."""

from __future__ import annotations

from openai import OpenAI

from core.config import AppConfig
from core.schemas import ExtractedRequirements, RetrievedPassage
from extraction.parser import PolicyExtractionParser
from prompts.templates import build_policy_extraction_prompt, POLICY_EXTRACTION_SYSTEM_PROMPT


class PolicyRequirementExtractor:
    """Extract structured policy requirements from retrieved passages."""

    def __init__(
        self,
        model: str | None = None,
        client: OpenAI | None = None,
        parser: PolicyExtractionParser | None = None,
        config: AppConfig | None = None,
    ) -> None:
        """Initialize the extractor with an OpenAI client and output parser."""
        app_config = config or AppConfig.from_env()
        self._model = model or app_config.extraction_model
        self._client = client or OpenAI()
        self._parser = parser or PolicyExtractionParser()

    def extract(
        self,
        user_query: str,
        passages: list[RetrievedPassage],
    ) -> ExtractedRequirements:
        """Extract supported policy requirements from retrieved evidence."""
        if not passages:
            return self._parser.empty()
        response = self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": POLICY_EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_policy_extraction_prompt(user_query=user_query, passages=passages),
                },
            ],
        )
        raw_output = getattr(response, "output_text", "") or ""
        return self._parser.parse(raw_output=raw_output, passages=passages)
