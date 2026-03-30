"""OpenAI embedding helpers for retrieval workflows."""

from __future__ import annotations

import math
from typing import Sequence

from openai import OpenAI

from core.config import AppConfig


class OpenAIEmbedder:
    """Create dense vector embeddings using the OpenAI API."""

    _MAX_TOKENS_PER_REQUEST = 250_000

    def __init__(
        self,
        model: str | None = None,
        client: OpenAI | None = None,
        config: AppConfig | None = None,
    ) -> None:
        """Initialize the embedder with an OpenAI client and model name."""
        app_config = config or AppConfig.from_env()
        self._model = model or app_config.embedding_model
        self._client = client or OpenAI()

    @property
    def model(self) -> str:
        """Return the configured embedding model name."""
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple text strings in a single API request."""
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for batch in self._chunk_texts(texts):
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            embeddings.extend(list(item.embedding) for item in response.data)
        return embeddings

    def _chunk_texts(self, texts: Sequence[str]) -> list[list[str]]:
        """Split texts into conservative token-sized batches for the embeddings API."""
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for text in texts:
            estimated_tokens = self._estimate_tokens(text)
            if current_batch and current_tokens + estimated_tokens > self._MAX_TOKENS_PER_REQUEST:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += estimated_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count conservatively from character length."""
        return max(1, math.ceil(len(text) / 3))
