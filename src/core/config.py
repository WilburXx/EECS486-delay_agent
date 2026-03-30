"""Application configuration for DelayAgent."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv(override=True)


class AppConfig(BaseModel):
    """Runtime configuration for the application."""

    app_name: str = Field(default="DelayAgent")
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    data_dir: Path = Field(default=Path("./data"))
    embedding_model: str = Field(default="text-embedding-3-small")
    extraction_model: str = Field(default="gpt-4.1-mini")

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build configuration from environment variables."""
        import os

        return cls(
            environment=os.getenv("DELAY_AGENT_ENV", "development"),
            log_level=os.getenv("DELAY_AGENT_LOG_LEVEL", "INFO"),
            data_dir=Path(os.getenv("DELAY_AGENT_DATA_DIR", "./data")),
            embedding_model=os.getenv("DELAY_AGENT_EMBEDDING_MODEL", "text-embedding-3-small"),
            extraction_model=os.getenv("DELAY_AGENT_EXTRACTION_MODEL", "gpt-4.1-mini"),
        )
