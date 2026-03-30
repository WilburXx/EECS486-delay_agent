"""FastAPI application exposing ingestion and claim analysis endpoints."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from app.service import ClaimAnalysisService, DocumentIngestionService
from core.config import AppConfig
from core.schemas import AnalysisResponse


class AnalyzeClaimRequest(BaseModel):
    """Input payload for the claim analysis endpoint."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)


class HealthResponse(BaseModel):
    """Health check response payload."""

    status: str


class IngestResponse(BaseModel):
    """Summary response for the ingestion endpoint."""

    documents_ingested: int
    chunks_indexed: int
    vector_store_path: str
    documents: list[dict[str, str]]


def get_config() -> AppConfig:
    """Provide the application configuration."""
    return AppConfig.from_env()


def get_ingestion_service(config: AppConfig = Depends(get_config)) -> DocumentIngestionService:
    """Provide the document ingestion service."""
    return DocumentIngestionService(config=config)


def get_claim_analysis_service(config: AppConfig = Depends(get_config)) -> ClaimAnalysisService:
    """Provide the claim analysis service."""
    return ClaimAnalysisService(config=config)


app = FastAPI(title="DelayAgent API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a simple health status."""
    return HealthResponse(status="ok")


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents(
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    """Ingest all PDFs from the raw data directory and rebuild the vector index."""
    result = ingestion_service.ingest_all_pdfs()
    return IngestResponse.model_validate(result)


@app.post("/analyze_claim", response_model=AnalysisResponse)
def analyze_claim(
    request: AnalyzeClaimRequest,
    claim_analysis_service: ClaimAnalysisService = Depends(get_claim_analysis_service),
) -> AnalysisResponse:
    """Analyze a natural-language disruption query."""
    return claim_analysis_service.analyze_claim(
        user_query=request.query,
        top_k=request.top_k,
    )
