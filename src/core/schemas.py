"""Pydantic schemas for retrieval, extraction, and analysis workflows."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Supported source types for retrieved claim evidence."""

    AIRLINE = "airline"
    INSURANCE = "insurance"
    CREDIT_CARD = "credit_card"


class EligibilityLabel(str, Enum):
    """Eligibility outcomes produced by the analysis pipeline."""

    ELIGIBLE = "Eligible"
    NOT_ELIGIBLE = "Not Eligible"
    UNCLEAR = "Unclear"


class DocumentMetadata(BaseModel):
    """Metadata describing a source document used during analysis."""

    document_id: str
    source_type: SourceType
    title: str
    provider_name: str
    product_name: str | None = None
    source_url: str | None = None
    published_at: str | None = None
    language: str = "en"
    tags: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """A chunked section of a source document."""

    chunk_id: str
    document_id: str
    text: str
    position: int = Field(ge=0)
    page_number: int | None = Field(default=None, ge=1)
    section_title: str | None = None
    token_count: int | None = Field(default=None, ge=0)


class RetrievedPassage(BaseModel):
    """A retrieved passage and its relevance information."""

    chunk: Chunk
    metadata: DocumentMetadata
    relevance_score: float = Field(ge=0.0)
    query: str
    citation: str
    rationale: str | None = None


class FloatRequirementField(BaseModel):
    """A numeric extracted value with supporting evidence."""

    value: float | None = Field(default=None, ge=0.0)
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class StringRequirementField(BaseModel):
    """A text extracted value with supporting evidence."""

    value: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class StringListRequirementField(BaseModel):
    """A list extracted value with supporting evidence."""

    value: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class ExtractedRequirements(BaseModel):
    """Normalized policy requirements extracted from source materials."""

    minimum_delay_threshold_hours: FloatRequirementField = Field(default_factory=FloatRequirementField)
    covered_expense_categories: StringListRequirementField = Field(default_factory=StringListRequirementField)
    reimbursement_cap: FloatRequirementField = Field(default_factory=FloatRequirementField)
    reimbursement_currency: StringRequirementField = Field(default_factory=StringRequirementField)
    required_documentation: StringListRequirementField = Field(default_factory=StringListRequirementField)
    filing_deadline: StringRequirementField = Field(default_factory=StringRequirementField)
    exclusions: StringListRequirementField = Field(default_factory=StringListRequirementField)


class EligibilityResult(BaseModel):
    """Eligibility determination for a claim under a single policy."""

    label: EligibilityLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: list[str] = Field(default_factory=list)
    qualifying_facts: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    supporting_chunk_ids: list[str] = Field(default_factory=list)


class ClaimPlan(BaseModel):
    """Action plan for preparing or filing a claim."""

    recommended_actions: list[str] = Field(default_factory=list)
    required_documents: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    filing_steps: list[str] = Field(default_factory=list)
    priority: str = "normal"


class AnalysisResponse(BaseModel):
    """Top-level structured response for a claim analysis request."""

    claim_id: str
    source_types_considered: list[SourceType] = Field(default_factory=list)
    retrieved_passages: list[RetrievedPassage] = Field(default_factory=list)
    extracted_requirements: ExtractedRequirements
    eligibility: EligibilityResult
    claim_plan: ClaimPlan
    summary: str
