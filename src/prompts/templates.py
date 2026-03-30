"""Prompt templates for retrieval-grounded policy extraction."""

from __future__ import annotations

from core.schemas import RetrievedPassage


POLICY_EXTRACTION_SYSTEM_PROMPT = """
You extract travel delay and baggage delay policy requirements from retrieved source text.

Rules:
- Return JSON only.
- Use only facts explicitly supported by the retrieved passages.
- Do not infer, estimate, normalize, or hallucinate missing values.
- If a field is not supported, use null for scalar values and [] for list values.
- Every populated field must include evidence_chunk_ids containing only chunk IDs from the provided passages.
- Do not cite a chunk unless it directly supports the field.
- Keep reimbursement_currency null unless the currency is explicitly stated.

Return a JSON object with exactly this structure:
{
  "minimum_delay_threshold_hours": {"value": null, "evidence_chunk_ids": []},
  "covered_expense_categories": {"value": [], "evidence_chunk_ids": []},
  "reimbursement_cap": {"value": null, "evidence_chunk_ids": []},
  "reimbursement_currency": {"value": null, "evidence_chunk_ids": []},
  "required_documentation": {"value": [], "evidence_chunk_ids": []},
  "filing_deadline": {"value": null, "evidence_chunk_ids": []},
  "exclusions": {"value": [], "evidence_chunk_ids": []}
}
""".strip()


def build_policy_extraction_prompt(user_query: str, passages: list[RetrievedPassage]) -> str:
    """Build the user prompt for policy extraction."""
    rendered_passages = "\n\n".join(_format_passage(passage) for passage in passages)
    return (
        f"User query:\n{user_query.strip()}\n\n"
        "Retrieved passages:\n"
        f"{rendered_passages}\n\n"
        "Extract policy requirements into the required JSON schema."
    )


def _format_passage(passage: RetrievedPassage) -> str:
    """Render a retrieved passage for the extraction prompt."""
    return (
        f"Chunk ID: {passage.chunk.chunk_id}\n"
        f"Citation: {passage.citation}\n"
        f"Source Type: {passage.metadata.source_type.value}\n"
        f"Title: {passage.metadata.title}\n"
        f"Provider: {passage.metadata.provider_name}\n"
        f"Text:\n{passage.chunk.text}"
    )
