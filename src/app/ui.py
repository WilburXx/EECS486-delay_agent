"""Streamlit user interface for DelayAgent."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.service import ClaimAnalysisService
from core.config import AppConfig
from core.schemas import AnalysisResponse, RetrievedPassage, StringListRequirementField, StringRequirementField
from ingestion.ingest_pipeline import IngestPipeline
from retrieval.retriever import PassageRetriever


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title="DelayAgent", page_icon="Delay", layout="wide")
    st.title("DelayAgent")
    st.caption("Analyze travel delay and baggage delay claims with retrieved policy evidence.")
    if "analysis_response" not in st.session_state:
        st.session_state.analysis_response = None
    if "analysis_query" not in st.session_state:
        st.session_state.analysis_query = ""
    if "follow_up_answer" not in st.session_state:
        st.session_state.follow_up_answer = ""

    config = AppConfig.from_env()
    _ensure_data_dirs(config)

    with st.sidebar:
        st.subheader("Inputs")
        uploaded_pdf = st.file_uploader(
            "Optional insurance or benefits PDF",
            type=["pdf"],
            accept_multiple_files=False,
        )
        top_k = st.slider("Retrieved passages", min_value=1, max_value=10, value=5)

    query = st.text_area(
        "Travel disruption description",
        placeholder=(
            "Example: My flight on Example Air was delayed 7 hours and I paid "
            "with a Chase card. Can I claim hotel and meal expenses?"
        ),
        height=140,
    )

    if st.button("Analyze claim", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Enter a travel disruption description to run the analysis.")
            return

        try:
            if uploaded_pdf is not None:
                saved_path = _save_uploaded_pdf(uploaded_pdf, config)
                _ingest_uploaded_pdf(saved_path, config)

            analysis_service = ClaimAnalysisService(config=config)
            response = analysis_service.analyze_claim(user_query=query.strip(), top_k=top_k)
        except Exception as error:
            st.error(f"Analysis failed: {error}")
            return

        st.session_state.analysis_response = response
        st.session_state.analysis_query = query.strip()
        st.session_state.follow_up_answer = ""

    if st.session_state.analysis_response is not None:
        _render_response(st.session_state.analysis_response, st.session_state.analysis_query)


def _ensure_data_dirs(config: AppConfig) -> None:
    """Ensure the expected data directories exist."""
    (config.data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (config.data_dir / "processed").mkdir(parents=True, exist_ok=True)


def _save_uploaded_pdf(uploaded_pdf: st.runtime.uploaded_file_manager.UploadedFile, config: AppConfig) -> Path:
    """Persist an uploaded PDF into the raw data directory."""
    destination = config.data_dir / "raw" / uploaded_pdf.name
    destination.write_bytes(uploaded_pdf.getvalue())
    return destination


def _ingest_uploaded_pdf(pdf_path: Path, config: AppConfig) -> None:
    """Ingest an uploaded PDF and rebuild the local FAISS index."""
    pipeline = IngestPipeline(config=config)
    retriever = PassageRetriever(config=config)

    pipeline.ingest_pdf(
        pdf_path=pdf_path,
        source_type=_infer_source_type(pdf_path),
        provider_name=pdf_path.stem.replace("_", " ").replace("-", " ").title(),
        tags=["uploaded"],
    )
    retriever.index_chunks(
        chunks_path=config.data_dir / "processed" / "chunks.jsonl",
        persist_directory=config.data_dir / "processed" / "vector_store",
    )


def _infer_source_type(pdf_path: Path) -> object:
    """Infer the source type for an uploaded PDF from its filename."""
    from core.schemas import SourceType

    normalized = pdf_path.name.lower()
    if "card" in normalized or "credit" in normalized:
        return SourceType.CREDIT_CARD
    if "airline" in normalized or "flight" in normalized or "carrier" in normalized:
        return SourceType.AIRLINE
    return SourceType.INSURANCE


def _render_response(response: AnalysisResponse, query: str) -> None:
    """Render the analysis response in a readable layout."""
    st.success(_human_summary(response))

    left_column, right_column = st.columns([1.2, 1.0], gap="large")

    with left_column:
        st.subheader("Retrieved passages")
        if response.retrieved_passages:
            for index, passage in enumerate(response.retrieved_passages, start=1):
                _render_passage(index, passage)
        else:
            st.info("No passages were retrieved. Ingest a relevant PDF first.")

        st.subheader("Extracted requirements")
        _render_requirements(response)

    with right_column:
        st.subheader("Eligibility result")
        st.write(f"Outcome: **{_human_label(response)}**")
        reimbursement_cap = response.extracted_requirements.reimbursement_cap.value
        if reimbursement_cap is not None:
            st.write(f"Possible reimbursement cap: **{_format_money_value(reimbursement_cap)}**")
        st.write(f"Confidence: `{_confidence_text(response.eligibility.confidence)}`")
        if response.eligibility.rationale:
            st.markdown("**Why this result**")
            for item in _human_rationale(response):
                st.write(f"- {item}")
        if response.eligibility.supporting_chunk_ids:
            st.caption(
                "Supporting chunks: " + ", ".join(response.eligibility.supporting_chunk_ids)
            )

        st.subheader("Action checklist")
        checklist_items = _build_action_checklist(response)
        for item in checklist_items:
            st.write(f"- {item}")

        st.subheader("Draft message")
        airline_tab, card_tab = st.tabs(["Airline outreach draft", "Card benefits draft"])
        with airline_tab:
            st.text_area(
                "Draft airline message",
                value=_build_airline_outreach_draft(response, query),
                height=260,
                label_visibility="collapsed",
            )
        with card_tab:
            st.text_area(
                "Draft card benefits message",
                value=_build_card_outreach_draft(response, query),
                height=260,
                label_visibility="collapsed",
            )

        st.subheader("Follow-up questions")
        follow_up_question = st.text_input(
            "Ask about next steps, documents, deadlines, or how to contact the airline",
            key="follow_up_question",
        )
        if st.button("Answer follow-up", use_container_width=True):
            st.session_state.follow_up_answer = _answer_follow_up(follow_up_question, response)
        if st.session_state.follow_up_answer:
            st.markdown(st.session_state.follow_up_answer)


def _render_passage(index: int, passage: RetrievedPassage) -> None:
    """Render a single retrieved passage."""
    header = f"{index}. {passage.metadata.title} {passage.citation}"
    with st.expander(header, expanded=index == 1):
        st.caption(
            f"Source: {passage.metadata.source_type.value} | "
            f"Provider: {passage.metadata.provider_name} | "
            f"Score: {passage.relevance_score:.3f}"
        )
        st.write(passage.chunk.text)


def _render_requirements(response: AnalysisResponse) -> None:
    """Render extracted requirements in a more readable format."""
    requirements = response.extracted_requirements
    items = [
        ("Delay threshold", _format_scalar_value(requirements.minimum_delay_threshold_hours.value, "hours")),
        ("Covered expenses", _format_list_value(requirements.covered_expense_categories.value)),
        ("Reimbursement cap", _format_money_value(requirements.reimbursement_cap.value) if requirements.reimbursement_cap.value is not None else "Not clearly stated in the retrieved text"),
        ("Required documents", _format_list_value(requirements.required_documentation.value)),
        ("Filing deadline", _format_scalar_value(requirements.filing_deadline.value, "days")),
        ("Exclusions", _format_list_value(requirements.exclusions.value)),
    ]
    for label, value in items:
        if label in {"Delay threshold", "Reimbursement cap"} and value != "Not clearly stated in the retrieved text":
            st.write(f"**{label}: {value}**")
        else:
            st.write(f"**{label}:** {value}")


def _serialize_requirements(response: AnalysisResponse) -> dict[str, object]:
    """Convert extracted requirements into a display-friendly dictionary."""
    requirements = response.extracted_requirements
    return {
        "minimum_delay_threshold_hours": requirements.minimum_delay_threshold_hours.model_dump(mode="json"),
        "covered_expense_categories": requirements.covered_expense_categories.model_dump(mode="json"),
        "reimbursement_cap": requirements.reimbursement_cap.model_dump(mode="json"),
        "reimbursement_currency": requirements.reimbursement_currency.model_dump(mode="json"),
        "required_documentation": requirements.required_documentation.model_dump(mode="json"),
        "filing_deadline": requirements.filing_deadline.model_dump(mode="json"),
        "exclusions": requirements.exclusions.model_dump(mode="json"),
    }


def _build_action_checklist(response: AnalysisResponse) -> list[str]:
    """Build a flat checklist from the analysis response."""
    items: list[str] = []
    items.extend(response.claim_plan.recommended_actions)
    items.extend(response.claim_plan.follow_up_questions)
    items.extend(response.claim_plan.filing_steps)
    if not items:
        items.append("Review the cited passages and gather any supporting documents.")
    return [_humanize_checklist_item(item) for item in items]


def _build_draft_message(response: AnalysisResponse) -> str:
    """Generate a simple draft claim or inquiry message."""
    required_documents = _field_value_to_items(response.extracted_requirements.required_documentation)
    covered_categories = _field_value_to_items(response.extracted_requirements.covered_expense_categories)
    threshold = response.extracted_requirements.minimum_delay_threshold_hours.value
    reimbursement_cap = response.extracted_requirements.reimbursement_cap.value

    if response.eligibility.label.value == "Not Eligible":
        lines = [
            "Hello,",
            "",
            "I reviewed the travel delay terms that appear most relevant to this claim.",
            f"My current assessment is: {_human_label(response)}.",
            "",
            "Summary",
            f"- {_human_summary(response)}",
        ]
        if threshold is not None:
            lines.extend(
                [
                    "",
                    "Key point",
                    f"- The cited policy appears to require a delay of at least {threshold:.0f} hours.",
                ]
            )
        lines.extend(
            [
                "- Based on the facts provided, this claim may fall below that threshold.",
                "",
                "What I am asking you to confirm",
                "- Whether another applicable policy or exception should also be reviewed.",
                "- Whether there is any alternative reimbursement path available.",
            ]
        )
        lines.append("")
        lines.append("Thank you.")
        return "\n".join(lines)

    lines = [
        "Hello,",
        "",
        "I would like assistance with a travel disruption claim.",
        "",
        "Summary",
        f"- Current assessment: {_human_label(response)}",
        f"- {_human_summary(response)}",
        "",
    ]

    if covered_categories:
        lines.extend(_build_bullet_section("Expenses identified so far", covered_categories))
    if reimbursement_cap is not None:
        lines.extend(
            [
                "Possible reimbursement cap",
                f"- {_format_money_value(reimbursement_cap)}",
                "",
            ]
        )
    if required_documents:
        lines.extend(_build_bullet_section("Documents that appear relevant", required_documents))
    if response.claim_plan.follow_up_questions:
        lines.append("Questions to confirm")
        for question in response.claim_plan.follow_up_questions:
            lines.append(f"- {question}")
        lines.append("")

    lines.extend(
        [
            "Request",
            "- Please confirm the applicable policy terms and the next steps for filing.",
            "",
            "Thank you.",
        ]
    )
    return "\n".join(lines)


def _build_airline_outreach_draft(response: AnalysisResponse, query: str) -> str:
    """Generate a draft message for reaching out to an airline."""
    covered_categories = _field_value_to_items(response.extracted_requirements.covered_expense_categories)
    required_documents = _field_value_to_items(response.extracted_requirements.required_documentation)
    filing_deadline = response.extracted_requirements.filing_deadline.value
    expenses_text = _join_as_sentence(covered_categories)
    documents_text = _join_as_sentence(required_documents)

    lines = [
        "Hello,",
        "",
        "I am reaching out about a recent travel disruption and would appreciate your help reviewing whether reimbursement, compensation, or service recovery is available under your policy.",
        "",
        f"My trip details are as follows: {query}",
    ]
    if expenses_text:
        lines.extend(
            [
                "",
                f"I incurred out-of-pocket expenses related to {expenses_text}.",
            ]
        )
    if documents_text:
        lines.extend(
            [
                "",
                f"I can provide supporting materials such as {documents_text}.",
            ]
        )
    if filing_deadline:
        lines.extend(
            [
                "",
                f"I understand there may be a filing or notice deadline of {filing_deadline}, so I wanted to reach out promptly.",
            ]
        )
    lines.extend(
        [
            "",
            "Please let me know what assistance or reimbursement may be available in this situation, what documentation you would like me to submit, and the best next step for moving this request forward.",
            "",
            "Thank you.",
        ]
    )
    return "\n".join(lines)


def _build_card_outreach_draft(response: AnalysisResponse, query: str) -> str:
    """Generate a draft message for reaching out to a card issuer or benefits administrator."""
    covered_categories = _field_value_to_items(response.extracted_requirements.covered_expense_categories)
    required_documents = _field_value_to_items(response.extracted_requirements.required_documentation)
    filing_deadline = response.extracted_requirements.filing_deadline.value
    reimbursement_cap = response.extracted_requirements.reimbursement_cap.value
    expenses_text = _join_as_sentence(covered_categories)
    documents_text = _join_as_sentence(required_documents)

    lines = [
        "Hello,",
        "",
        "I am reaching out to ask for help reviewing whether this travel disruption is covered under my card benefits.",
        "",
        f"My trip details are as follows: {query}",
    ]
    if expenses_text:
        lines.extend(
            [
                "",
                f"The out-of-pocket expenses I am hoping to submit include {expenses_text}.",
            ]
        )
    if reimbursement_cap is not None:
        lines.extend(
            [
                "",
                f"My understanding is that the possible reimbursement cap may be {_format_money_value(reimbursement_cap)}.",
            ]
        )
    if documents_text:
        lines.extend(
            [
                "",
                f"I can provide supporting materials such as {documents_text}.",
            ]
        )
    if filing_deadline:
        lines.extend(
            [
                "",
                f"I also understand there may be a filing deadline of {filing_deadline}, so I wanted to contact you promptly.",
            ]
        )
    lines.extend(
        [
            "",
            "Please let me know whether this event is covered, what documents you would like me to submit, and how I should proceed with the claim.",
            "",
            "Thank you.",
        ]
    )
    return "\n".join(lines)


def _answer_follow_up(question: str, response: AnalysisResponse) -> str:
    """Answer simple follow-up questions from the current analysis context."""
    normalized = question.strip().lower()
    if not normalized:
        return "Enter a follow-up question about next steps, documents, deadlines, or how to contact the airline or card benefits administrator."

    if any(term in normalized for term in ("next step", "what should i do", "what now", "next")):
        return "\n".join([
            "- Gather your itinerary, receipts, and any airline delay notice.",
            "- Decide whether you want to contact the airline, the card benefits administrator, or both.",
            "- Use the draft in the UI and ask for the exact filing steps in writing.",
        ])

    if any(term in normalized for term in ("document", "receipt", "need")):
        documents = _field_value_to_text(response.extracted_requirements.required_documentation)
        if documents:
            return "\n".join([
                "- The most useful documents appear to be:",
                *[f"  - {item}" for item in _field_value_to_items(response.extracted_requirements.required_documentation)],
            ])
        return "\n".join([
            "- Keep your receipts.",
            "- Keep your itinerary details.",
            "- Keep any airline delay statement or baggage report.",
        ])

    if any(term in normalized for term in ("deadline", "when", "how long")):
        filing_deadline = response.extracted_requirements.filing_deadline.value
        if filing_deadline:
            return "\n".join([
                f"- The retrieved material mentions a filing deadline of {filing_deadline}.",
                "- Double-check the exact wording in the source before relying on it.",
            ])
        return "- I did not find a clear filing deadline in the retrieved passages."

    if any(term in normalized for term in ("airline", "contact", "email", "reach out")):
        return "\n".join([
            "- Use the airline outreach draft in the UI.",
            "- Attach your itinerary and receipts.",
            "- Ask the airline to confirm whether reimbursement, compensation, or service recovery is available.",
        ])

    if any(term in normalized for term in ("card", "benefit", "issuer", "administrator", "insurance")):
        return "\n".join([
            "- Use the card benefits draft in the UI.",
            "- Include your itinerary, card payment details, receipts, and the reason for the delay.",
            "- Ask the benefits administrator to confirm coverage and claim steps.",
        ])

    if any(term in normalized for term in ("eligible", "why")):
        rationale = " ".join(_human_rationale(response)[:3])
        return "\n".join([
            f"- Current result: {_human_label(response)}",
            f"- Main reasons: {rationale}",
        ])

    return "\n".join([
        "- Try asking about next steps.",
        "- You can also ask about needed documents, deadlines, or how to contact the airline or card benefits administrator.",
    ])


def _human_summary(response: AnalysisResponse) -> str:
    """Render a more natural summary for the current result."""
    summary = response.summary
    summary = summary.replace("Eligibility assessment:", "").strip()
    summary = summary.replace("via airline policy", "under the airline policy")
    summary = summary.replace("via benefit policy", "under the card or insurance benefit")
    summary = summary.replace("Secondary lane results:", "Other possible path:")
    return summary[:1].upper() + summary[1:] if summary else "Analysis complete."


def _human_label(response: AnalysisResponse) -> str:
    """Return a friendlier label for display."""
    if response.summary.lower().startswith("eligibility assessment: potentially eligible via airline policy"):
        return "Possibly eligible"
    label = response.eligibility.label.value
    if label == "Unclear":
        return "Needs more review"
    return label


def _confidence_text(confidence: float) -> str:
    """Convert a numeric confidence score to plain language."""
    if confidence >= 0.8:
        return "High"
    if confidence >= 0.6:
        return "Medium"
    return "Low"


def _human_rationale(response: AnalysisResponse) -> list[str]:
    """Turn internal rationale into simpler wording."""
    items: list[str] = []
    for item in response.eligibility.rationale:
        text = item
        text = text.replace("Retrieved policy threshold is", "The policy appears to require")
        text = text.replace("Policy exclusions were extracted and should be reviewed for disqualifying conditions.", "There are exclusions in the policy, so it is worth checking whether any of them apply.")
        text = text.replace("Filing deadline reference found:", "The retrieved material mentions a filing deadline of")
        text = text.replace("Required documentation identified:", "The policy appears to ask for")
        text = text.replace("Claimed delay meets or exceeds the explicit threshold", "Your stated delay appears to meet the policy threshold")
        text = text.replace("Claimed delay is below the explicit threshold", "Your stated delay appears to fall below the policy threshold")
        text = text.replace("Eligibility cannot be determined confidently because explicit policy requirements or user facts are missing.", "There is not enough clear information yet to make a confident call.")
        text = text.replace("Required information is missing, so the conservative classification is Unclear.", "Some important details are still missing, so the safest result is to treat this as uncertain for now.")
        items.append(text)
    return items


def _humanize_checklist_item(item: str) -> str:
    """Convert checklist language into a more conversational style."""
    mapping = {
        "Review the cited policy passages before filing the claim.": "Read the cited policy sections before you file anything.",
        "Resolve ambiguous policy wording with the issuer, insurer, or airline.": "If any policy wording is unclear, ask the airline, card issuer, or insurer to confirm it in writing.",
        "Collect the missing claim facts before relying on the analysis.": "Gather the missing trip details before relying on this result.",
        "Confirm the applicable policy or benefits guide.": "Make sure you are using the right airline policy or card benefits guide.",
        "Match the disruption facts to the cited policy thresholds and exclusions.": "Compare your delay, expenses, and trip details against the quoted rules and exclusions.",
        "Submit the claim with supporting evidence and keep copies of all records.": "Submit the request with supporting documents and keep copies of everything.",
    }
    return mapping.get(item, item)


def _format_scalar_value(value: object, suffix: str | None = None) -> str:
    """Format a scalar requirement value for display."""
    if value in (None, "", []):
        return "Not clearly stated in the retrieved text"
    if suffix:
        return f"{value} {suffix}"
    return str(value)


def _format_list_value(values: list[str]) -> str:
    """Format a list requirement value for display."""
    if not values:
        return "Not clearly stated in the retrieved text"
    return ", ".join(values)


def _field_value_to_text(field: StringListRequirementField | StringRequirementField) -> str:
    """Render extracted field values as readable text."""
    if isinstance(field, StringListRequirementField):
        return ", ".join(field.value)
    return field.value or ""


def _field_value_to_items(field: StringListRequirementField | StringRequirementField) -> list[str]:
    """Render extracted field values as list items for draft sections."""
    if isinstance(field, StringListRequirementField):
        return [item for item in field.value if item]
    value = field.value or ""
    return [value] if value else []


def _build_bullet_section(title: str, items: list[str]) -> list[str]:
    """Build a draft section with a heading and bullet points."""
    if not items:
        return []
    lines = [title]
    for item in items:
        lines.append(f"- {item}")
    lines.append("")
    return lines


def _format_money_value(value: object) -> str:
    """Format a possible reimbursement cap for human-readable drafts."""
    if isinstance(value, (int, float)):
        return f"${value:,.0f}"
    return str(value)


def _join_as_sentence(items: list[str]) -> str:
    """Join list items into a readable sentence fragment."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


if __name__ == "__main__":
    main()
