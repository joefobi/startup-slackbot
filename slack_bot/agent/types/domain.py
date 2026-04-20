"""Pydantic domain models shared across nodes, tools, and prompts."""

from pydantic import BaseModel, ConfigDict, Field


class ArtifactResult(BaseModel):
    """One hybrid-search hit (BM25 and/or vector) stored in ``fts_results``.

    ``rank`` is set for BM25 rows and omitted for vector-only hydrations.

    """

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    artifact_type: str
    title: str
    summary: str
    created_at: str | None = None
    customer_id: str | None = None
    scenario_id: str | None = None
    product_id: str | None = None
    competitor_id: str | None = None
    token_estimate: int | None = None
    metadata_json: str | None = None
    rank: float | None = None


class AnswerOutput(BaseModel):
    """Structured LLM output from ``answer_node`` (answer + confidence + cites)."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        description=(
            "A factual answer grounded entirely in the retrieved sources. "
            "Every claim must be attributable to fts_results, sql_results, or full_artifact."
        )
    )
    confidence: str = Field(
        description=(
            "Overall confidence in the answer. "
            "'high': all key facts are grounded and consistent. "
            "'medium': most facts are grounded, minor gaps. "
            "'low': significant facts could not be grounded or sources contradict."
        )
    )
    citations: list[str] = Field(
        default_factory=list,
        description=(
            "List of source references used: artifact_id values or "
            "'table:tablename row:identifier' strings."
        ),
    )
