"""LangGraph checkpoint schema (TypedDict) for the full Slack bot graph."""

from __future__ import annotations

from typing import Any, NotRequired

from langchain.agents.middleware import AgentState as LCAgentState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class UnifiedGraphState(LCAgentState[Any]):
    """Checkpoint schema for the outer LangGraph plus the ReAct subgraph.

    Extends LangChain agent state (``messages``, etc.) with Slack memory,
    retrieval fields, SQL outputs, and formatted answers.

    """

    summary: NotRequired[str]
    recent_turns: NotRequired[Annotated[list[AnyMessage], add_messages]]
    question: NotRequired[str]

    artifact_types: NotRequired[list[str]]
    recency_sensitive: NotRequired[bool]
    needs_fts: NotRequired[bool]
    needs_sql: NotRequired[bool]
    fts_queries: NotRequired[list[str]]
    sql_intent: NotRequired[str]
    relevant_tables: NotRequired[list[str]]

    fts_results: NotRequired[list[dict[str, Any]]]
    bm25_hits_debug: NotRequired[list[dict[str, Any]] | None]
    vec_hits_debug: NotRequired[list[dict[str, Any]] | None]
    generated_sql: NotRequired[str]
    validation_passed: NotRequired[bool]
    validation_error: NotRequired[str | None]
    retry_count: NotRequired[int]
    sql_aborted: NotRequired[bool]
    sql_results: NotRequired[list[dict[str, Any]]]
    full_artifact: NotRequired[str | None]

    react_iterations: NotRequired[int]
    tool_call_count: NotRequired[int]

    draft_answer: NotRequired[str]
    confidence: NotRequired[str]
    replan_attempted: NotRequired[bool]
    final_answer: NotRequired[str]
