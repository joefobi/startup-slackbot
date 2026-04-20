"""
answer_node — grounded answer + ReAct metrics from the same state snapshot.

Computes react_iterations / tool_call_count from ``messages`` (cheap, no extra
LLM). Single LLM call for draft_answer + confidence.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from slack_bot.prompts.answer import USER_PROMPT as ANSWER_PROMPT
from slack_bot.agent.types import AnswerOutput, ArtifactResult, UnifiedGraphState
from slack_bot.startup import CTX


def _rows_to_artifacts(rows: list) -> list[ArtifactResult]:
    """Normalize mixed dict / model rows to ``ArtifactResult`` instances.

    Args:
        rows (list): Items from ``fts_results`` (dicts or ``ArtifactResult``).

    Returns:
        list[ArtifactResult]: Parsed models; skips unsupported entries.

    """
    out: list[ArtifactResult] = []
    for r in rows or []:
        if isinstance(r, ArtifactResult):
            out.append(r)
        elif isinstance(r, dict):
            out.append(ArtifactResult(**r))
    return out


def _format_fts(results: list) -> str:
    """Render FTS hits as readable text for the answer LLM.

    Args:
        results (list): ``fts_results`` rows (dict or ``ArtifactResult``).

    Returns:
        str: Multi-artifact text block, or ``(none)`` if empty.

    """
    models = _rows_to_artifacts(results)
    if not models:
        return "(none)"
    lines = []
    for r in models:
        lines.append(
            f"[{r.artifact_id}] {r.artifact_type} — "
            f"{r.title}\n{r.summary}"
        )
    return "\n\n".join(lines)


def _format_sql(results: list[dict]) -> str:
    """Serialize SQL result rows as indented JSON for the answer prompt.

    Args:
        results (list[dict]): Rows from ``sql_results``.

    Returns:
        str: JSON string or ``(none)`` when there are no rows.

    """
    if not results:
        return "(none)"
    return json.dumps(results, indent=2, default=str)


def answer_node(state: UnifiedGraphState) -> dict:
    """Produce a grounded answer and ReAct loop metrics for evals.

    Counts tool calls and assistant rounds from ``messages``, then invokes the
    answer model with structured output (answer + confidence).

    Args:
        state (UnifiedGraphState): State after the ReAct subgraph, including
            ``messages``, ``question``, ``fts_results``, ``sql_results``, and
            ``full_artifact``.

    Returns:
        dict: Keys ``draft_answer``, ``confidence``, ``react_iterations``, and
            ``tool_call_count``.

    """
    msgs = state.get("messages") or []
    tool_call_count = 0
    for m in msgs:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call_count += len(m.tool_calls)
    react_iterations = sum(1 for m in msgs if isinstance(m, AIMessage))

    prompt = ANSWER_PROMPT.format(
        question=state.get("question") or "",
        fts_results=_format_fts(state.get("fts_results") or []),
        sql_results=_format_sql(state.get("sql_results") or []),
        full_artifact=state.get("full_artifact") or "(not loaded)",
    )

    messages = [
        SystemMessage(content=CTX.answer_prompt),
        HumanMessage(content=prompt),
    ]

    result: AnswerOutput = CTX.llm.with_structured_output(AnswerOutput).invoke(messages)

    return {
        "draft_answer": result.answer,
        "confidence": result.confidence,
        "react_iterations": react_iterations,
        "tool_call_count": tool_call_count,
    }
