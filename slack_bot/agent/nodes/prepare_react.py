"""
prepare_react_node — runs **before** the create_agent subgraph each Slack turn.

Why it exists as its own node (not inside create_agent):

- The subgraph only sees ``messages`` + tool state; it does not know when a *new*
  user question started or that ``summary`` / ``recent_turns`` from the outer
  graph should become this turn's USER_PROMPT.
- We must **replace** the ReAct transcript each turn (``RemoveMessage`` +
  fresh ``HumanMessage``) so prior turns' tool traces do not accumulate in
  ``messages`` while ``recent_turns`` holds the real conversation window.
- Same update clears stale fts/sql/draft fields so retrieval is always for this
  question only.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage

from slack_bot.agent.types import UnifiedGraphState
from slack_bot.prompts.agent import USER_PROMPT


def prepare_react_node(state: UnifiedGraphState) -> dict:
    """Reset ReAct messages and retrieval fields for the current Slack turn.

    Clears the prior tool transcript, injects ``summary`` / ``recent_turns`` /
    ``question`` into a fresh user message, and zeroes stale FTS/SQL/draft state
    so retrieval applies only to this question.

    Args:
        state (UnifiedGraphState): Checkpointed graph state at turn start.

    Returns:
        dict: State delta for ``messages`` (remove-all + HumanMessage) and
            cleared retrieval/output keys for the subgraph.

    """
    recent_turns = state.get("recent_turns") or []
    recent_turns_text = (
        "\n".join(
            f"{'User' if isinstance(t, HumanMessage) else 'Assistant'}: {t.content}"
            for t in recent_turns
        )
        or "(none yet)"
    )

    user_content = USER_PROMPT.format(
        summary=state.get("summary") or "(none yet)",
        recent_turns=recent_turns_text,
        question=state.get("question") or "",
    )

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            HumanMessage(content=user_content),
        ],
        "fts_results": [],
        "sql_results": [],
        "bm25_hits_debug": None,
        "vec_hits_debug": None,
        "full_artifact": None,
        "sql_aborted": False,
        "generated_sql": "",
        "draft_answer": "",
        "confidence": "",
        "final_answer": "",
        "needs_fts": False,
        "needs_sql": False,
        "fts_queries": [],
        "sql_intent": "",
        "relevant_tables": [],
        "artifact_types": [],
        "recency_sensitive": False,
        "replan_attempted": False,
        "validation_passed": False,
        "validation_error": None,
        "retry_count": 0,
        "react_iterations": 0,
        "tool_call_count": 0,
    }
