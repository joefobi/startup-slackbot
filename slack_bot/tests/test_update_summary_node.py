"""
Tests for update_summary_node.

Covers the eviction + compression logic without caring about LLM output
content. LLM calls are mocked; the logic under test is:
  - append current turn as HumanMessage + AIMessage
  - evict oldest pair via RemoveMessage when window is full
  - call LLM only when eviction occurs
  - return only the delta (removals + new messages), not the full window
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from slack_bot.agent.nodes.update_summary import update_summary_node
from slack_bot.config import VERBATIM_WINDOW


def _turns(n: int) -> list:
    """Build a list of n alternating HumanMessage/AIMessage objects."""
    turns = []
    for i in range(n // 2):
        turns.append(HumanMessage(content=f"question {i}"))
        turns.append(AIMessage(content=f"answer {i}"))
    return turns


def _state(
    recent_turns: list,
    summary: str = "",
    question: str = "new q",
    final_answer: str = "new a",
) -> dict:
    return {
        "messages": [],
        "question": question,
        "final_answer": final_answer,
        "recent_turns": recent_turns,
        "summary": summary,
    }


def _mock_llm_response(content: str = "updated summary"):
    mock = MagicMock()
    mock.content = content
    return mock


# ---------------------------------------------------------------------------
# No eviction (window not full after appending)
# ---------------------------------------------------------------------------


def test_window_not_full_no_llm_call():
    """
    Starting with VERBATIM_WINDOW - 2 turns: after appending the current turn
    the window is exactly full — no eviction, no LLM call.
    """
    initial = _turns(VERBATIM_WINDOW - 2)  # 6 items
    state = _state(initial)

    with patch("slack_bot.agent.nodes.update_summary.CTX") as mock_ctx:
        result = update_summary_node(state)
        mock_ctx.llm.invoke.assert_not_called()

    # No eviction: node returns only the 2 new messages (no RemoveMessages)
    turns = result["recent_turns"]
    assert not any(isinstance(t, RemoveMessage) for t in turns)
    assert len(turns) == 2


def test_window_not_full_appends_current_turn():
    initial = _turns(VERBATIM_WINDOW - 4)  # 4 items
    state = _state(initial, question="my question", final_answer="my answer")

    with patch("slack_bot.agent.nodes.update_summary.CTX"):
        result = update_summary_node(state)

    turns = result["recent_turns"]
    assert turns[-2].content == "my question"
    assert turns[-1].content == "my answer"
    assert isinstance(turns[-2], HumanMessage)
    assert isinstance(turns[-1], AIMessage)


# ---------------------------------------------------------------------------
# Eviction (window full after appending)
# ---------------------------------------------------------------------------


def test_eviction_triggers_llm_call():
    """Starting with VERBATIM_WINDOW items: appending pushes over the limit."""
    initial = _turns(VERBATIM_WINDOW)  # 8 items → after append = 10 > 8
    state = _state(initial)

    with patch("slack_bot.agent.nodes.update_summary.CTX") as mock_ctx:
        mock_ctx.llm.invoke.return_value = _mock_llm_response("new summary")
        result = update_summary_node(state)
        mock_ctx.llm.invoke.assert_called_once()

    assert result["summary"] == "new summary"


def test_eviction_removes_oldest_pair():
    oldest_user = HumanMessage(content="OLDEST QUESTION")
    oldest_asst = AIMessage(content="OLDEST ANSWER")
    rest = _turns(VERBATIM_WINDOW - 2)
    initial = [oldest_user, oldest_asst] + rest

    state = _state(initial)

    with patch("slack_bot.agent.nodes.update_summary.CTX") as mock_ctx:
        mock_ctx.llm.invoke.return_value = _mock_llm_response()
        result = update_summary_node(state)

    turns = result["recent_turns"]
    removals = [t for t in turns if isinstance(t, RemoveMessage)]
    removed_ids = {r.id for r in removals}
    assert oldest_user.id in removed_ids
    assert oldest_asst.id in removed_ids


def test_eviction_returns_removals_plus_new_turn():
    """After eviction the node returns 2 RemoveMessages + 2 new messages."""
    initial = _turns(VERBATIM_WINDOW)
    state = _state(initial)

    with patch("slack_bot.agent.nodes.update_summary.CTX") as mock_ctx:
        mock_ctx.llm.invoke.return_value = _mock_llm_response()
        result = update_summary_node(state)

    turns = result["recent_turns"]
    removals = [t for t in turns if isinstance(t, RemoveMessage)]
    new_msgs = [t for t in turns if not isinstance(t, RemoveMessage)]
    assert len(removals) == 2
    assert len(new_msgs) == 2


def test_evicted_content_passed_to_llm():
    """The evicted turn content should appear in the LLM prompt."""
    oldest_user = HumanMessage(content="evicted question")
    oldest_asst = AIMessage(content="evicted answer")
    initial = [oldest_user, oldest_asst] + _turns(VERBATIM_WINDOW - 2)
    state = _state(initial, summary="existing summary")

    with patch("slack_bot.agent.nodes.update_summary.CTX") as mock_ctx:
        mock_ctx.llm.invoke.return_value = _mock_llm_response()
        update_summary_node(state)

    call_args = mock_ctx.llm.invoke.call_args
    # The human message content should contain the evicted turn text
    messages = call_args[0][0]
    human_content = messages[1].content  # HumanMessage is second
    assert "evicted question" in human_content
    assert "evicted answer" in human_content
    assert "existing summary" in human_content
