"""
synthesize_node — format the verified answer as a Slack markdown message.

Single LLM call. Handles Slack markdown, citations, confidence / sql_aborted,
and long-answer splitting via [CONTINUED].
"""

from langchain_core.messages import HumanMessage, SystemMessage

from slack_bot.prompts.synthesize import USER_PROMPT as SYNTHESIZE_PROMPT
from slack_bot.agent.types import UnifiedGraphState
from slack_bot.startup import CTX


def synthesize_node(state: UnifiedGraphState) -> dict:
    """Turn the draft answer into Slack-ready markdown.

    Applies formatting rules (length, tone, optional ``[CONTINUED]`` handling)
    via a single LLM call.

    Args:
        state (UnifiedGraphState): Must include ``draft_answer``, ``confidence``,
            and ``sql_aborted``.

    Returns:
        dict: Mapping ``final_answer`` to the string posted to Slack.

    """
    prompt = SYNTHESIZE_PROMPT.format(
        verified_answer=state.get("draft_answer") or "",
        citations="",
        confidence=state.get("confidence") or "medium",
        sql_aborted=state.get("sql_aborted"),
    )

    messages = [
        SystemMessage(content=CTX.synthesize_prompt),
        HumanMessage(content=prompt),
    ]

    result = CTX.llm.invoke(messages)
    final_answer = result.content.strip()

    return {"final_answer": final_answer}
