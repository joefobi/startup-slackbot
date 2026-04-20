"""
update_summary_node — append the current turn and evict the oldest if needed.

Uses unified state: recent_turns + summary with add_messages reducer.
"""

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from slack_bot.prompts.update_summary import USER_PROMPT as UPDATE_SUMMARY_PROMPT
from slack_bot.agent.types import UnifiedGraphState
from slack_bot.config import VERBATIM_WINDOW
from slack_bot.startup import CTX


def update_summary_node(state: UnifiedGraphState) -> dict:
    """Append this turn to ``recent_turns`` and compress history when full.

    When over ``VERBATIM_WINDOW``, rolls the oldest user/assistant pair into
    ``summary`` via the summary LLM and removes those messages.

    Args:
        state (UnifiedGraphState): Needs ``question``, ``final_answer``,
            ``recent_turns``, and ``summary``.

    Returns:
        dict: Delta with ``summary`` and ``recent_turns`` updates (including
            ``RemoveMessage`` entries when evicting).

    """
    existing: list = list(state.get("recent_turns") or [])
    current_summary: str = state.get("summary") or ""

    new_turn = [
        HumanMessage(content=state.get("question") or ""),
        AIMessage(content=state.get("final_answer") or ""),
    ]

    all_messages = existing + new_turn

    if len(all_messages) > VERBATIM_WINDOW:
        evicted_user, evicted_assistant = all_messages[0], all_messages[1]

        prompt = UPDATE_SUMMARY_PROMPT.format(
            current_summary=current_summary or "(none yet)",
            user_turn=evicted_user.content,
            assistant_turn=evicted_assistant.content,
        )

        llm_messages = [
            SystemMessage(content=CTX.summary_prompt),
            HumanMessage(content=prompt),
        ]

        result = CTX.llm.invoke(llm_messages)
        current_summary = result.content.strip()

        return {
            "summary": current_summary,
            "recent_turns": [
                RemoveMessage(id=evicted_user.id),
                RemoveMessage(id=evicted_assistant.id),
            ]
            + new_turn,
        }

    return {
        "summary": current_summary,
        "recent_turns": new_turn,
    }
