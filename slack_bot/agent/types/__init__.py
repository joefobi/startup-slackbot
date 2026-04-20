"""
Type definitions for the agent: Pydantic domain models + LangGraph graph state.

Import from here in new code::

    from slack_bot.agent.types import ArtifactResult, AnswerOutput, UnifiedGraphState
"""

from slack_bot.agent.types.domain import AnswerOutput, ArtifactResult
from slack_bot.agent.types.graph_state import UnifiedGraphState

__all__ = ["AnswerOutput", "ArtifactResult", "UnifiedGraphState"]
