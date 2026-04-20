"""
graph.py — LangGraph StateGraph (unified state).

prepare_react_node → react_agent_graph (create_agent subgraph) → answer_node →
synthesize_node → update_summary_node

prepare_react_node resets the per-turn ReAct ``messages`` channel and clears
stale retrieval/output fields before the subgraph runs. create_agent owns the
model↔tools loop.
"""

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from slack_bot.agent.nodes.answer import answer_node
from slack_bot.agent.nodes.prepare_react import prepare_react_node
from slack_bot.agent.nodes.synthesize import synthesize_node
from slack_bot.agent.nodes.update_summary import update_summary_node
from slack_bot.agent.types import UnifiedGraphState


def build_graph(
    checkpointer: AsyncSqliteSaver | None = None,
) -> CompiledStateGraph:
    """Compile the Slack bot LangGraph pipeline.

    Order: prepare_react → react_agent_graph → answer → synthesize →
    update_summary.

    Args:
        checkpointer (AsyncSqliteSaver | None): Async SQLite saver for thread
            checkpoints, or ``None`` for ephemeral runs.

    Returns:
        CompiledStateGraph: Runnable graph (invoke / astream / astream_events).

    """
    from slack_bot.startup import CTX

    builder = StateGraph(UnifiedGraphState)

    builder.add_node("prepare_react_node", prepare_react_node)
    builder.add_node("react_agent_graph", CTX.react_agent_graph)
    builder.add_node("answer_node", answer_node)
    builder.add_node("synthesize_node", synthesize_node)
    builder.add_node("update_summary_node", update_summary_node)

    builder.add_edge(START, "prepare_react_node")
    builder.add_edge("prepare_react_node", "react_agent_graph")
    builder.add_edge("react_agent_graph", "answer_node")
    builder.add_edge("answer_node", "synthesize_node")
    builder.add_edge("synthesize_node", "update_summary_node")
    builder.add_edge("update_summary_node", END)

    return builder.compile(checkpointer=checkpointer)
