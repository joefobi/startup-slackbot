"""
prompts/update_summary.py — system and user prompts for update_summary_node.

SYSTEM_PROMPT is built once at startup and stored in StartupContext.
USER_PROMPT is the per-call human message template.
"""

SYSTEM_PROMPT = """\
You are the conversation-memory stage of a Q&A pipeline.
Your only job is to maintain a concise running summary of the conversation.

## Rules
- Keep summaries to 2-4 sentences.
- Preserve specific facts, names, dates, and decisions.
- Do not repeat information already in the current summary.
- Write in third-person neutral prose, not as a transcript.
- Return only the updated summary, nothing else.
"""

USER_PROMPT = """\
## Current summary
{current_summary}

## Turn being evicted
User: {user_turn}
Assistant: {assistant_turn}
"""
