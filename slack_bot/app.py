"""
app.py — FastAPI application, Slack event handler, security validation.

Security:
  - HMAC-SHA256 signature validated on every incoming request before any
    LangGraph invocation. Uses hmac.compare_digest (constant-time) to
    prevent timing attacks.
  - 5-minute timestamp window prevents replay attacks.
  - Event-ID deduplication cache (TTL 10 min) prevents double-processing.
  - All secrets loaded from environment variables, never hardcoded.

UX:
  - Handler returns HTTP 200 within the 3-second Slack deadline.
  - Agent runs as an asyncio background task.
  - :thinking_face: reaction posted immediately so the user sees activity.
  - If the agent takes > 8 seconds, a "Working on it..." placeholder is
    posted and later updated in-place with the real answer.
  - Answers longer than SLACK_CHAR_LIMIT are split and posted as follow-ups
    in the same thread.

Threading:
  - thread_ts is used as the LangGraph checkpointer thread_id so each Slack
    thread gets its own isolated conversation state.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from slack_sdk.web.async_client import AsyncWebClient

from slack_bot.agent.graph import build_graph
from slack_bot.config import (
    EVENT_DEDUP_TTL,
    MAX_REACT_ITERATIONS,
    SLACK_CHAR_LIMIT,
    STATUS_UPDATE_MIN_INTERVAL,
)

_GRAPH_RECURSION_LIMIT = 2 * MAX_REACT_ITERATIONS + 12

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SLACK_SIGNING_SECRET: str = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_TOKEN: str = os.environ["SLACK_BOT_TOKEN"]

# Human-readable status for outer LangGraph nodes (astream_events v2 uses
# metadata["langgraph_node"] on on_chain_* events).
_OUTER_NODE_STATUS: dict[str, str] = {
    "prepare_react_node": "_Gathering sources..._",
    "react_agent_graph": "_Gathering sources..._",
    "answer_node": "_Drafting answer..._",
    "synthesize_node": "_Formatting response..._",
}

# create_agent inner tools — surfaced via on_tool_start (runnable name).
_TOOL_START_STATUS: dict[str, str] = {
    "hybrid_search": "_Searching documents..._",
    "run_sql": "_Querying database..._",
}


def _slack_status_from_stream_event(ev: dict[str, Any]) -> str | None:
    """
    Map a single astream_events (v2) payload to a short Slack status, or None.

    Uses on_tool_start for hybrid_search / run_sql inside create_agent, and
    on_chain_start for top-level graph nodes (langgraph_node metadata).
    """
    et = ev.get("event") or ""
    meta = ev.get("metadata") or {}
    lg_node = meta.get("langgraph_node")
    if isinstance(lg_node, str) and lg_node:
        pass
    else:
        lg_node = ""

    if et == "on_tool_start":
        name = str(ev.get("name") or "")
        for tool_key, label in _TOOL_START_STATUS.items():
            if tool_key == name or name.endswith(tool_key):
                return label
        return None

    if et == "on_chain_start" and lg_node in _OUTER_NODE_STATUS:
        return _OUTER_NODE_STATUS[lg_node]

    return None


def _extract_final_answer_from_event_output(output: Any) -> str | None:
    """Best-effort: find final_answer in on_chain_end data.output (dict)."""
    if output is None:
        return None
    if hasattr(output, "final_answer"):
        fa = getattr(output, "final_answer", None)
        if isinstance(fa, str) and fa.strip():
            return fa
    if isinstance(output, dict):
        fa = output.get("final_answer")
        if isinstance(fa, str) and fa.strip():
            return fa
        for v in output.values():
            found = _extract_final_answer_from_event_output(v)
            if found:
                return found
    if isinstance(output, (list, tuple)):
        for item in output:
            found = _extract_final_answer_from_event_output(item)
            if found:
                return found
    return None


def _slack_delivery_chunks(text: str, max_chars: int = SLACK_CHAR_LIMIT) -> list[str]:
    """Split text into segments safe for ``chat.update`` / ``chat.postMessage``.

    Slack returns ``msg_too_long`` when a single ``text`` payload exceeds its
    limit (~4k). We first honor ``[CONTINUED]`` from ``synthesize_node``, then
    hard-chunk any segment that still exceeds ``max_chars``.

    Args:
        text (str): Full ``final_answer`` string.
        max_chars (int): Upper bound per Slack message. Defaults to
            ``SLACK_CHAR_LIMIT``.

    Returns:
        list[str]: Non-empty chunks to post in order.

    """
    parts: list[str] = []
    for segment in text.split("[CONTINUED]"):
        segment = segment.strip()
        if not segment:
            continue
        while segment:
            if len(segment) <= max_chars:
                parts.append(segment)
                break
            head = segment[:max_chars]
            cut = head.rfind("\n\n")
            if cut < max_chars // 2:
                cut = head.rfind("\n")
            if cut < max_chars // 2:
                cut = head.rfind(" ")
            if cut < max_chars // 2:
                cut = max_chars
            chunk = segment[:cut].strip()
            if not chunk:
                chunk = segment[:max_chars]
                segment = segment[max_chars:].strip()
            else:
                segment = segment[cut:].strip()
            parts.append(chunk)
    return parts


# ---------------------------------------------------------------------------
# TTL deduplication cache — prevents double-processing retried Slack events
# ---------------------------------------------------------------------------

class _TTLSet:
    def __init__(self, ttl: int = EVENT_DEDUP_TTL) -> None:
        self._store: dict[str, float] = {}
        self._ttl = ttl

    def contains(self, key: str) -> bool:
        self._evict()
        return key in self._store

    def add(self, key: str) -> None:
        self._store[key] = time.monotonic()

    def _evict(self) -> None:
        cutoff = time.monotonic() - self._ttl
        expired = [k for k, t in self._store.items() if t < cutoff]
        for k in expired:
            del self._store[k]


_seen_events = _TTLSet()


# ---------------------------------------------------------------------------
# Slack signature validation
# ---------------------------------------------------------------------------

def _validate_slack_signature(
    body: bytes,
    timestamp: str,
    signature: str,
) -> None:
    """
    Verify the request came from Slack using HMAC-SHA256.
    Raises HTTP 403 on any failure — before any business logic runs.
    """
    try:
        request_ts = int(timestamp)
    except (ValueError, TypeError):
        raise HTTPException(status_code=403, detail="Invalid timestamp header.")

    if abs(time.time() - request_ts) > 300:
        raise HTTPException(status_code=403, detail="Request timestamp too old.")

    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    expected = (
        "v0="
        + hmac.new(
            SLACK_SIGNING_SECRET.encode("utf-8"),
            sig_basestring.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    )

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=403, detail="Invalid Slack signature.")


# ---------------------------------------------------------------------------
# Slack client
# ---------------------------------------------------------------------------

slack = AsyncWebClient(token=SLACK_BOT_TOKEN)

# Set during lifespan startup — available to all background tasks
_graph = None


# ---------------------------------------------------------------------------
# Agent runner — background task
# ---------------------------------------------------------------------------

async def _run_agent(
    channel: str,
    thread_ts: str,
    msg_ts: str,
    user_message: str,
) -> None:
    """
    Stream the LangGraph agent and post live status updates to Slack.
    Runs as an asyncio background task so the handler can return 200 fast.

    Flow:
      1. Post :thinking_face: reaction immediately.
      2. Post a placeholder message immediately (no delay).
      3. Stream astream_events (v2) — placeholder text from outer node starts
         and inner tool starts; throttled to one Slack update per second.
      4. Replace the placeholder with the final answer when the graph ends.
    """
    final_answer: str = "Sorry, I couldn't find an answer."
    placeholder_ts: str | None = None

    try:
        # Step 1 — immediate reaction so the user sees activity at once
        await slack.reactions_add(
            channel=channel, timestamp=msg_ts, name="thinking_face"
        )

        # Step 2 — post placeholder immediately; we'll update it in-place
        try:
            resp = await slack.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text="_Thinking..._",
            )
            placeholder_ts = resp["ts"]
        except Exception:
            logger.exception("Failed to post placeholder message.")

        # Step 3 — stream graph events (v2) for finer-grained Slack status:
        # outer nodes via on_chain_start + langgraph_node; inner tools via on_tool_start.
        config = {
            "configurable": {"thread_id": thread_ts},
            "recursion_limit": _GRAPH_RECURSION_LIMIT,
        }
        last_update_time: float = 0.0

        async for ev in _graph.astream_events(
            {"question": user_message},
            config=config,
            version="v2",
        ):
            if not isinstance(ev, dict):
                continue

            if ev.get("event") == "on_chain_end":
                out = (ev.get("data") or {}).get("output")
                found = _extract_final_answer_from_event_output(out)
                if found:
                    final_answer = found

            status = _slack_status_from_stream_event(ev)
            if status and placeholder_ts:
                now = time.monotonic()
                if now - last_update_time >= STATUS_UPDATE_MIN_INTERVAL:
                    try:
                        await slack.chat_update(
                            channel=channel,
                            ts=placeholder_ts,
                            text=status,
                        )
                        last_update_time = now
                    except Exception:
                        logger.warning(
                            "Failed to update status for event %s name=%s",
                            ev.get("event"),
                            ev.get("name"),
                        )

    except Exception:
        logger.exception("Agent invocation failed for thread %s", thread_ts)
        final_answer = (
            "Sorry, something went wrong while processing your question. "
            "Please try again."
        )
    finally:
        try:
            await slack.reactions_remove(
                channel=channel, timestamp=msg_ts, name="thinking_face"
            )
        except Exception:
            pass  # reaction may already be gone — not fatal

    # Step 4 — replace the placeholder with the real answer.
    # Honor [CONTINUED] from synthesize_node, then enforce Slack length limits
    # so chat.update never hits msg_too_long if the model omits markers.
    parts = _slack_delivery_chunks(final_answer)
    if not parts:
        parts = [final_answer[:SLACK_CHAR_LIMIT] or "_(empty reply)_"]

    for i, part in enumerate(parts):
        if i == 0 and placeholder_ts:
            # Update the placeholder in-place with the first chunk
            await slack.chat_update(
                channel=channel, ts=placeholder_ts, text=part
            )
        else:
            await slack.chat_postMessage(
                channel=channel, thread_ts=thread_ts, text=part
            )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    checkpoint_path = os.environ.get("CHECKPOINT_DB_PATH", "checkpoints.db")
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
        _graph = build_graph(checkpointer=saver)
        logger.info("Slack Q&A bot ready.")
        yield
    logger.info("Slack Q&A bot shut down.")


app = FastAPI(title="Slack Q&A Bot", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/slack/events")
async def slack_events(request: Request) -> Response:
    body = await request.body()

    # 1. Validate signature — before touching any payload
    _validate_slack_signature(
        body=body,
        timestamp=request.headers.get("X-Slack-Request-Timestamp", ""),
        signature=request.headers.get("X-Slack-Signature", ""),
    )

    payload = json.loads(body)

    # 2. URL verification handshake (one-time during app setup)
    if payload.get("type") == "url_verification":
        return Response(
            content=json.dumps({"challenge": payload["challenge"]}),
            media_type="application/json",
        )

    # 3. Deduplicate — Slack retries events that don't get a fast 200
    event_id = payload.get("event_id", "")
    if _seen_events.contains(event_id):
        return Response(content=json.dumps({"ok": True}), media_type="application/json")
    _seen_events.add(event_id)

    event = payload.get("event", {})

    # 4. Ignore messages from bots (including our own replies)
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return Response(content=json.dumps({"ok": True}), media_type="application/json")

    # 5. Only handle message and app_mention events
    if event.get("type") not in ("message", "app_mention"):
        return Response(content=json.dumps({"ok": True}), media_type="application/json")

    user_message: str = event.get("text", "").strip()
    if not user_message:
        return Response(content=json.dumps({"ok": True}), media_type="application/json")

    channel: str = event["channel"]
    msg_ts: str = event["ts"]
    # Use thread_ts if this is already a threaded reply; otherwise start a new thread
    thread_ts: str = event.get("thread_ts", msg_ts)

    # 6. Dispatch agent as background task — return 200 immediately
    asyncio.create_task(
        _run_agent(
            channel=channel,
            thread_ts=thread_ts,
            msg_ts=msg_ts,
            user_message=user_message,
        )
    )

    return Response(content=json.dumps({"ok": True}), media_type="application/json")
